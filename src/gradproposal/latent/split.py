"""Split-information probe: does a latent cache carry facts the receiver lacks?

Two agents, one model. The sender holds a GSM8K problem's facts (every sentence but the last); the
solver holds only the final question sentence, so it cannot answer alone. Arms:

  latent   sender reads the facts, takes m latent steps; the solver continues from its cache
  wrong    as latent, but the cache was built from another question's facts (its partner)
  nocache  the solver alone (same prompt as latent): the floor
  text     sender writes a message (thinking on); the solver reads the visible part in its prompt
  latentonly  as latent, but the solver sees only the m latent positions (plus the sender's first
              token as an attention sink); the fact-bearing prompt positions are masked out
  wrongonly   latentonly on the partner's cache: its control

    python -m gradproposal.latent.split --conditions latent,wrong,nocache,text \
        --out latent/results/split-probe/split-gsm8k.jsonl
    python -m gradproposal.latent.split --report latent/results/split-probe/split-gsm8k.jsonl

Runs log to W&B (hosted cloud only; see gradproposal.tracking) and must write into
latent/results/split-probe/ (latent/results/dev/ for smoke runs). --report reads a finished file,
runs no model and needs no W&B.

Pre-registered kill criterion (NOTEBOOK.md): if latent - wrong < 10 points, the latent channel
is not carrying the facts usefully at m = 40.

Compute per row: estimated forward FLOPs per stage (sender, solver prompt, solver answer), message
size (KV-cache bytes or text tokens and bytes) and measured seconds per stage. Seconds are batch
wall-clock / batch size; run with --batch-size 1 for per-question timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from gradproposal import tracking
from gradproposal.latent import bench as B
from gradproposal.latent.latentmas import Chain, decode, run_latent_agent
from gradproposal.latent.run import Runner, git_rev, partners

CONDITIONS = ("latent", "wrong", "nocache", "text", "latentonly", "wrongonly")
CACHE_ARMS = ("latent", "wrong", "latentonly", "wrongonly")
WRONG_ARMS = ("wrong", "wrongonly")
ONLY_ARMS = ("latentonly", "wrongonly")
SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"“'$])")

# ---------------------------------------------------------------- data


def split_question(question: str) -> tuple[str, str] | None:
    """(facts, ask): ask is the last sentence. None unless it ends in '?', carries no digit (so
    no number leaks to the solver) and at least one sentence of facts precedes it."""
    sents = SENTENCE.split(re.sub(r"\s+", " ", question).strip())
    if len(sents) < 2 or not sents[-1].endswith("?") or re.search(r"\d", sents[-1]):
        return None
    return " ".join(sents[:-1]), sents[-1]


def load_split(n: int = 100) -> tuple[list[dict], int]:
    """First n questions of the fixed GSM8K 300 subset that split cleanly; also the number of
    questions skipped on the way."""
    rows, skipped = [], 0
    for r in B.load("gsm8k"):
        parts = split_question(r["question"])
        if parts is None:
            skipped += 1
            continue
        rows.append({**r, "facts": parts[0], "ask": parts[1]})
        if len(rows) == n:
            break
    return rows, skipped


# ---------------------------------------------------------------- prompts


def sender_messages(facts: str) -> list[dict]:
    user = f"""You are an Information Agent. You hold the facts of a math problem. Another agent must answer a question about these facts, but it cannot see them.

Facts: {facts}

Pass on every quantity and relationship the other agent will need. Do not guess the question and do not solve anything.
Now output your message below:
"""  # noqa: E501
    return B._messages(user)


def text_solver_messages(ask: str, message: str) -> list[dict]:
    """The LatentMAS solver prompt with the latent slot replaced by the sender's text."""
    user = f"""
Target Question: {ask}

You are a helpful assistant. You are provided with a message from an Information Agent for reference and a target question to solve.

Message from the Information Agent:
{message}

The message might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""  # noqa: E501
    return B._messages(user)


def visible(text: str) -> tuple[str, bool]:
    """Text after the thinking block, and whether thinking was closed. An unclosed block (cut by
    the token limit) is sent whole, minus the <think> tag, and flagged."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip(), True
    return text.replace("<think>", "").strip(), False


# ---------------------------------------------------------------- compute accounting


def forward_flops(cfg, new: int, ctx: int = 0) -> float:
    """Forward FLOPs (2 per multiply-add) to add `new` positions on top of `ctx` cached ones in a
    dense Qwen3-style decoder: weight matmuls per position (q, k, v, o, gated MLP) plus attention
    scores and weighted sums, 4 * heads * head_dim per (query, key) pair. Norms, rotary and the
    softmax are left out (under 1%). Additive: f(a + b, c) = f(a, c) + f(b, c + a)."""
    d, L, f = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    hq, hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or d // hq
    per_pos = 2 * L * (d * hq * hd + 2 * d * hkv * hd + hq * hd * d + 3 * d * f)
    pairs = new * ctx + new * (new + 1) // 2
    return float(per_pos * new + 4 * L * hq * hd * pairs)


def head_flops(cfg, n: int) -> float:
    """LM-head FLOPs for n next-token distributions."""
    return float(2 * cfg.hidden_size * cfg.vocab_size * n)


def kv_bytes(cfg, positions: int, bytes_per_value: int = 2) -> int:
    """Size of a KV cache over `positions` (all layers, K and V)."""
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * positions * bytes_per_value


def stage_flops(cfg, cond: str, *, sender_prompt: int, m: int, sender_out: int,
                solver_prompt: int, answer: int) -> dict:
    """Estimated FLOPs per stage for one question. Latent steps add 2 d^2 each (the alignment
    map); a decoded sequence of k tokens samples k + 1 times (the last one is the stop)."""
    d = cfg.hidden_size
    if cond in CACHE_ARMS:  # masked slots in the -only arms are still computed, so same ctx
        sender = forward_flops(cfg, sender_prompt + m) + 2 * d * d * m
        ctx = sender_prompt + m
    elif cond == "text":
        sender = forward_flops(cfg, sender_prompt + sender_out) + head_flops(cfg, sender_out + 1)
        ctx = 0
    else:
        sender, ctx = 0.0, 0
    prompt = forward_flops(cfg, solver_prompt, ctx)
    answer_f = forward_flops(cfg, answer, ctx + solver_prompt) + head_flops(cfg, answer + 1)
    return {"flops_sender": sender, "flops_solver_prompt": prompt, "flops_solver_answer": answer_f}


def latent_only(chain: Chain, m: int) -> None:
    """Hide every sender prompt slot but the first real one (in place), so the solver sees only
    the m latent positions plus an attention sink. Positions are kept as written (keys stay
    rotated; the solver's positions follow the full sender length). Dense caches only: a
    recurrent state has already folded the prompt in."""
    if chain.hybrid:
        raise ValueError("latent-only needs a pure KV cache (recurrent states hold the prompt)")
    if m < 1:
        raise ValueError("latent-only needs m >= 1")
    rows = torch.arange(chain.mask.shape[0], device=chain.mask.device)
    first = chain.mask.argmax(1)  # first real slot per row (prompts are left-padded)
    keep = torch.zeros_like(chain.mask)
    keep[:, -m:] = 1
    keep[rows, first] = 1
    chain.mask = keep


# ---------------------------------------------------------------- runner


class SplitRunner(Runner):
    """Runner with the two-agent split arms; reuses model loading, chat rendering and decoding."""

    def _sync(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

    def _ntok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def run(self, cond, items, mates, m, gen):
        """Per-row dicts: solver text, answer tokens, message and size fields, seconds per stage."""
        batch = len(items)
        out = [{} for _ in items]
        t0 = time.perf_counter()
        if cond == "text":
            send = [self.chat(sender_messages(r["facts"])) for r in items]
            raw, n_out = self._decode_fresh(send, gen)
            for o, s, t, k in zip(out, send, raw, n_out, strict=True):
                msg, closed = visible(t)
                o.update(sender_prompt=self._ntok(s), sender_out=k, message=msg,
                         message_tokens=self._ntok(msg), message_bytes=len(msg.encode()),
                         sender_closed=closed)
            prompts = [self.chat(text_solver_messages(r["ask"], o["message"]), self.think)
                       for r, o in zip(items, out, strict=True)]
            self._sync()
            t1 = time.perf_counter()
            texts, ntok = self._decode_fresh(prompts, gen)
            prefill_s = None  # not separated for chunked fresh decoding
        else:
            prompts = [self.chat(B.latent_messages("judger", r["ask"], "gsm8k"), self.think)
                       for r in items]
            if cond == "nocache":
                chain = Chain.empty(self.model, batch)
            else:
                src = mates if cond in WRONG_ARMS else items
                send = [self.chat(sender_messages(r["facts"]), self.think) for r in src]
                chain = run_latent_agent(self.model, self.tok, Chain.empty(self.model, batch),
                                         send, m, self.w_a, self.norm)
                if cond in ONLY_ARMS:
                    latent_only(chain, m)
                for o, s in zip(out, send, strict=True):
                    sent = 1 + m if cond in ONLY_ARMS else self._ntok(s) + m
                    o.update(sender_prompt=self._ntok(s), sender_out=0, message_tokens=sent,
                             message_bytes=kv_bytes(self.model.config, sent))
            self._sync()
            t1 = time.perf_counter()
            stats = {}
            ids = decode(self.model, self.tok, chain, prompts, max_new_tokens=self.max_new,
                         temperature=self.temperature, top_p=self.top_p, gen=gen, stats=stats)
            texts = [self.tok.decode(x, skip_special_tokens=True).strip() for x in ids]
            ntok, prefill_s = [len(x) for x in ids], stats["prefill_s"]
        self._sync()
        t2 = time.perf_counter()
        for o, p, t, k in zip(out, prompts, texts, ntok, strict=True):
            o.update(text=t, tokens=k, solver_prompt=self._ntok(p),
                     time_sender_s=round((t1 - t0) / batch, 3),
                     time_solver_s=round((t2 - t1) / batch, 3),
                     time_solver_prompt_s=(None if prefill_s is None
                                           else round(prefill_s / batch, 3)))
            o.setdefault("sender_prompt", 0)
            o.setdefault("sender_out", 0)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--n", type=int, default=100, help="questions (first that split cleanly)")
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--out")
    ap.add_argument("--report", help="print the summary of a finished results file")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.report)
    conds = a.conditions.split(",")
    assert all(c in CONDITIONS for c in conds), conds
    if not a.out:
        ap.error("--out is required (unless --report)")

    # W&B cloud logging is required: check it, and where the results file is filed, before
    # loading any data or model.
    tracking.require_wandb()
    out = Path(a.out)
    old_meta = tracking.read_results(out)[0] if out.exists() else None
    layout = tracking.layout_for("split", vars(a), out,
                                 git=[old_meta["git"], git_rev()] if old_meta else git_rev())
    tracking.require_results_folder(layout)

    rows, skipped = load_split(a.n)
    partner = [rows[j] for j in partners(len(rows), len(rows))]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    scored = {}  # condition -> {qid: correct}, earlier attempts included
    if out.exists():
        for r in map(json.loads, open(out, encoding="utf-8")):
            if "qid" in r:
                done.add((r["qid"], r["condition"]))
                scored.setdefault(r["condition"], {})[r["qid"]] = r["correct"]
    runner = SplitRunner(a.model, True, a.max_new_tokens, a.temperature, a.top_p, a.think)
    cfg, dev = runner.model.config, runner.model.device
    meta = {"meta": vars(a), "git": git_rev(), "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "date": datetime.now(UTC).isoformat(timespec="seconds"), "questions": len(rows),
            "skipped_unsplittable": skipped}
    # A resumed run keeps the config of its first attempt (the file's meta row).
    with tracking.TrackedRun(layout, tracking.config_from_meta(old_meta or meta)) as run:
        run.summary({"last_host": meta["host"], "last_gpu": meta["gpu"]})
        f = open(out, "a", encoding="utf-8", newline="\n")
        ftext = open(out.with_suffix(".texts.jsonl"), "a", encoding="utf-8", newline="\n")
        if old_meta is None:
            f.write(json.dumps({**meta, "wandb": run.pointer()}) + "\n")
        for cond in conds:
            acc_all = scored.setdefault(cond, {})
            for bi in range(0, len(rows), a.batch_size):
                items, mates = rows[bi : bi + a.batch_size], partner[bi : bi + a.batch_size]
                if all((r["id"], cond) in done for r in items):
                    continue
                gen = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + bi)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                res = runner.run(cond, items, mates, a.m, gen)
                dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
                for r, mate, o in zip(items, mates, res, strict=True):
                    pred, ok = B.score(o["text"], r["gold"])
                    m = a.m if cond in CACHE_ARMS else 0
                    row = {"qid": r["id"], "condition": cond, "seed": a.seed, "m": m,
                           "answer": pred, "gold": r["gold"], "correct": ok,
                           "partner": mate["id"] if cond in WRONG_ARMS else None,
                           "batch_size": len(items), "peak_mem_gb": round(peak, 2),
                           **{k: v for k, v in o.items() if k not in ("text", "message")},
                           **stage_flops(cfg, cond, sender_prompt=o["sender_prompt"], m=m,
                                         sender_out=o["sender_out"],
                                         solver_prompt=o["solver_prompt"], answer=o["tokens"])}
                    f.write(json.dumps(row) + "\n")
                    ftext.write(json.dumps({"qid": r["id"], "condition": cond, "text": o["text"],
                                            "message": o.get("message")},
                                           ensure_ascii=False) + "\n")
                    acc_all[r["id"]] = ok
                f.flush()
                ftext.flush()
                acc = sum(B.score(o["text"], r["gold"])[1]
                          for r, o in zip(items, res, strict=True))
                run.log({f"{cond}/running_acc": 100 * sum(acc_all.values()) / len(acc_all),
                         f"{cond}/questions_done": len(acc_all), f"{cond}/batch_s": dt,
                         f"{cond}/peak_mem_gb": peak, "batch_s": dt, "peak_mem_gb": peak})
                print(f"{cond} batch {bi // a.batch_size + 1}: {acc}/{len(items)} correct, "
                      f"peak {peak:.1f} GB", flush=True)
        f.close()
        ftext.close()
        run.log_results([out])


# ---------------------------------------------------------------- report


def paired_ci(a: list[int], b: list[int], reps: int = 10_000, seed: int = 0) -> tuple:
    """Mean of a - b in points and a 95% bootstrap interval over questions."""
    diff = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(reps))
    return (100 * sum(diff) / len(diff), 100 * boots[int(0.025 * reps)],
            100 * boots[int(0.975 * reps)])


def report(path: str) -> dict:
    rows = [r for r in map(json.loads, open(path, encoding="utf-8")) if "qid" in r]
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["qid"]] = r
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print("| Arm | Accuracy (%) | Sender TFLOPs | Solver TFLOPs | Message | Answer tokens |")
    print("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        if c not in by:
            continue
        rs = list(by[c].values())
        kv_mb = med([r.get("message_bytes", 0) for r in rs]) / 2**20
        solver = med([r["flops_solver_prompt"] + r["flops_solver_answer"] for r in rs])
        msg = (f"{kv_mb:.1f} MB KV" if c in CACHE_ARMS
               else f"{med([r['message_tokens'] for r in rs])} tokens" if c == "text" else "none")
        print(f"| {c} | {100 * sum(r['correct'] for r in rs) / len(rs):.1f} "
              f"| {med([r['flops_sender'] for r in rs]) / 1e12:.2f} "
              f"| {solver / 1e12:.2f} "
              f"| {msg} | {med([r['tokens'] for r in rs])} |")
    out = {}
    for a, b in (("latent", "wrong"), ("latent", "nocache"), ("text", "nocache"),
                 ("latent", "text"), ("latentonly", "wrongonly"), ("latentonly", "nocache"),
                 ("latent", "latentonly")):
        if a in by and b in by:
            q = sorted(set(by[a]) & set(by[b]))
            out[f"{a}-{b}"] = paired_ci([by[a][i]["correct"] for i in q],
                                        [by[b][i]["correct"] for i in q])
            print(f"{a} - {b}: {out[f'{a}-{b}'][0]:+.1f} "
                  f"[{out[f'{a}-{b}'][1]:+.1f}, {out[f'{a}-{b}'][2]:+.1f}] points")
    if "latent-wrong" in out:
        verdict = "passes" if out["latent-wrong"][0] >= 10 else "KILLED"
        print(f"Kill criterion (latent - wrong >= 10 points): {verdict}")
    return out


if __name__ == "__main__":
    main()
