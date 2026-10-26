"""MASBench Reasoning (Chu et al., arXiv 2610.04672) on the latent channel.

Two agents, Alice and Bob, each hold a different slice of the evidence for one HotpotQA-style
question
and cannot answer alone. Here Alice is the sender (reads her evidence, takes m latent steps) and Bob
the solver (continues from her cache with his own evidence and the question), the same one-hop
pattern as the split probe (`split.py`), so the arms, compute accounting and runner carry over:

    latent    Bob continues from Alice's cache (prompt + m latent positions)
    wrong     Bob gets the cache Alice built for the partner question (derangement)
    nocache   Bob alone with his half (floor: the published single-half number)
    text      Alice writes a message (visible text after </think>); Bob reads it as text
    both      one agent with both halves (ceiling; no channel)

Scoring is MASBench's own: the last <A>...</A> span, SQuAD normalization, word-level F1; the paper's
rule scores a missing answer 0. Data: latent/data/masbench_reasoning.json.gz, the 300 released
questions (CC BY-SA 4.0, adapted from HotpotQA), each with `context` (articles per agent),
`supporting_facts` and `answer`.

Run (GPU; one question at a time for hybrid models, see latentmas.py):
    python -m gradproposal.latent.masbench --model Qwen/Qwen3.5-2B --arms nocache,latent \\
        --limit 2 \\
        --out latent/results/dev/masbench-smoke.jsonl
    python -m gradproposal.latent.masbench --report latent/results/dev/masbench-smoke.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import re
import string
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from gradproposal import tracking
from gradproposal.latent import bench as B
from gradproposal.latent.latentmas import Chain, decode, run_latent_agent
from gradproposal.latent.run import git_rev, partners
from gradproposal.latent.split import SplitRunner, kv_bytes, stage_flops, visible

DATA = Path(__file__).resolve().parents[3] / "latent" / "data" / "masbench_reasoning.json.gz"
ARMS = ("latent", "wrong", "nocache", "text", "both")
CACHE_ARMS = ("latent", "wrong")

# ---------------------------------------------------------------- data


def load(limit: int | None = None, level: str | None = None) -> list[dict]:
    """Questions in dataset order with the two agents' evidence rendered as MASBench renders it
    (`_build_context_for_agent`: "Article: <title>" then the passages)."""
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        items = json.load(f)
    rows = []
    for it in items:
        if level and it["level"] != level:
            continue
        rows.append(
            {
                "id": it["id"],
                "level": it["level"],
                "question": it["question"],
                "gold": it["answer"],
                "type": it.get("type"),
                "a": render_context(it["context"]["agent_a"]),
                "b": render_context(it["context"]["agent_b"]),
                "facts_a": [f[2] for f in it["supporting_facts"]["agent_a"]],
                "facts_b": [f[2] for f in it["supporting_facts"]["agent_b"]],
            }
        )
    return rows[:limit] if limit else rows


def render_context(articles: list) -> str:
    lines = []
    for title, passages in articles:
        lines.append(f"Article: {title}")
        lines.extend(p.strip() for p in passages if p and p.strip())
        lines.append("")
    return "\n".join(lines).strip()


# ------------------------------------------------------- prompts (ours, fixed before any run)

GUIDE = (
    "Your private information is incomplete by design. When you are confident of the complete "
    "final answer, output it ONCE as <A>answer</A>; the span must hold the full answer, "
    "nothing else."
)


def sender_messages(question: str, evidence: str) -> list[dict]:
    user = f"""You are Alice, collaborating with Bob to answer a multi-hop question. Bob holds other
    evidence and cannot see yours.

Question: {question}

Information (only visible to you):
{evidence}

Pass on every fact from your information that Bob will need to answer the question. Do not guess the answer.
Now output your message to Bob below:
"""  # noqa: E501
    return B._messages(user)


def solver_messages(question: str, evidence: str, message: str | None, latent: bool) -> list[dict]:
    """Bob's prompt. With a text message it is quoted; with a latent cache the slot says so (as the
    LatentMAS solver prompt does); with neither, Bob answers from his half alone."""
    if latent:
        slot = (
            "Alice has passed you her information as latent KV representations that you can attend "
            "to; they are not visible as text."
        )
    elif message is not None:
        slot = f"Message from Alice:\n{message}"
    else:
        slot = "Alice is unavailable; answer from your own information."
    user = f"""You are Bob, collaborating with Alice to answer a multi-hop question. Alice holds
    other evidence.

Question: {question}

Information (only visible to you):
{evidence}

{slot}

{GUIDE}
Reason step by step, then give the final answer as <A>answer</A>.
"""  # noqa: E501
    return B._messages(user)


def both_messages(question: str, evidence_a: str, evidence_b: str) -> list[dict]:
    user = f"""Answer a multi-hop question from the information below.

Question: {question}

Information:
{evidence_a}

{evidence_b}

{GUIDE}
Reason step by step, then give the final answer as <A>answer</A>.
"""  # noqa: E501
    return B._messages(user)


# ---------------------------------------------------------------- scoring (MASBench metrics.py and
# conversation.py)

ANSWER_TAG = re.compile(r"<A>(.*?)</A>", re.DOTALL | re.IGNORECASE)
INVALID = {"answer", "final answer", "the answer", "answer:"}


def parse_answer(text: str) -> str | None:
    """Last valid <A>...</A> span, trailing period stripped (MASBench `parse_answer`)."""
    if not text:
        return None
    for cand in reversed(ANSWER_TAG.findall(text)):
        c = cand.strip()
        if c.lower() in INVALID:
            continue
        if c.endswith("."):
            c = c[:-1]
        if c:
            return c
    return None


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(ref: str, pred: str) -> float:
    r, p = normalize(ref).split(), normalize(pred).split()
    if not r or not p:
        return 1.0 if r == p else 0.0
    same = sum((Counter(r) & Counter(p)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def score(text: str, gold: str) -> tuple[str | None, float]:
    """(parsed answer, F1); a missing answer scores 0, the paper's rule."""
    pred = parse_answer(text)
    return pred, (f1(gold, pred) if pred is not None else 0.0)


# ---------------------------------------------------------------- runner


class MASBenchRunner(SplitRunner):
    """The split-probe runner on the Alice/Bob split; `run` returns per-row dicts like
    SplitRunner's."""

    def run(self, cond, items, mates, m, gen):
        batch = len(items)
        out = [{} for _ in items]
        t0 = time.perf_counter()
        if cond == "both":
            prompts = [
                self.chat(both_messages(r["question"], r["a"], r["b"]), self.think) for r in items
            ]
            self._sync()
            t1 = time.perf_counter()
            texts, ntok = self._decode_fresh(prompts, gen)
            prefill_s = None
        elif cond == "text":
            send = [self.chat(sender_messages(r["question"], r["a"]), self.think) for r in items]
            raw, n_out = self._decode_fresh(send, gen)
            for o, s, t, k in zip(out, send, raw, n_out, strict=True):
                msg, closed = visible(t)
                o.update(
                    sender_prompt=self._ntok(s),
                    sender_out=k,
                    message=msg,
                    message_tokens=self._ntok(msg),
                    message_bytes=len(msg.encode()),
                    sender_closed=closed,
                )
            prompts = [
                self.chat(solver_messages(r["question"], r["b"], o["message"], False), self.think)
                for r, o in zip(items, out, strict=True)
            ]
            self._sync()
            t1 = time.perf_counter()
            texts, ntok = self._decode_fresh(prompts, gen)
            prefill_s = None
        else:
            latent = cond in CACHE_ARMS
            prompts = [
                self.chat(solver_messages(r["question"], r["b"], None, latent), self.think)
                for r in items
            ]
            if cond == "nocache":
                chain = Chain.empty(self.model, batch)
            else:
                src = mates if cond == "wrong" else items
                send = [self.chat(sender_messages(r["question"], r["a"]), self.think) for r in src]
                chain = run_latent_agent(
                    self.model,
                    self.tok,
                    Chain.empty(self.model, batch),
                    send,
                    m,
                    self.w_a,
                    self.norm,
                )
                for o, s in zip(out, send, strict=True):
                    sent = self._ntok(s) + m
                    o.update(
                        sender_prompt=self._ntok(s),
                        sender_out=0,
                        message_tokens=sent,
                        message_bytes=kv_bytes(self.model.config, sent),
                    )
            self._sync()
            t1 = time.perf_counter()
            stats = {}
            ids = decode(
                self.model,
                self.tok,
                chain,
                prompts,
                max_new_tokens=self.max_new,
                temperature=self.temperature,
                top_p=self.top_p,
                gen=gen,
                stats=stats,
            )
            texts = [self.tok.decode(x, skip_special_tokens=True).strip() for x in ids]
            ntok, prefill_s = [len(x) for x in ids], stats["prefill_s"]
        self._sync()
        t2 = time.perf_counter()
        for o, p, t, k in zip(out, prompts, texts, ntok, strict=True):
            o.update(
                text=t,
                tokens=k,
                solver_prompt=self._ntok(p),
                time_sender_s=round((t1 - t0) / batch, 3),
                time_solver_s=round((t2 - t1) / batch, 3),
                time_solver_prompt_s=None if prefill_s is None else round(prefill_s / batch, 3),
            )
            o.setdefault("sender_prompt", 0)
            o.setdefault("sender_out", 0)
        return out


def summarize(rows: list[dict]) -> dict:
    """Mean F1 x100 per arm (paper rule), with easy/hard splits and answered counts."""
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        d = {
            "n": len(rs),
            "answered": sum(r["answer"] is not None for r in rs),
            "f1": round(100 * sum(r["f1"] for r in rs) / len(rs), 2),
        }
        for level in ("easy", "hard"):
            ls = [r for r in rs if r.get("level") == level]
            if ls:
                d[f"f1_{level}"] = round(100 * sum(r["f1"] for r in ls) / len(ls), 2)
        out[cond] = d
    return out


# ---------------------------------------------------------------- entrypoint


def report(path: str) -> dict:
    rows = tracking.unique_rows("masbench", [path])
    summary = summarize(rows)
    for cond, d in summary.items():
        extra = ""
        if cond in CACHE_ARMS:
            mb = sum(r["message_bytes"] for r in rows if r["condition"] == cond) / d["n"] / 2**20
            extra = f", {mb:.1f} MB KV per question"
        elif cond == "text":
            tk = sum(r["message_tokens"] for r in rows if r["condition"] == cond) / d["n"]
            extra = f", {tk:.0f} message tokens per question"
        print(f"{cond:8s} F1 {d['f1']:6.2f}  (easy {d.get('f1_easy', float('nan')):6.2f}, "
              f"hard {d.get('f1_hard', float('nan')):6.2f}; answered {d['answered']}/{d['n']})"
              f"{extra}")
    return summary


def _flops_cond(cond: str) -> str:
    return "latent" if cond in CACHE_ARMS else "text" if cond == "text" else "nocache"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, help="first N questions (smoke)")
    ap.add_argument("--level", choices=["easy", "hard"])
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--batch-size", type=int, default=1, help="hybrid models need 1")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0, help="paper setting: greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--out")
    ap.add_argument("--report", help="print the summary of a finished results file")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.report)
    arms = a.arms.split(",")
    assert all(c in ARMS for c in arms), arms
    if not a.out:
        ap.error("--out is required (unless --report)")

    tracking.require_wandb()
    out = Path(a.out)
    old_meta = tracking.read_results(out)[0] if out.exists() else None
    layout = tracking.layout_for("masbench", vars(a), out,
                                 git=[old_meta["git"], git_rev()] if old_meta else git_rev())
    tracking.require_results_folder(layout)

    rows = load(a.limit, a.level)
    partner = [rows[j] for j in partners(len(rows), len(rows))]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for r in tracking.read_results(out)[1]:
            done.add((r["qid"], r["condition"]))
    runner = MASBenchRunner(a.model, True, a.max_new_tokens, a.temperature, a.top_p, a.think)
    cfg, dev = runner.model.config, runner.model.device
    meta = {"meta": vars(a), "git": git_rev(), "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "date": datetime.now(UTC).isoformat(timespec="seconds"), "questions": len(rows)}
    with tracking.TrackedRun(layout, tracking.config_from_meta(old_meta or meta)) as run:
        run.summary({"last_host": meta["host"], "last_gpu": meta["gpu"]})
        f = open(out, "a", encoding="utf-8", newline="\n")
        ftext = open(out.with_suffix(".texts.jsonl"), "a", encoding="utf-8", newline="\n")
        if old_meta is None:
            f.write(json.dumps({**meta, "wandb": run.pointer()}) + "\n")
        for cond in arms:
            f1s = []
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
                    pred, sc = score(o["text"], r["gold"])
                    m = a.m if cond in CACHE_ARMS else 0
                    row = {"qid": r["id"], "level": r["level"], "condition": cond,
                           "seed": a.seed, "m": m, "answer": pred, "gold": r["gold"],
                           "f1": round(sc, 4), "partner": mate["id"] if cond == "wrong" else None,
                           "batch_size": len(items), "peak_mem_gb": round(peak, 2),
                           **{k: v for k, v in o.items() if k not in ("text", "message")},
                           **stage_flops(cfg, _flops_cond(cond), sender_prompt=o["sender_prompt"],
                                         m=m, sender_out=o["sender_out"],
                                         solver_prompt=o["solver_prompt"], answer=o["tokens"])}
                    f.write(json.dumps(row) + "\n")
                    ftext.write(json.dumps({"qid": r["id"], "condition": cond, "text": o["text"],
                                            "message": o.get("message")},
                                           ensure_ascii=False) + "\n")
                    f1s.append(sc)
                f.flush()
                ftext.flush()
                running = 100 * sum(f1s) / len(f1s)
                run.log({f"{cond}/running_f1": running, f"{cond}/questions_done": len(f1s),
                         f"{cond}/batch_s": dt, f"{cond}/peak_mem_gb": peak,
                         "batch_s": dt, "peak_mem_gb": peak})
                print(f"{cond} {len(f1s)}/{len(rows)}: running F1 {running:.1f}, {dt:.0f} s, "
                      f"peak {peak:.1f} GB", flush=True)
        f.close()
        ftext.close()
        run.log_results([out])


if __name__ == "__main__":
    main()
