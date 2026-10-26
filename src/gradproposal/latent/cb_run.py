"""CacheBack on the FanOutQA panel, in three stages (Qwen3-8B senders and receiver):

    python -m gradproposal.latent.cb_run produce            # sender prefill + 40 latent steps +
                                                            # receiver-query scores, banked once
    python -m gradproposal.latent.cb_run text               # Qwen3-8B text reports, once
    python -m gradproposal.latent.cb_run receive --arms question,text,cb4,cb16,wrong4,noise4

Receiver arms (same questions, 3 draws each; the message is fixed across draws, as in the paper):
  question  the receiver alone, no messages
  text      the three senders' visible text reports in the "Worker reports" slot ("same-size
            text": text senders of the receiver's size)
  cbR       CacheBack at relative ratio R: each sender's kept rows (budget ceil(T / R)) + its 40
            latent rows, prefilled before the receiver prompt
  mqaR      as cbR with mean query attention (no CacheBack correction)
  wrongR    the cbR message built for the partner question (the next one in the panel)
  noiseR    the cbR message with every row replaced by Gaussian noise of the row's mean and std

Resumable: every output is appended per batch and finished keys are skipped on restart.
Outputs live in latent/results/cacheback-fanoutqa/ (the W&B group folder). Every stage logs to
W&B (hosted cloud only; see gradproposal.tracking): one run for `produce`, one per text shard, one
per receiver arm file; a restarted stage continues the same W&B run.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from gradproposal import tracking
from gradproposal.latent import bench as B
from gradproposal.latent import cacheback as cb
from gradproposal.latent import fanoutqa as fq
from gradproposal.latent.latentmas import Chain, decode
from gradproposal.latent.run import git_rev, partners

RESULTS = B.DATA.parent / "results"
GROUP_DIR = RESULTS / "cacheback-fanoutqa"  # folder = W&B group
TOP_K = 20  # Qwen3 recommended sampling (paper Table 11), with temperature 0.6 and top-p 0.95


class Lab:
    def __init__(self, model: str, max_new: int, closer: int, kv_budget: int):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = cb.load_model(model)
        cb.growable_kv()  # in-place cache appends: torch.cat copies the cache every step
        self.max_new, self.closer, self.kv_budget = max_new, closer, kv_budget
        self.think_close = self.tok.convert_tokens_to_ids("</think>")
        self.rows = B.load_fanoutqa()

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def worker_ids(self, item, w: int) -> list[int]:
        body = B.worker_body(item["question"], item["workers"][w]["text"])
        return self.ids(B.render(self.tok, body))

    def generate(self, messages, prompt_ids, gen):
        """Sample after [message rows][prompt]; a draw that hits the ceiling inside its thinking
        block gets `</think>` and continues for `closer` tokens (official answer closer)."""
        model, n = self.model, len(prompt_ids)

        def run(msgs, pids, max_new):
            chain = cb.receiver_chain(model, msgs) if msgs else Chain.empty(model, len(pids))
            out = decode(model, self.tok, chain, None, prompt_ids=pids, max_new_tokens=max_new,
                         temperature=0.6, top_p=0.95, top_k=TOP_K, gen=gen, prompt_chunk=4096)
            del chain
            return out

        outs = run(messages, prompt_ids, self.max_new)
        redo = [i for i, o in enumerate(outs)
                if len(o) >= self.max_new and self.think_close not in o]
        if redo and self.closer:
            cont = run([messages[i] for i in redo] if messages else None,
                       [prompt_ids[i] + outs[i] + [self.think_close] for i in redo], self.closer)
            for i, c in zip(redo, cont, strict=True):
                outs[i] = outs[i] + [self.think_close] + c
        injected = [i in redo for i in range(n)]
        return outs, injected

    def batches(self, widths: list[int]):
        """Consecutive groups of equal width (the draws of one question) within the KV budget.
        Equal widths mean no padding, so attention takes SDPA's fast grouped-query path;
        with a padding mask transformers copies every K/V to all query heads per step."""
        out, cur = [], []
        for i, w in enumerate(widths):
            full = (len(cur) + 1) * (w + self.max_new) > self.kv_budget
            if cur and (widths[cur[0]] != w or full):
                out.append(cur)
                cur = []
            cur.append(i)
        return out + ([cur] if cur else [])


def _meta(a, lab) -> dict:
    return {"meta": vars(a), "git": git_rev(), "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "date": datetime.now(UTC).isoformat(timespec="seconds")}


def _old_meta(out: Path) -> dict | None:
    """The meta row of an existing results file (a resumed stage), else None."""
    return tracking.read_results(out)[0] if out.suffix == ".jsonl" and out.exists() else None


def _done(path: Path, key) -> set:
    if not path.exists():
        return set()
    return {key(r) for r in map(json.loads, open(path, encoding="utf-8")) if "meta" not in r}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


# ---------------------------------------------------------------- stages


def produce(a, lab: Lab, track) -> None:
    bank = Path(a.bank)
    bank.mkdir(parents=True, exist_ok=True)
    with track(bank) as run:
        for item in lab.rows[: a.limit]:
            path = bank / f"{item['id']}.pt"
            if path.exists():
                continue
            q = item["question"]
            mgr_ids, lo, hi = B.question_span(lab.tok, B.render(lab.tok, B.manager_body(q)), q)
            workers = []
            for w in range(B.FANOUT_WORKERS):
                ids = lab.worker_ids(item, w)
                _peak_reset()
                _sync()
                t0 = time.perf_counter()
                chain = cb.prefill_head(lab.model, ids)
                _sync()
                t1 = time.perf_counter()
                sender = cb.roll(lab.model, chain, ids)
                _sync()
                t2 = time.perf_counter()
                sc = cb.scores(lab.model, sender, mgr_ids, lo, hi)
                _sync()
                t3 = time.perf_counter()
                workers.append({"prompt_ids": ids, "latents": sender.latents.cpu(), **sc,
                                "prefill_s": t1 - t0, "roll_s": t2 - t1, "score_s": t3 - t2,
                                "peak_gb": _peak()})
                del chain, sender
                torch.cuda.empty_cache()
            torch.save({"qid": item["id"], "judger_ids": mgr_ids, "lo": lo, "hi": hi,
                        "workers": workers}, path)
            seconds = sum(w["prefill_s"] + w["roll_s"] + w["score_s"] for w in workers)
            peak = max(w["peak_gb"] for w in workers)
            run.log({"sender_s": seconds, "peak_mem_gb": peak,
                     "sender_positions": sum(len(w["prompt_ids"]) + 40 for w in workers),
                     "questions_banked": len(list(bank.glob("*.pt")))})
            print(f"{item['id']}: T = {[len(w['prompt_ids']) + 40 for w in workers]}, "
                  f"{seconds:.0f}s, peak {peak:.1f} GB", flush=True)
        run.summary({"questions_banked": len(list(bank.glob("*.pt")))})


def report_files(path: str) -> list[Path]:
    """The reports file and its per-shard siblings (`<stem>.shardKofN.jsonl`)."""
    p = Path(path)
    return sorted(p.parent.glob(f"{p.stem}*.jsonl"))


def text_out(a) -> Path:
    """This text shard's reports file."""
    k, n = map(int, a.shard.split("/"))
    base = Path(a.reports)
    return base if n == 1 else base.with_name(f"{base.stem}.shard{k}of{n}.jsonl")


def text(a, lab: Lab, track) -> None:
    k, n = map(int, a.shard.split("/"))
    out = text_out(a)
    done = set()
    for p in report_files(a.reports):
        done |= _done(p, lambda r: (r["qid"], r["worker"]))
    jobs = [(item, w) for item in lab.rows[: a.limit] for w in range(B.FANOUT_WORKERS)][k::n]
    jobs = [(item, w) for item, w in jobs if (item["id"], w) not in done]
    new = not out.exists() or out.stat().st_size == 0
    with track(out) as run:
        f = open(out, "a", encoding="utf-8", newline="\n")
        if new:
            f.write(json.dumps({**_meta(a, lab), "wandb": run.pointer()}) + "\n")
        pids = [lab.worker_ids(item, w) for item, w in jobs]
        for bi, idx in enumerate(lab.batches([len(p) for p in pids])):
            gen = torch.Generator(device=lab.model.device).manual_seed(a.seed * 1_000_003 + bi)
            _peak_reset()
            t0 = time.perf_counter()
            outs, injected = lab.generate(None, [pids[i] for i in idx], gen)
            _sync()
            dt = time.perf_counter() - t0
            for i, o, inj in zip(idx, outs, injected, strict=True):
                item, w = jobs[i]
                raw = lab.tok.decode(o, skip_special_tokens=False)
                f.write(json.dumps({"qid": item["id"], "worker": w, "report": fq.visible(raw),
                                    "raw": raw, "tokens": len(o), "closer": inj,
                                    "time_s": round(dt / len(idx), 2), "batch_size": len(idx),
                                    "peak_mem_gb": round(_peak(), 2)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            run.log({"batch_s": dt, "peak_mem_gb": _peak(), "batch_rows": len(idx),
                     "batch_tokens_mean": sum(len(o) for o in outs) / len(outs),
                     "batch_closer": sum(injected)})
            print(f"reports batch {bi}: {len(idx)} rows, {dt:.0f}s, peak {_peak():.1f} GB, "
                  f"tokens {[len(o) for o in outs]}", flush=True)
        f.close()
        run.log_results([out])


def _arm(arm: str) -> tuple[str, int]:
    m = re.fullmatch(r"(question|text|cb|mqa|wrong|noise)(\d*)", arm)
    if not m:
        raise ValueError(arm)
    return m.group(1), int(m.group(2) or 0)


def message(lab: Lab, bankq: dict, kind: str, ratio: int):
    """Continuous rows of the three senders, per-sender keeps, and the kept prompt text (decoded
    run by run, for message recall)."""
    rows, keeps, texts = [], [], []
    for w in bankq["workers"]:
        keep = cb.select(w["snap" if kind == "mqa" else "support"], ratio)
        keeps.append(keep)
        rows.append(cb.message_rows(lab.model, w["prompt_ids"], w["latents"], keep))
        texts.append(fq.retained_text(w["prompt_ids"], keep, lab.tok.decode))
    return torch.cat(rows), keeps, fq.GAP.join(texts)


def payload(lab: Lab, items, qi, kind, ratio, bank, reports, mate) -> dict:
    """Message rows (or None), receiver prompt ids, keeps, partner id and message recall hits
    (reference groups of this question present in the message; None when it carries no text)."""
    item = items[qi]
    q = item["question"]
    out = {"msg": None, "keeps": None, "partner": None, "recall": None}
    if kind == "text":
        rep = [reports[(item["id"], w)]["report"] for w in range(B.FANOUT_WORKERS)]
        out["ids"] = lab.ids(B.render(lab.tok, B.manager_body(q, B.report_payload(rep))))
        out["recall"] = fq.group_hits(fq.gold_groups(item["id"], item["answer"]), fq.GAP.join(rep))
        out["text_tokens"] = [reports[(item["id"], w)]["tokens"] for w in range(B.FANOUT_WORKERS)]
        out["report_tokens"] = [len(lab.ids(r)) for r in rep]
        out["sender_s"] = sum(reports[(item["id"], w)]["time_s"] for w in range(B.FANOUT_WORKERS))
    else:
        out["ids"] = lab.ids(B.render(lab.tok, B.manager_body(q)))
    if kind in ("cb", "mqa", "wrong", "noise"):
        src = lab.rows[mate[qi]] if kind == "wrong" else item
        out["partner"] = src["id"] if kind == "wrong" else None
        bq = torch.load(bank / f"{src['id']}.pt", weights_only=False)
        out["msg"], out["keeps"], kept_text = message(lab, bq, kind, ratio)
        if kind == "noise":
            g = torch.Generator(device=out["msg"].device).manual_seed(10_007 + qi)
            out["msg"] = cb.noise_rows(out["msg"], g)
        else:
            groups = fq.gold_groups(item["id"], item["answer"])
            out["recall"] = fq.group_hits(groups, kept_text)
        out["sender_positions"] = [len(w["prompt_ids"]) + cb.LATENT_STEPS for w in bq["workers"]]
        out["sender_s"] = sum(w["prefill_s"] + w["roll_s"] + w["score_s"] for w in bq["workers"])
    out["positions"] = 0 if out["msg"] is None else out["msg"].shape[0]
    return out


def receive(a, lab: Lab, track) -> None:
    items = lab.rows[: a.limit]
    bank = Path(a.bank)
    reports = {}
    for p in report_files(a.reports):
        for r in map(json.loads, open(p, encoding="utf-8")):
            if "meta" not in r:
                reports[(r["qid"], r["worker"])] = r
    mate = partners(len(lab.rows), len(lab.rows))
    draws = [int(x) for x in a.draws.split(",")]
    for arm in a.arms.split(","):
        kind, ratio = _arm(arm)
        out = Path(a.out_dir) / f"cb-{arm}.jsonl"
        done = _done(out, lambda r: (r["qid"], r["draw"]))
        strict = {}  # (qid, draw) -> strict, earlier attempts included
        if out.exists():
            strict = {(r["qid"], r["draw"]): r["strict"]
                      for r in tracking.unique_rows("cacheback", [out])}
        new = not out.exists() or out.stat().st_size == 0
        with track(out) as run:
            f = open(out, "a", encoding="utf-8", newline="\n")
            ftext = open(out.with_suffix(".texts.jsonl"), "a", encoding="utf-8", newline="\n")
            if new:
                f.write(json.dumps({**_meta(a, lab), "wandb": run.pointer()}) + "\n")
            jobs = [(qi, d) for qi in range(len(items)) for d in draws
                    if (items[qi]["id"], d) not in done]
            cache: dict[int, tuple] = {}

            def load(qi, kind=kind, ratio=ratio, cache=cache):
                if qi not in cache:
                    cache.clear()  # jobs are question-major: one question's payload at a time
                    cache[qi] = payload(lab, items, qi, kind, ratio, bank, reports, mate)
                return cache[qi]

            widths = [load(qi)["positions"] + len(load(qi)["ids"]) for qi, _ in jobs]
            n_strict = 0.0
            for bi, idx in enumerate(lab.batches(widths)):
                batch = [jobs[i] for i in idx]
                loads = [load(qi) for qi, _ in batch]
                msgs = [p["msg"] for p in loads] if loads[0]["msg"] is not None else None
                gen = torch.Generator(device=lab.model.device).manual_seed(
                    a.seed * 1_000_003 + batch[0][0] * 7 + batch[0][1])
                _peak_reset()
                t0 = time.perf_counter()
                outs, injected = lab.generate(msgs, [p["ids"] for p in loads], gen)
                _sync()
                dt = time.perf_counter() - t0
                for (qi, d), p, o, inj in zip(batch, loads, outs, injected, strict=True):
                    item = items[qi]
                    raw = lab.tok.decode(o, skip_special_tokens=False)
                    ans = fq.visible(raw)
                    s = fq.score(fq.gold_groups(item["id"], item["answer"]), ans)
                    first = d == draws[0]
                    f.write(json.dumps({
                        "qid": item["id"], "arm": arm, "kind": kind, "ratio": ratio, "draw": d,
                        "strict": s["strict"], "loose": s["loose"], "tokens": len(o),
                        "closer": inj, "unclosed": "</think>" not in raw,
                        "time_s": round(dt / len(batch), 2), "batch_size": len(batch),
                        "peak_mem_gb": round(_peak(), 2), "msg_positions": p["positions"],
                        "msg_bytes": p["positions"] * lab.model.config.hidden_size * 2,
                        "prompt_tokens": len(p["ids"]), "partner": p["partner"],
                        "recall": None if p["recall"] is None else [int(x) for x in p["recall"]],
                        "sender_positions": p.get("sender_positions"),
                        "report_tokens": p.get("report_tokens"),
                        "sender_s": None if p.get("sender_s") is None else round(p["sender_s"], 2),
                        "keeps": p["keeps"] if first else None}) + "\n")
                    ftext.write(json.dumps({"qid": item["id"], "arm": arm, "draw": d,
                                            "answer": ans, "raw": raw},
                                           ensure_ascii=False) + "\n")
                    n_strict += s["strict"]
                    strict[(item["id"], d)] = s["strict"]
                f.flush()
                ftext.flush()
                run.log({"strict_running": 100 * sum(strict.values()) / len(strict),
                         "rows_done": len(strict), "batch_s": dt, "peak_mem_gb": _peak(),
                         "batch_tokens_mean": sum(len(o) for o in outs) / len(outs)})
                print(f"{arm} batch {bi}: {len(batch)} rows, {dt:.0f}s, peak {_peak():.1f} GB, "
                      f"tokens {[len(o) for o in outs]}, strict so far {n_strict:.0f}",
                      flush=True)
                torch.cuda.empty_cache()
            f.close()
            ftext.close()
            run.log_results([out])


def outputs(a) -> list[Path]:
    """What a stage writes: the bank folder, this text shard's file, or one file per arm."""
    if a.stage == "produce":
        return [Path(a.bank)]
    if a.stage == "text":
        return [text_out(a)]
    return [Path(a.out_dir) / f"cb-{arm}.jsonl" for arm in a.arms.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["produce", "text", "receive"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (pilots)")
    ap.add_argument("--bank", default=str(GROUP_DIR / "cb-bank"))
    ap.add_argument("--reports", default=str(GROUP_DIR / "cb-text-reports.jsonl"))
    ap.add_argument("--out-dir", default=str(GROUP_DIR))
    ap.add_argument("--arms", default="question,text,cb4,cb16,wrong4,noise4")
    ap.add_argument("--draws", default="0,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="text stage: worker prompts k::n of the rest")
    ap.add_argument("--max-new-tokens", type=int, default=24_000)
    ap.add_argument("--closer", type=int, default=4096)
    ap.add_argument("--kv-budget", type=int, default=300_000,
                    help="padded cache tokens per batch (~44 GB on Qwen3-8B)")
    a = ap.parse_args(argv)
    for arm in a.arms.split(","):
        _arm(arm)  # reject a bad arm name before loading anything

    # W&B cloud logging is required: check it, and where each output is filed, before loading
    # the panel or the model.
    tracking.require_wandb()
    layouts = {}
    for out in outputs(a):
        old = _old_meta(out)
        layouts[out] = tracking.layout_for("cacheback", vars(a), out,
                                           git=[old["git"], git_rev()] if old else git_rev())
        tracking.require_results_folder(layouts[out])

    lab = Lab(a.model, a.max_new_tokens, a.closer, a.kv_budget)

    def track(out: Path) -> tracking.TrackedRun:
        """The W&B run of one output. A resumed results file keeps its first attempt's config;
        the bank has no meta row, so `produce` uses the arguments alone."""
        old = _old_meta(out)
        config = old or ({"meta": vars(a)} if a.stage == "produce" else _meta(a, lab))
        run = tracking.TrackedRun(layouts[out], tracking.config_from_meta(config))
        run.summary({"last_host": platform.node(),
                     "last_gpu": torch.cuda.get_device_name() if torch.cuda.is_available()
                     else "cpu"})
        return run

    {"produce": produce, "text": text, "receive": receive}[a.stage](a, lab, track)


if __name__ == "__main__":
    main()
