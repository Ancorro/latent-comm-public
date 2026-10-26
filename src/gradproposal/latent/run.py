"""Run conditions on a fixed benchmark subset: one JSONL row per question, condition and seed.

    python -m gradproposal.latent.run --bench gsm8k --conditions latent,wrong,noise,nocache \
        --m 40 --seeds 0 --out latent/results/latentmas-qwen3-4b/gsm8k-cache-s0.jsonl

Conditions (all share questions, batches and per-batch sampling seeds):
  single   one agent, chain-of-thought decoding (official single-agent prompt)
  textmas  planner -> critic -> refiner -> solver, each decodes text read by the next
  latent   LatentMAS: each agent reads its prompt on top of the inherited cache, takes m latent
           steps, and hands the cache on; the solver decodes. `--m 0` = prompt caches only.
  wrong    as latent, but the solver gets the planner/critic/refiner cache built for another
           question (its partner: the next question in its group of --partner-group consecutive
           questions, wrapping around); its own question is in its prompt
  noise    as latent, but every transferred K and V is replaced by Gaussian noise with the same
           shape and per-row, per-layer, per-K/V mean and std
  nocache  the solver alone with its LatentMAS prompt and an empty cache

The first row of each output file is run metadata. Re-running with the same --out resumes, and
continues the same W&B run. --out must sit in latent/results/<W&B group>/ (latent/results/dev/
for --limit smoke runs); see gradproposal.tracking for the layout and the online-only W&B rule.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradproposal import tracking
from gradproposal.latent import bench as B
from gradproposal.latent.latentmas import (
    Chain,
    alignment_matrix,
    decode,
    noise_like,
    run_latent_agent,
)

CHAIN_CONDITIONS = ("latent", "wrong", "noise")
CONDITIONS = ("single", "textmas", *CHAIN_CONDITIONS, "nocache")
# Cache tokens per decode chunk from an empty cache (~15 GB on Qwen3-4B). Single and No-cache
# batches of 32 (~2.3K tokens each) stay one chunk; TextMAS solver batches split.
KV_BUDGET = 100_000
NAMES = {"planner": "Planner", "critic": "Critic", "refiner": "Refiner", "judger": "Judger"}


class Runner:
    def __init__(self, model_name: str, realign: bool, max_new_tokens: int, temperature: float,
                 top_p: float, think: bool, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa"
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.w_a, self.norm = alignment_matrix(self.model, realign=realign)
        self.max_new, self.temperature, self.top_p, self.think = (
            max_new_tokens, temperature, top_p, think)

    def chat(self, messages, think: bool = False) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + "<think>" if think else text  # official --think appends "<think>"

    def _decode(self, chain, prompts, gen):
        ids = decode(self.model, self.tok, chain, prompts, max_new_tokens=self.max_new,
                     temperature=self.temperature, top_p=self.top_p, gen=gen)
        texts = [self.tok.decode(x, skip_special_tokens=True).strip() for x in ids]
        return texts, [len(x) for x in ids]

    def _decode_fresh(self, prompts, gen):
        """Decode from an empty cache, in chunks whose padded cache stays within KV_BUDGET tokens
        (TextMAS solver prompts carry three earlier outputs, up to ~6K tokens each)."""
        lens = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        texts, ntok, start = [], [], 0
        while start < len(prompts):
            end, width = start + 1, lens[start]
            while end < len(prompts):
                w = max(width, lens[end])
                if (end + 1 - start) * (w + self.max_new) > KV_BUDGET:
                    break
                end, width = end + 1, w
            t, n = self._decode(Chain.empty(self.model, end - start), prompts[start:end], gen)
            texts, ntok, start = texts + t, ntok + n, end
        return texts, ntok

    def chain(self, qs, bench, m):
        chain = Chain.empty(self.model, len(qs))
        for role in B.ROLES[:-1]:
            prompts = [self.chat(B.latent_messages(role, q, bench), self.think) for q in qs]
            chain = run_latent_agent(self.model, self.tok, chain, prompts, m, self.w_a, self.norm)
        return chain

    def run(self, cond, qs, partner_qs, bench, m, gen, noise_gen):
        """Final texts and total output tokens per question for one batch."""
        batch = len(qs)
        if cond == "single":
            prompts = [self.chat(B.single_messages(q, bench)) for q in qs]
            return self._decode_fresh(prompts, gen)
        if cond == "textmas":
            contexts, total = [""] * batch, [0] * batch
            for role in B.ROLES:
                # official context[:text_mas_context_length]: the default -1 drops the last char
                prompts = [self.chat(B.textmas_messages(role, q, c[:-1], bench))
                           for q, c in zip(qs, contexts, strict=True)]
                texts, n = self._decode_fresh(prompts, gen)
                total = [a + b for a, b in zip(total, n, strict=True)]
                contexts = [c + f"[{NAMES[role]}]:\n{t}\n\n" for c, t in zip(contexts, texts,
                                                                             strict=True)]
            return texts, total
        judger = [self.chat(B.latent_messages("judger", q, bench), self.think) for q in qs]
        if cond == "nocache":
            return self._decode_fresh(judger, gen)
        chain = self.chain(partner_qs if cond == "wrong" else qs, bench, m)
        if cond == "noise":
            chain = noise_like(chain, noise_gen)
        return self._decode(chain, judger, gen)


def partners(n: int, group: int) -> list[int]:
    """Wrong-cache derangement: question i's partner is the next one in its group of `group`
    consecutive questions, wrapping around. Fixed by position, so it does not depend on batch size
    (with --batch-size equal to the group it is a row rotation within each batch)."""
    out = []
    for start in range(0, n, group):
        size = min(group, n - start)
        out += [start + (i + 1) % size for i in range(size)]
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # deployed with git archive: no .git
        return Path("REVISION").read_text().strip() if Path("REVISION").exists() else "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=B.BENCHES, required=True)
    ap.add_argument("--conditions", default="latent")
    ap.add_argument("--m", default="40", help="comma list of latent step counts (chain conditions)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300, help="subset file size")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--partner-group", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--no-realign", dest="realign", action="store_false")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    conds = a.conditions.split(",")
    assert all(c in CONDITIONS for c in conds), conds
    ms = [int(x) for x in a.m.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    # W&B cloud logging is required: check it, and where the results file is filed, before
    # loading any data or model.
    tracking.require_wandb()
    out = Path(a.out)
    old_meta = tracking.read_results(out)[0] if out.exists() else None
    layout = tracking.layout_for("latentmas", vars(a), out,
                                 git=[old_meta["git"], git_rev()] if old_meta else git_rev())
    tracking.require_results_folder(layout)

    rows = B.load(a.bench, a.n, a.limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    texts_path = out.with_suffix(".texts.jsonl")
    done = set()
    scored = {}  # (condition, seed, m) -> {qid: correct}, earlier attempts included
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if "qid" in r:
                done.add((r["qid"], r["condition"], r["seed"], r["m"]))
                scored.setdefault((r["condition"], r["seed"], r["m"]), {})[r["qid"]] = r["correct"]

    runner = Runner(a.model, a.realign, a.max_new_tokens, a.temperature, a.top_p, a.think)
    dev = runner.model.device
    w_a_dev = float((runner.w_a - torch.eye(runner.w_a.shape[0], device=runner.w_a.device))
                    .norm() / runner.w_a.shape[0] ** 0.5)
    meta = {"meta": vars(a), "git": git_rev(), "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "w_a_minus_I_rms": w_a_dev, "target_norm": float(runner.norm)}
    # A resumed run keeps the config of its first attempt (the file's meta row).
    with tracking.TrackedRun(layout, tracking.config_from_meta(old_meta or meta)) as run:
        run.summary({"last_host": meta["host"], "last_gpu": meta["gpu"]})
        f = open(out, "a", encoding="utf-8", newline="\n")
        ftext = open(texts_path, "a", encoding="utf-8", newline="\n")
        if old_meta is None:
            f.write(json.dumps({**meta, "wandb": run.pointer()}) + "\n")

        jobs = [(c, m) for c in conds for m in (ms if c in CHAIN_CONDITIONS else [0])]
        partner = [rows[j] for j in partners(len(rows), a.partner_group)]
        assert all(p["id"] != r["id"] for p, r in zip(partner, rows, strict=True)) or len(rows) < 2
        starts = range(0, len(rows), a.batch_size)
        batches = [(rows[i : i + a.batch_size], partner[i : i + a.batch_size]) for i in starts]
        for seed in seeds:
            for cond, m in jobs:
                correct = n_done = 0
                label = (cond + (f".m{m}" if cond in CHAIN_CONDITIONS else "")
                         + (f".s{seed}" if len(seeds) > 1 else ""))
                acc = scored.setdefault((cond, seed, m), {})
                for bi, (items, mates) in enumerate(batches):
                    if all((r["id"], cond, seed, m) in done for r in items):
                        continue
                    gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + bi)
                    noise_gen = torch.Generator(device=dev).manual_seed(
                        seed * 1_000_003 + bi + 7919)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    texts, ntok = runner.run(cond, [r["question"] for r in items],
                                             [r["question"] for r in mates], a.bench, m, gen,
                                             noise_gen)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
                    for r, mate, text, n in zip(items, mates, texts, ntok, strict=True):
                        pred, ok = B.score(text, r["gold"])
                        row = {"qid": r["id"], "bench": a.bench, "condition": cond, "seed": seed,
                               "m": m, "answer": pred, "gold": r["gold"], "correct": ok,
                               "tokens": n, "time_s": round(dt / len(items), 3), "batch": bi,
                               "batch_size": len(items), "peak_mem_gb": round(peak, 2),
                               "partner": mate["id"] if cond == "wrong" else None}
                        f.write(json.dumps(row) + "\n")
                        ftext.write(json.dumps({"qid": r["id"], "condition": cond, "seed": seed,
                                                "m": m, "text": text}, ensure_ascii=False) + "\n")
                        correct += ok
                        n_done += 1
                        acc[r["id"]] = ok
                    f.flush()
                    ftext.flush()
                    run.log({f"{label}/running_acc": 100 * sum(acc.values()) / len(acc),
                             f"{label}/questions_done": len(acc), f"{label}/batch_s": dt,
                             f"{label}/peak_mem_gb": peak, "batch_s": dt, "peak_mem_gb": peak})
                    print(f"seed {seed} {cond} m={m} batch {bi + 1}/{len(batches)}: "
                          f"acc {correct}/{n_done}, {dt:.0f}s, peak {peak:.1f} GB", flush=True)
        f.close()
        ftext.close()
        run.log_results([out])


if __name__ == "__main__":
    main()
