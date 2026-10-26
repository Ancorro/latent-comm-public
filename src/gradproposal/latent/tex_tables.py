"""LaTeX tables and number macros for paper/, generated from committed result files.

    python -m gradproposal.latent.tex_tables [--out paper/generated]

Reads only committed files under latent/results/ and writes booktabs tables (the tabular only;
captions live in the papers) plus numbers.tex, which defines one macro per in-text number, so
prose in paper/ never types a result. Every file starts with a do-not-edit header naming the git
revision and each source file with its sha256.

Definitions, matching latent/NOTEBOOK.md:
- LatentMAS grid: accuracy is the mean over seeds of per-seed accuracy +/- the sample SD across
  seeds; tokens and time are means over rows, memory the maximum (report.summarize).
- Paired differences (a - b, points): per question, the difference averaged over the seeds both
  arms share; 95% percentile bootstrap over questions, 10,000 resamples, Random(0)
  (split.paired_ci). Per-seed values are differences of per-seed accuracies.
- Partner answers: a wrong answer equal to the gold answer of the question's wrong-cache partner,
  the partner recorded in the wrong-cache rows (fixed by position, so the same for every arm).
- Split probe: accuracy and paired intervals as in split.report. Medians follow the notebook,
  which used two conventions: split.report's upper median sorted(x)[n // 2] for sender FLOPs,
  message size and answer tokens, and statistics.median for per-question stage sums (FLOPs to
  answer start, total FLOPs) and for all batch-1 timings.
- Numbers from other papers are constants below, each with the repo file it is quoted from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from gradproposal.latent.cb_report import PAPER as CB_PAPER
from gradproposal.latent.report import LABELS, LATENTMAS_DIR, RESULTS, load_rows, summarize
from gradproposal.latent.run import git_rev
from gradproposal.latent.split import paired_ci

ROOT = RESULTS.parents[1]
OUT = ROOT / "paper" / "generated"
SPLIT = RESULTS / "split-probe" / "split-gsm8k.jsonl"
TIMING = RESULTS / "split-probe" / "split-gsm8k-timing.jsonl"
CHECKS = {  # macro stem -> real-model check file (python -m gradproposal.latent.check)
    "DenseBf": RESULTS / "checks" / "check-qwen3-4b.json",
    "DenseFp": RESULTS / "checks" / "check-qwen3-4b-fp32.json",
    "HybridBf": RESULTS / "checks" / "check-qwen3.5-4b.json",
    "HybridFp": RESULTS / "checks" / "check-qwen3.5-4b-fp32.json",
}
CB_CHECK = RESULTS / "cacheback-fanoutqa" / "cb-check-short.json"
MAIN_M = 40

BENCHES = {"gsm8k": ("Gsm", "GSM8K"), "arc_challenge": ("Arc", "ARC-Challenge")}
ARMS = {  # (condition, m) -> (macro stem, table label, short label); report.LABELS arms and order
    ("single", 0): ("Single", "Single", "Single"),
    ("textmas", 0): ("TextMAS", "TextMAS", "TextMAS"),
    ("latent", MAIN_M): ("LatentMAS", "LatentMAS", "LatentMAS"),
    ("wrong", MAIN_M): ("Wrong", "Wrong cache", "Wrong"),
    ("nocache", 0): ("NoCache", "No cache", "No cache"),
    ("noise", MAIN_M): ("Noise", "Noise cache", "Noise"),
    ("latent", 0): ("MZero", "$m = 0$ (prompt caches only)", "$m{=}0$"),
    ("wrong", 0): ("WrongMZero", "Wrong cache, $m = 0$", "Wrong $m{=}0$"),
}
assert [(c, MAIN_M if m is None else m) for c, m in LABELS] == list(ARMS)
CONTRASTS = [  # (a, b): the notebook's paired table, then TextMAS - LatentMAS (notebook text)
    (("latent", MAIN_M), ("single", 0)),
    (("latent", MAIN_M), ("nocache", 0)),
    (("latent", MAIN_M), ("wrong", MAIN_M)),
    (("wrong", MAIN_M), ("nocache", 0)),
    (("latent", 0), ("nocache", 0)),
    (("latent", 0), ("latent", MAIN_M)),
    (("textmas", 0), ("single", 0)),
    (("latent", 0), ("wrong", 0)),
    (("wrong", 0), ("nocache", 0)),
    (("wrong", 0), ("wrong", MAIN_M)),
    (("textmas", 0), ("latent", MAIN_M)),
]
NUMWORD = {0: "Zero", 10: "Ten", 20: "Twenty", 40: "Forty"}
SPLIT_ARMS = {  # split.CONDITIONS -> (macro stem, label)
    "latent": ("Right", "Right cache"),
    "wrong": ("Wrong", "Wrong cache"),
    "nocache": ("NoCache", "No cache"),
    "text": ("Text", "Text"),
}
SPLIT_CONTRASTS = [("latent", "wrong"), ("latent", "nocache"), ("text", "nocache"),
                   ("latent", "text")]

# ---------------------------------------------------------------- quoted numbers (other papers)

NB = "latent/NOTEBOOK.md"
PAPER_TABLE1 = {  # LatentMAS paper Table 1, Qwen3-4B, sequential, full test sets (NOTEBOOK.md)
    ("gsm8k", "single"): 82.4, ("gsm8k", "textmas"): 89.8, ("gsm8k", "latent"): 88.2,
    ("arc_challenge", "single"): 89.2, ("arc_challenge", "textmas"): 90.0,
    ("arc_challenge", "latent"): 92.3,
}
PAPER_TOKEN_CUT = {"gsm8k": 81, "arc_challenge": 73}  # LatentMAS vs TextMAS output tokens, %
AUDIT_PAIRING = {"gsm8k": 1.5, "arc_challenge": 0.2}  # causal audit, Qwen3-4B, 40 steps per role
QUOTED = {  # macro -> (typeset value, where in the repo it is quoted from)
    "QLatentMASAccGain": ("14.6", "papers/NOTES.md (LatentMAS: up to 14.6% higher accuracy)"),
    "QLatentMASTokCut": ("70.8--83.7", "papers/NOTES.md (LatentMAS: % fewer output tokens)"),
    "QLatentMASSpeedup": ("4", "papers/NOTES.md (LatentMAS: about 4x faster inference)"),
    "QAuditRight": ("100", "papers/NOTES.md (causal audit, private facts: matched cache %)"),
    "QAuditWrong": ("23--25", "papers/NOTES.md (causal audit, private facts: other caches %)"),
    "QAuditStdGap": ("2.8", "papers/NOTES.md (causal audit, standard benchmarks: points)"),
    "QCBRecall": ("91.5--95.8", "papers/LITERATURE.md (CacheBack message recall, %)"),
    "QCBTextRecall": ("84.8--91.1", "papers/LITERATURE.md (same-size text recall, %)"),
    "QCBRecallSixteen": ("91", "papers/LITERATURE.md (CacheBack recall stays above, to 16x)"),
    "QCBRecallOneTwoEight": ("50--63", "papers/LITERATURE.md (CacheBack recall at 128x, %)"),
    "QCBStrictFour": ("55.3", NB + " (CacheBack paper, Qwen3-8B, 4x, strict %)"),
    "QCBCeiling": ("96.4", NB + " (message-recall ceiling, 477 of 495 groups, %)"),
    "QHiddenMulti": ("30.1", "papers/BENCHMARKS.md (HiddenBench, multi-agent %)"),
    "QHiddenSingle": ("80.7", "papers/BENCHMARKS.md (HiddenBench, one agent with all info %)"),
    "QAbsCoTRL": ("82.0", "papers/LITERATURE.md (Abstract CoT, RL only)"),
    "QAbsCoTWarm": ("90.8", "papers/LITERATURE.md (Abstract CoT, warm-up then RL)"),
    "QOBFKeep": ("9.9--20.2", "papers/NOTES.md (compressed relay: % of prompt KV kept)"),
    "QOBFBandwidth": ("4.7", "papers/NOTES.md (compressed relay: bandwidth cut, x)"),
    "QKITECut": ("28--36", "papers/LITERATURE.md (KITE: less communication than full KV, x)"),
    "QLatComSpeedup": ("2.46", "papers/LITERATURE.md (LatCom vs LatentMAS speed, x)"),
    "QProxifieldGain": ("53", "papers/LITERATURE.md (Proxifield over star, %, 25 agents)"),
    "QProxifieldAgents": ("25", "papers/LITERATURE.md (Proxifield team size)"),
    "QHogwildTwo": ("40", "grfp/directions.md (Hogwild! attention slowdown, 2 workers, %)"),
    "QHogwildFour": ("100", "grfp/directions.md (Hogwild! attention slowdown, 4 workers, %)"),
    "QFiveHundredRatio": ("6--500", "papers/LITERATURE.md (500xCompressor ratios, x)"),
    "QFiveHundredFOne": ("70--74", "papers/LITERATURE.md (500xCompressor, % of F1 kept)"),
    "QICAERatio": ("4", "papers/LITERATURE.md (ICAE near-lossless ratio, x)"),
    "QGsmTestSize": ("1{,}319", NB + " (GSM8K test questions)"),
    "QArcTestSize": ("1{,}172", NB + " (ARC-Challenge test questions)"),
}
CB_PAPER_STEMS = {"question": "Alone", "text": "Text", "cb4": "Four", "cb16": "Sixteen"}
INF_SENDERS, INF_RAW, INF_SUMMARY = 3, 50_000, 16  # grfp/directions.md inference-view scenario

# ---------------------------------------------------------------- formatting


def f1(x: float) -> str:
    return f"{x:.1f}"


def signed(x: float, nd: int = 1) -> str:
    s = f"{x:+.{nd}f}"
    return "+" + s[1:] if float(s) == 0 else s  # no "-0.0"


def ci(lo: float, hi: float) -> str:
    return f"[{signed(lo)}, {signed(hi)}]"


def sci(x: float, nd: int = 1) -> str:
    mant, exp = f"{x:.{nd}e}".split("e")
    return mant if int(exp) == 0 else f"{mant}\\times 10^{{{int(exp)}}}"


def fmt_kl(x: float) -> str:
    """A KL estimate; |x| < 1e-8 (fp32 values come out slightly negative) is zero to rounding."""
    return "<10^{-8}" if abs(x) < 1e-8 else sci(x)


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def ratio(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def upper_median(xs):
    """split.report's median: the upper middle value."""
    return sorted(xs)[len(xs) // 2]


class Macros(dict):
    """name -> (LaTeX body, comment). Names are letters only (LaTeX control words)."""

    def add(self, name: str, value, note: str = "", math: bool = False) -> None:
        assert name.isalpha() and name not in self, name
        self[name] = (f"\\ensuremath{{{value}}}" if math else str(value), note)


def header(what: str, sources: list[Path]) -> str:
    lines = [f"% {what}",
             "% Generated by `python -m gradproposal.latent.tex_tables`. DO NOT EDIT.",
             f"% Git revision: {git_rev()}",
             "% Sources (sha256, first 12 hex digits):"]
    lines += [f"%   {p.relative_to(ROOT).as_posix()}  "
              f"{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}" for p in sources]
    return "\n".join(lines) + "\n"


def tabular(cols: str, head: list[str], body: list[str]) -> str:
    """booktabs tabular; rule lines (\\midrule, \\cmidrule) are written as they are."""
    row = lambda r: r if r.startswith(("\\midrule", "\\cmidrule")) else r + " \\\\"  # noqa: E731
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *map(row, head), "\\midrule",
             *map(row, body), "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- LatentMAS grid


def latentmas_files() -> list[Path]:
    return sorted(p for p in LATENTMAS_DIR.glob("*.jsonl") if ".texts." not in p.name)


def correct_index(rows: list[dict]) -> dict:
    """(bench, condition, m) -> {(qid, seed): 0 or 1}."""
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["bench"], r["condition"], r["m"])][(r["qid"], r["seed"])] = int(r["correct"])
    return idx


def paired(A: dict, B: dict) -> dict:
    """a - b in points: bootstrap over questions of the per-question seed-averaged difference."""
    keys = A.keys() & B.keys()
    per_q = defaultdict(list)
    for q, s in keys:
        per_q[q].append(A[q, s] - B[q, s])
    d = [statistics.mean(per_q[q]) for q in sorted(per_q)]
    mean, lo, hi = paired_ci(d, [0] * len(d))
    seeds = sorted({s for _, s in keys})
    per_seed = [100 * statistics.mean(A[q, s] - B[q, s] for q, s2 in keys if s2 == s)
                for s in seeds]
    return {"mean": mean, "lo": lo, "hi": hi, "per_seed": per_seed, "n": len(d)}


CHOICES = ("a", "b", "c", "d")  # ARC-Challenge answer letters (the prompt asks for A-D)


def partner_answers(rows: list[dict]) -> tuple[dict, dict]:
    """(bench, condition, m) -> [wrong answers equal to the partner's gold, wrong answers,
    chance-expected matches], and the partner map (bench, qid) -> partner qid recorded in the
    wrong-cache rows. Chance (multiple choice only): each wrong answer that is a choice letter is
    taken as uniform over the three non-gold letters, so it matches the partner's gold with
    probability 1/3 when the two golds differ, else 0."""
    partner, gold = {}, {}
    for r in rows:
        gold[r["bench"], r["qid"]] = r["gold"]
        if r["condition"] == "wrong":
            p = partner.setdefault((r["bench"], r["qid"]), r["partner"])
            assert p == r["partner"], ("partner differs across seeds or m", r)
    out = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        if not r["correct"]:
            c = out[r["bench"], r["condition"], r["m"]]
            pg = gold[r["bench"], partner[r["bench"], r["qid"]]]
            c[0] += r["answer"] is not None and r["answer"] == pg
            c[1] += 1
            c[2] += (r["answer"] in CHOICES and pg != r["gold"]) / 3
    return dict(out), partner


def two_bench_head(first: str, cols: list[str]) -> list[str]:
    k = len(cols)
    return [f"& \\multicolumn{{{k}}}{{c}}{{GSM8K}} & \\multicolumn{{{k}}}{{c}}{{ARC-Challenge}}",
            f"\\cmidrule(lr){{2-{k + 1}}}\\cmidrule(lr){{{k + 2}-{2 * k + 1}}}",
            " & ".join([first, *cols, *cols])]


def latentmas(M: Macros) -> dict[str, str]:
    files = latentmas_files()
    rows = load_rows(files)
    S = summarize(rows)
    idx = correct_index(rows)
    metas = [json.loads(open(p, encoding="utf-8").readline()) for p in files]
    T = {}

    first = metas[0]["meta"]
    M.add("LatentModel", first["model"].split("/")[-1], "model of every LatentMAS run")
    M.add("LatentTemp", first["temperature"])
    M.add("LatentTopP", first["top_p"])
    M.add("LatentMaxNew", thousands(first["max_new_tokens"]), "new tokens per agent")
    M.add("LatentBatch", first["batch_size"])
    M.add("LatentMainM", MAIN_M)
    M.add("LatentGPUs", ", ".join(sorted({m["gpu"].replace("NVIDIA ", "") for m in metas})),
          "GPUs named in the run metadata")

    for bench, (bs, _) in BENCHES.items():
        body = []
        for (cond, m), (stem, label, _) in ARMS.items():
            s = S[(bench, cond, m)]
            M.add(bs + stem + "Acc", f1(s["acc"]), "% , mean over seeds")
            M.add(bs + stem + "Sd", f1(s["sd"]), "SD over seeds")
            M.add(bs + stem + "Tokens", f"{s['tokens']:.0f}")
            M.add(bs + stem + "Time", f"{s['time']:.2f}")
            M.add(bs + stem + "Mem", f1(s["mem"]))
            dag = "$^\\dagger$" if cond == "textmas" else ""
            body.append(f"{label} & {f1(s['acc'])} $\\pm$ {f1(s['sd'])} & {s['tokens']:.0f} & "
                        f"{s['time']:.2f}{dag} & {f1(s['mem'])}")
        M.add(bs + "Seeds", S[(bench, "single", 0)]["seeds"])
        M.add(bs + "N", S[(bench, "single", 0)]["n"], "questions per seed")
        T[f"latentmas_{bench}"] = tabular(
            "lrrrr", ["Condition & Accuracy (\\%) & Output tokens & Time / question (s) & "
                      "Peak memory (GB)"], body)

        tok_cut = 100 * (1 - S[(bench, "latent", MAIN_M)]["tokens"]
                         / S[(bench, "textmas", 0)]["tokens"])
        M.add(bs + "TokCut", f"{tok_cut:.0f}", "LatentMAS vs TextMAS, % fewer output tokens")
        M.add(bs + "PaperTokCut", PAPER_TOKEN_CUT[bench], "quoted: " + NB)
        for cond, stem in (("single", "Single"), ("textmas", "TextMAS"), ("latent", "LatentMAS")):
            ours = S[(bench, cond, MAIN_M if cond == "latent" else 0)]["acc"]
            M.add(f"Paper{bs}{stem}", f1(PAPER_TABLE1[bench, cond]), "quoted: " + NB)
            M.add(f"{bs}{stem}VsPaper", signed(ours - PAPER_TABLE1[bench, cond]),
                  "ours minus paper, points", True)
        M.add(f"Paper{bs}Gain",
              signed(PAPER_TABLE1[bench, "latent"] - PAPER_TABLE1[bench, "single"]),
              "paper LatentMAS minus Single, points", True)

    # latent-step sweep (GSM8K)
    body = []
    for m in sorted(m for (b, c, m) in S if b == "gsm8k" and c == "latent"):
        s = S[("gsm8k", "latent", m)]
        M.add(f"GsmSweep{NUMWORD[m]}Acc", f1(s["acc"]), f"GSM8K LatentMAS, m = {m}")
        M.add(f"GsmSweep{NUMWORD[m]}Sd", f1(s["sd"]))
        body.append(f"{m} & {f1(s['acc'])} $\\pm$ {f1(s['sd'])} & {s['tokens']:.0f} & "
                    f"{s['time']:.2f} & {f1(s['mem'])}")
    T["latentmas_sweep"] = tabular(
        "rrrrr", ["Latent steps $m$ & Accuracy (\\%) & Output tokens & Time / question (s) & "
                  "Peak memory (GB)"], body)

    # paired differences
    body = []
    for a, b in CONTRASTS:
        cells = []
        for bench, (bs, _) in BENCHES.items():
            r = paired(idx[(bench, *a)], idx[(bench, *b)])
            name = f"{bs}Diff{ARMS[a][0]}{ARMS[b][0]}"
            seeds = ",\\,".join(signed(x) for x in r["per_seed"])
            M.add(name, signed(r["mean"]), f"{a} - {b}, points", True)
            M.add(name + "CI", ci(r["lo"], r["hi"]), "95% bootstrap over questions", True)
            M.add(name + "Seeds", seeds, "per seed", True)
            est = signed(r["mean"])
            if r["lo"] > 0 or r["hi"] < 0:  # interval excludes 0
                est = f"\\mathbf{{{est}}}"
            cells += [f"${est}$", f"${ci(r['lo'], r['hi'])}$", f"${seeds}$"]
        body.append(f"{ARMS[a][2]} $-$ {ARMS[b][2]} & " + " & ".join(cells))
    T["latentmas_paired"] = tabular(
        "lrcrrcr", two_bench_head("Contrast", ["$\\Delta$", "95\\% CI", "Per seed"]), body)
    for bench, (bs, _) in BENCHES.items():
        r = paired(idx[(bench, "latent", MAIN_M)], idx[(bench, "wrong", MAIN_M)])
        M.add(bs + "AuditPairing", f1(AUDIT_PAIRING[bench]), "quoted: " + NB)
        M.add(bs + "AuditRatio", ratio(r["mean"] / AUDIT_PAIRING[bench]),
              "our LatentMAS - Wrong over the audit's pairing cost")

    # partner answers (chance column for multiple choice only)
    hits, _ = partner_answers(rows)
    body = []
    for (cond, m), (stem, label, _) in ARMS.items():
        cells = []
        for bench, (bs, _) in BENCHES.items():
            h, n, chance = hits.get((bench, cond, m), (0, 0, 0.0))
            share = 100 * h / n if n else 0.0
            M.add(f"{bs}Partner{stem}Hits", h)
            M.add(f"{bs}Partner{stem}Errors", n, "wrong answers over all seeds")
            M.add(f"{bs}Partner{stem}Share", f"{share:.0f}", "% of wrong answers")
            cells += [f"{h} / {n}", f"{share:.0f}"]
            if bench == "arc_challenge":
                M.add(f"{bs}Partner{stem}Chance", f1(chance), "chance-expected matches")
                cells.append(f1(chance))
        body.append(f"{label} & " + " & ".join(cells))
    T["latentmas_partner"] = tabular(
        "lrrrrr", ["& \\multicolumn{2}{c}{GSM8K} & \\multicolumn{3}{c}{ARC-Challenge}",
                   "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
                   "Condition & Partner / wrong & \\% & Partner / wrong & \\% & Chance"], body)
    for bench, (bs, _) in BENCHES.items():
        h, n, _ = hits[(bench, "wrong", MAIN_M)]
        extra = n - hits[(bench, "latent", MAIN_M)][1]
        M.add(bs + "PartnerExtraShare", f"{100 * h / extra:.0f}",
              "partner answers / (wrong-cache errors - LatentMAS errors), %")

    # against the paper
    body = []
    for cond, stem in (("single", "Single"), ("textmas", "TextMAS"), ("latent", "LatentMAS")):
        cells = []
        for bench in BENCHES:
            ours = S[(bench, cond, MAIN_M if cond == "latent" else 0)]
            cells += [f1(PAPER_TABLE1[bench, cond]), f"{f1(ours['acc'])} $\\pm$ {f1(ours['sd'])}"]
        body.append(f"{stem} & " + " & ".join(cells))
    cells = [x for bs, _ in BENCHES.values()
             for x in (M[bs + "PaperTokCut"][0], M[bs + "TokCut"][0])]
    body += ["\\midrule", "Output-token cut, LatentMAS vs TextMAS (\\%) & " + " & ".join(cells)]
    T["latentmas_vs_paper"] = tabular(
        "lrrrr", two_bench_head("", ["Paper (quoted)", "Ours"]), body)

    return {k: header(f"LatentMAS grid table: {k}", files) + v for k, v in T.items()}


# ---------------------------------------------------------------- split-information probe


def load_split(path: Path) -> tuple[dict, dict[str, dict]]:
    """Metadata row and condition -> {qid: row}; a resumed batch keeps its last row."""
    meta, by = None, defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "qid" in r:
            by[r["condition"]][r["qid"]] = r
        elif meta is None:
            meta = r
    return meta, dict(by)


def split_probe(M: Macros) -> dict[str, str]:
    meta, by = load_split(SPLIT)
    tmeta, tby = load_split(TIMING)
    limit = meta["meta"]["max_new_tokens"]
    T = {}
    M.add("SplitN", meta["questions"], "questions")
    M.add("SplitSkipped", meta["skipped_unsplittable"], "questions that did not split cleanly")
    M.add("SplitScanned", meta["questions"] + meta["skipped_unsplittable"])
    M.add("SplitM", meta["meta"]["m"])
    M.add("SplitBatch", meta["meta"]["batch_size"])
    M.add("SplitGPU", meta["gpu"].replace("NVIDIA ", ""))
    M.add("SplitCommit", meta["git"][:7])
    M.add("SplitMaxNew", thousands(limit))
    M.add("SplitKillThreshold", 10, "pre-registered kill line, points (split.report)")

    body = []
    for c, (stem, label) in SPLIT_ARMS.items():
        rs = list(by[c].values())
        acc = 100 * sum(r["correct"] for r in rs) / len(rs)
        sender = upper_median([r["flops_sender"] for r in rs]) / 1e12
        start = statistics.median(r["flops_sender"] + r["flops_solver_prompt"] for r in rs) / 1e12
        total = statistics.median(r["flops_sender"] + r["flops_solver_prompt"]
                                  + r["flops_solver_answer"] for r in rs) / 1e12
        answer = upper_median([r["tokens"] for r in rs])
        if c in ("latent", "wrong"):
            kv = upper_median([r["message_bytes"] for r in rs]) / 2**20
            msg = f"{f1(kv)} MiB KV ({upper_median([r['message_tokens'] for r in rs])} pos.)"
        elif c == "text":
            msg = f"{upper_median([r['message_tokens'] for r in rs])} tokens"
        else:
            msg = "none"
        M.add(f"Split{stem}Acc", f1(acc))
        M.add(f"Split{stem}SenderTF", f"{sender:.2f}", "sender TFLOPs, upper median")
        M.add(f"Split{stem}StartTF", f"{start:.2f}", "TFLOPs to answer start, median of sums")
        M.add(f"Split{stem}TotalTF", f"{total:.2f}", "total TFLOPs, median of sums")
        M.add(f"Split{stem}AnswerTokens", answer, "upper median")
        M.add(f"Split{stem}HitLimit", sum(r["tokens"] >= limit for r in rs),
              "answers that reached the token limit")
        body.append(f"{label} & {f1(acc)} & {msg} & {sender:.2f} & {start:.2f} & {total:.2f} & "
                    f"{answer}")
    T["split_probe"] = tabular(
        "lrlrrrr", ["Arm & Acc.\\ (\\%) & Message & Sender & To answer start & Total & "
                    "Answer tokens",
                    "& & & TFLOPs & TFLOPs & TFLOPs &"], body)

    lat, txt = list(by["latent"].values()), list(by["text"].values())
    per_pos = {r["message_bytes"] / r["message_tokens"] for r in lat}
    assert len(per_pos) == 1, per_pos  # KV bytes are linear in positions
    per_pos = int(per_pos.pop())
    kv_bytes = upper_median([r["message_bytes"] for r in lat])
    text_bytes = upper_median([r["message_bytes"] for r in txt])
    M.add("SplitKVMiB", f1(kv_bytes / 2**20), "MiB, upper median")
    M.add("SplitKVPositions", upper_median([r["message_tokens"] for r in lat]),
          "sender prompt + latent steps, upper median")
    M.add("SplitTextTokens", upper_median([r["message_tokens"] for r in txt]), "upper median")
    M.add("SplitTextBytes", text_bytes, "UTF-8 bytes, upper median")
    M.add("SplitTextSenderOut", upper_median([r["sender_out"] for r in txt]),
          "tokens the text sender generated, thinking included, upper median")
    M.add("SplitTextUnclosed", sum(r.get("sender_closed") is False for r in txt),
          "text senders cut off inside their thinking")
    M.add("SplitBytesRatio", thousands(int(round(kv_bytes / text_bytes, -3))),
          "KV bytes / text bytes, ratio of upper medians, to the nearest 1,000")
    M.add("SplitFlopsRatio", ratio(statistics.median(
        r["flops_sender"] + r["flops_solver_prompt"] + r["flops_solver_answer"] for r in txt)
        / statistics.median(r["flops_sender"] + r["flops_solver_prompt"]
                            + r["flops_solver_answer"] for r in lat)),
        "text over right-cache total TFLOPs (medians)")
    M.add("KVKiBPerPos", per_pos // 1024, "Qwen3-4B bf16 KV KiB per position, all layers")
    M.add("InfSenders", INF_SENDERS, "grfp/directions.md scenario")
    M.add("InfRawPositions", thousands(INF_RAW), "grfp/directions.md scenario")
    M.add("InfSummaryPositions", INF_SUMMARY, "grfp/directions.md scenario")
    M.add("InfRawGiB", f1(INF_SENDERS * INF_RAW * per_pos / 2**30),
          "KV read per decoded token over all raw positions, GiB")
    M.add("InfSummaryMiB", f1(INF_SENDERS * INF_SUMMARY * per_pos / 2**20),
          "KV read per decoded token over the summaries only, MiB")

    body = []
    for a, b in SPLIT_CONTRASTS:
        q = sorted(set(by[a]) & set(by[b]))
        mean, lo, hi = paired_ci([by[a][i]["correct"] for i in q], [by[b][i]["correct"] for i in q])
        name = f"Split{SPLIT_ARMS[a][0]}Minus{SPLIT_ARMS[b][0]}"
        M.add(name, signed(mean), "points", True)
        M.add(name + "CI", ci(lo, hi), "95% bootstrap over questions", True)
        body.append(f"{SPLIT_ARMS[a][1]} $-$ {SPLIT_ARMS[b][1]} & ${signed(mean)}$ & "
                    f"${ci(lo, hi)}$ & {len(q)}")
    T["split_paired"] = tabular("lrrr", ["Contrast & $\\Delta$ (points) & 95\\% CI & Questions"],
                                body)

    # batch-1 timing (statistics.median, as in the notebook)
    M.add("TimingN", tmeta["questions"])
    M.add("TimingGPU", tmeta["gpu"].replace("NVIDIA ", ""))
    body = []
    for c in ("latent", "text", "nocache", "wrong"):
        stem, label = SPLIT_ARMS[c]
        rs = list(tby[c].values())
        send = statistics.median(r["time_sender_s"] for r in rs)
        sp = [r["time_solver_prompt_s"] for r in rs if r["time_solver_prompt_s"] is not None]
        sp = f"{statistics.median(sp):.2f}" if sp else "n/a"
        solver = statistics.median(r["time_solver_s"] for r in rs)
        k = sum(r["correct"] for r in rs)
        M.add(f"Timing{stem}Sender", f"{send:.2f}" if send < 10 else f1(send), "s, median")
        M.add(f"Timing{stem}Prompt", sp, "s, median")
        M.add(f"Timing{stem}Solver", f1(solver), "s, median")
        M.add(f"Timing{stem}Correct", k)
        body.append(f"{label} & {M[f'Timing{stem}Sender'][0]} & {sp} & {f1(solver)} & "
                    f"{k} / {len(rs)}")
    T["split_timing"] = tabular(
        "lrrrr", ["Arm & Sender (s) & Solver prompt (s) & Solver total (s) & Correct"], body)
    M.add("TimingTextSenderOut",
          f"{statistics.median(r['sender_out'] for r in tby['text'].values()):.0f}",
          "tokens the text sender generated in the timing run, median")
    speed = (statistics.median(r["time_sender_s"] for r in tby["text"].values())
             / statistics.median(r["time_sender_s"] for r in tby["latent"].values()))
    M.add("TimingSpeedup", ratio(speed), "text over right-cache sender time, medians")

    return {k: header(f"Split-information probe table: {k}", [SPLIT, TIMING]) + v
            for k, v in T.items()}


# ---------------------------------------------------------------- real-model checks


def checks(M: Macros) -> dict[str, str]:
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in CHECKS.items()}
    dtype = lambda d: "fp32" if d.get("dtype") == "float32" else "bf16"  # noqa: E731
    model = lambda d: d["model"].split("/")[-1]  # noqa: E731
    T = {}
    M.add("CheckWaMinusIFro", sci(data["DenseBf"]["w_a_minus_I_fro"]), "Qwen3-4B", True)
    M.add("CheckTargetNorm", f"{data['DenseBf']['target_norm']:.3f}", "mean embedding norm")

    body = []
    for key, d in data.items():
        exact = [x for b in BENCHES for x in d[b]["theorem_3_3_exact"]]
        gap = max(x["max_abs"] for x in exact)
        kl = max(x["kl_max"] for x in exact)
        agree = 100 * min(x["argmax_agree_frac"] for x in exact)
        kl0 = max(d[b]["theorem_3_3_exact_m0"]["kl_max"] for b in BENCHES)
        pos = [x["positions_compared"] for x in exact]
        M.add(f"Check{key}Gap", f"{gap:.3f}" if gap < 0.1 else f1(gap), "max |logit diff|")
        M.add(f"Check{key}KL", sci(kl), "max KL over positions, m = 40", True)
        mean_kl = max(x["kl_mean"] for x in exact)
        M.add(f"Check{key}MeanKL", fmt_kl(mean_kl), "max mean KL", True)
        M.add(f"Check{key}Agree", f1(agree), "min % of positions with the same top token")
        M.add(f"Check{key}ZeroKL", sci(kl0), "m = 0 control, max KL", True)
        body.append(f"{model(d)} & {dtype(d)} & {len(exact)} & {min(pos)}--{max(pos)} & "
                    f"{M[f'Check{key}Gap'][0]} & ${sci(kl)}$ & {f1(agree)} & ${sci(kl0)}$")
    T["checks_theorem"] = tabular(
        "llrrrrrr", ["Model & dtype & Questions & Positions & Max $|\\Delta|$ logit & Max KL & "
                     "Top-1 agree (\\%) & $m{=}0$ max KL"], body)

    body = []
    for key in ("DenseBf", "DenseFp"):
        d = data[key]
        off = [x for b in BENCHES for x in d[b]["official"]]
        bvs = [x for b in BENCHES for x in d[b]["batched_vs_single"]]
        bkl = max(x["kl"] for x in bvs)
        M.add(f"Check{key}OfficialN", len(off), "questions compared with the official code")
        M.add(f"Check{key}BatchKL", sci(bkl), "batched vs single rows, max KL", True)
        same = all(x["argmax_agree"] for x in off + bvs)
        body.append(f"{model(d)} & {dtype(d)} & {len(off)} & "
                    f"{max(x['cache_keys_max_abs_diff'] for x in off):.1f} & "
                    f"{max(x['max_abs'] for x in off):.1f} & {max(x['kl'] for x in off):.1f} & "
                    f"${sci(bkl)}$ & {'yes' if same else 'no'}")
    T["checks_official"] = tabular(
        "llrrrrrc", ["Model & dtype & Questions & Max $|\\Delta|$ key & Max $|\\Delta|$ logit & "
                     "KL & Batched KL & Same top-1"], body)

    cb = json.loads(CB_CHECK.read_text(encoding="utf-8"))
    body, ts = [], []
    for it in cb["items"]:
        ws = it["workers"]
        ts += [w["T"] for w in ws]
        equal = (all(w["prompt_equal"] and w["question_found"] and w["keep4_port_equal"]
                     and w["keep16_port_equal"] for w in ws)
                 and it["manager_prompt_equal"] and it["question_rows_equal"]
                 and it["keeps4_equal"])
        diff = max([it["payload_max_abs_diff"]]
                   + [w[k] for w in ws for k in ("latent_max_abs_diff", "cache_keys_max_abs_diff",
                                                  "snap_rel_diff", "support_rel_diff")])
        rl = it["receiver_logits"]
        body.append(f"\\texttt{{{it['qid'][:8]}}} & {', '.join(thousands(w['T']) for w in ws)} & "
                    f"{'yes' if equal else 'no'} & {diff:.1f} & "
                    f"{thousands(it['receiver_positions'])} & ${fmt_kl(rl['kl'])}$ & "
                    f"{'yes' if rl['argmax_agree'] else 'no'} & {f1(rl['max_abs'])}")
    M.add("CBCheckModel", model(cb))
    M.add("CBCheckChars", thousands(cb["evidence_chars"]), "evidence cut, characters")
    M.add("CBCheckQuestions", len(cb["items"]))
    M.add("CBCheckTMin", thousands(min(ts)))
    M.add("CBCheckTMax", thousands(max(ts)))
    T["checks_cacheback"] = tabular(
        "lrcrrrcr", ["Question & Sender positions $T_i$ & Equal & Max diff & Receiver pos. & "
                     "KL & Same top-1 & Max $|\\Delta|$ logit"], body)
    srcs = [*CHECKS.values(), CB_CHECK]
    return {k: header(f"Real-model check table: {k}", srcs) + v for k, v in T.items()}


def provenance() -> dict[str, str]:
    """One row per result file: question rows, then commit, GPU and host from its metadata."""
    files = [*latentmas_files(), SPLIT, TIMING]
    body = []
    for p in files:
        lines = open(p, encoding="utf-8").read().splitlines()
        meta = json.loads(lines[0])
        rows = sum('"qid"' in line for line in lines[1:])
        name = p.relative_to(RESULTS).as_posix().replace("_", "\\_")
        body.append(f"\\texttt{{{name}}} & {rows} & \\texttt{{{meta['git'][:7]}}} & "
                    f"{meta['gpu'].replace('NVIDIA ', '')} & "
                    f"\\texttt{{{meta['host'].split('.')[0]}}}")
    return {"provenance": header("Result-file provenance (metadata rows)", files) + tabular(
        "lrlll", ["File (under \\texttt{latent/results/}) & Rows & Commit & GPU & Host"], body)}


# ---------------------------------------------------------------- main


def write_all(out: Path = OUT) -> Macros:
    """Write every table and numbers.tex into `out`; returns the macros."""
    out.mkdir(parents=True, exist_ok=True)
    M = Macros()
    tables = latentmas(M) | split_probe(M) | checks(M) | provenance()
    for arm, (strict, recall) in CB_PAPER.items():
        stem = CB_PAPER_STEMS[arm]
        M.add(f"QCBPaper{stem}Strict", strict, "quoted: cb_report.PAPER (paper Table 6)")
        if recall is not None:
            M.add(f"QCBPaper{stem}Recall", recall, "quoted: cb_report.PAPER (paper Table 5)")
    for name, (value, src) in QUOTED.items():
        M.add(name, value, "quoted: " + src)
    for name, text in tables.items():
        (out / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
    sources = [*latentmas_files(), SPLIT, TIMING, *CHECKS.values(), CB_CHECK]
    lines = [header("Number macros for the papers. Q* macros are quoted from other papers.",
                    sources)]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" + (f"  % {n}" if n else "")
              for k, (v, n) in M.items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out = Path(ap.parse_args(argv).out)
    M = write_all(out)
    print(f"{len(M)} macros, {len(list(out.glob('*.tex')))} files in {out}")


if __name__ == "__main__":
    main()
