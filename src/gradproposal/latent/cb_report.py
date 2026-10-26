"""Summarize CacheBack runs (latent/results/cacheback-fanoutqa/cb-*.jsonl): table, contrasts, plot.

    python -m gradproposal.latent.cb_report [--plot latent/cacheback_by_arm.png]

Strict accuracy is the mean over draws of per-draw accuracy, +/- the sample standard deviation
across draws. Message recall pools the panel's reference groups (495 on 50 questions) and counts
each question once. Contrasts resample questions (draws averaged within a question), as the
paper's hierarchical bootstrap does with its three draws.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from gradproposal.latent.report import RESULTS

LABELS = {  # arm -> display name, in display order
    "question": "Receiver alone",
    "text": "Same-size text",
    "cb4": "CacheBack 4×",
    "cb16": "CacheBack 16×",
    "mqa4": "Mean query attention 4×",
    "wrong4": "Wrong message 4×",
    "noise4": "Noise message 4×",
}
PAPER = {  # Qwen3-8B, paper Table 6 (strict %) and Table 5 (message recall %)
    "question": (19, None), "text": (41, 89.5), "cb4": (55, 95.8), "cb16": (46, 91.5),
}


def load(paths) -> dict[str, list[dict]]:
    arms: dict[str, dict] = defaultdict(dict)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "qid" in r:
                arms[r["arm"]][(r["qid"], r["draw"])] = r  # resumed batches: keep the last
    return {a: list(v.values()) for a, v in arms.items()}


def summarize(rows: list[dict]) -> dict:
    by_draw = defaultdict(list)
    for r in rows:
        by_draw[r["draw"]].append(r["strict"])
    accs = [100 * statistics.mean(v) for v in by_draw.values()]
    first = {}
    for r in rows:
        if r.get("recall") is not None:
            first[r["qid"]] = r["recall"]
    hits = sum(sum(v) for v in first.values())
    groups = sum(len(v) for v in first.values())
    mean = statistics.mean
    return {
        "strict": mean(accs), "sd": statistics.stdev(accs) if len(accs) > 1 else float("nan"),
        "loose": 100 * mean(r["loose"] for r in rows),
        "draws": len(accs), "n": len({r["qid"] for r in rows}),
        "recall": 100 * hits / groups if groups else None,
        "positions": mean(r["msg_positions"] for r in rows),
        "mb": mean(r["msg_bytes"] for r in rows) / 2**20,
        "report_tokens": (mean(sum(r["report_tokens"]) for r in rows)
                          if rows[0].get("report_tokens") else None),
        "tokens": mean(r["tokens"] for r in rows),
        "closer": 100 * mean(r["closer"] for r in rows),
        "unclosed": 100 * mean(r["unclosed"] for r in rows),
        "receiver_s": mean(r["time_s"] for r in rows),
        "sender_s": (mean(r["sender_s"] for r in rows)
                     if rows[0].get("sender_s") is not None else None),
        "mem": max(r["peak_mem_gb"] for r in rows),
    }


def paired(a: list[dict], b: list[dict], n_boot: int = 10_000) -> str:
    """Strict-accuracy difference a - b (points), 95% interval over resampled questions."""
    def per_q(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["qid"]].append(r["strict"])
        return {q: statistics.mean(v) for q, v in out.items()}

    pa, pb = per_q(a), per_q(b)
    qs = sorted(set(pa) & set(pb))
    d = [pa[q] - pb[q] for q in qs]
    rng = random.Random(0)
    boot = sorted(100 * statistics.mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return (f"{100 * statistics.mean(d):+.1f} [{boot[int(0.025 * n_boot)]:+.1f}, "
            f"{boot[int(0.975 * n_boot)]:+.1f}] ({len(qs)} questions)")


def fmt(x, spec=".1f"):
    return "–" if x is None else format(x, spec)


def table(S: dict) -> str:
    lines = ["| Arm | Strict acc. (%) | Loose (%) | Message recall (%) | Positions sent | MB sent "
             "| Report tokens | Output tokens | Closer (%) | Sender s / q | Receiver s / q "
             "| Peak GB | Paper strict / recall |",
             "|---" * 13 + "|"]
    for arm, label in LABELS.items():
        s = S.get(arm)
        if not s:
            continue
        paper = PAPER.get(arm)
        ptxt = "–" if not paper else f"{paper[0]} / {fmt(paper[1])}"
        lines.append(
            f"| {label} | {s['strict']:.1f} ± {fmt(s['sd'])} | {s['loose']:.1f} | "
            f"{fmt(s['recall'])} | {s['positions']:.0f} | {s['mb']:.0f} | "
            f"{fmt(s['report_tokens'], '.0f')} | {s['tokens']:.0f} | {s['closer']:.0f} | "
            f"{fmt(s['sender_s'], '.0f')} | {s['receiver_s']:.0f} | {s['mem']:.0f} | {ptxt} |")
    return "\n".join(lines)


def plot(S: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in LABELS if a in S]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    color = {"text": "#d9822b", "cb4": "#2f6db3", "cb16": "#5b8fd1", "mqa4": "#8fb0de"}
    xs = range(len(arms))
    ax1.bar(xs, [S[a]["strict"] for a in arms],
            yerr=[0 if S[a]["sd"] != S[a]["sd"] else S[a]["sd"] for a in arms],
            color=[color.get(a, "#c7cfd9") for a in arms], capsize=3, label="ours")
    for i, a in enumerate(arms):
        if a in PAPER:
            ax1.plot(i, PAPER[a][0], "k_", markersize=18, mew=2,
                     label="paper (Qwen3-8B)" if i == 0 else None)
        ax1.text(i, S[a]["strict"] + 1.5, f"{S[a]['strict']:.0f}", ha="center", fontsize=8)
    ax1.set_xticks(list(xs), [LABELS[a] for a in arms], rotation=35, ha="right")
    ax1.set_ylabel("Strict accuracy (%), mean ± SD over 3 draws")
    ax1.set_title("FanOutQA accuracy (Qwen3-8B, 50 questions)")
    ax1.legend(fontsize=8, loc="upper right")
    rec = [a for a in arms if S[a]["recall"] is not None]
    ax2.bar(range(len(rec)), [S[a]["recall"] for a in rec],
            color=[color.get(a, "#c7cfd9") for a in rec])
    for i, a in enumerate(rec):
        if a in PAPER and PAPER[a][1]:
            ax2.plot(i, PAPER[a][1], "k_", markersize=18, mew=2)
        ax2.text(i, S[a]["recall"] + 1, f"{S[a]['recall']:.1f}", ha="center", fontsize=8)
    ax2.axhline(96.4, ls=":", color="gray", lw=1)
    ax2.text(len(rec) - 0.5, 97.2, "ceiling 96.4", fontsize=7, color="gray", ha="right")
    ax2.set_xticks(range(len(rec)), [LABELS[a] for a in rec], rotation=35, ha="right")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Message recall of reference groups (%)")
    ax2.set_title("What reached the receiver (black: paper)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=str(RESULTS.parent / "cacheback_by_arm.png"))
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    files = a.files or sorted(p for p in (RESULTS / "cacheback-fanoutqa").glob("cb-*.jsonl")
                              if ".texts." not in p.name and "reports" not in p.name)
    arms = load(files)
    S = {arm: summarize(rows) for arm, rows in arms.items()}
    print(table(S))
    for x, y in [("cb4", "text"), ("cb16", "text"), ("cb4", "question"), ("text", "question"),
                 ("cb4", "wrong4"), ("cb4", "mqa4"), ("cb4", "cb16")]:
        if x in arms and y in arms:
            print(f"- {LABELS[x]} − {LABELS[y]}: {paired(arms[x], arms[y])}")
    if S:
        plot(S, Path(a.plot))
        print(f"\nplot: {a.plot}")


if __name__ == "__main__":
    main()
