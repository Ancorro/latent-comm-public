"""Summarize latent/results/latentmas-qwen3-4b/*.jsonl: a table per benchmark and the plot.

    python -m gradproposal.latent.report [--main-m 40] [--plot latent/accuracy_by_condition.png]

Accuracy is the mean over seeds of per-seed accuracy, +/- the sample standard deviation across
seeds. Tokens, time and memory are means over all rows (time is batch wall-clock / batch size).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "latent" / "results"
LATENTMAS_DIR = RESULTS / "latentmas-qwen3-4b"  # one folder per experiment (= W&B group)
LABELS = {  # (condition, m or None for "the main m") -> table label, in display order
    ("single", 0): "Single",
    ("textmas", 0): "TextMAS",
    ("latent", None): "LatentMAS",
    ("wrong", None): "Wrong cache",
    ("nocache", 0): "No cache",
    ("noise", None): "Noise cache",
    ("latent", 0): "m = 0 (prompt cache only)",
    ("wrong", 0): "Wrong cache, m = 0",
}


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "qid" in r:
                rows.append(r)
    # A resumed batch can repeat rows; keep the last per key.
    keyed = {(r["bench"], r["qid"], r["condition"], r["seed"], r["m"]): r for r in rows}
    return list(keyed.values())


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["bench"], r["condition"], r["m"])].append(r)
    out = {}
    for key, rs in groups.items():
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["seed"]].append(r["correct"])
        accs = [100 * sum(v) / len(v) for v in by_seed.values()]
        out[key] = {
            "acc": statistics.mean(accs),
            "sd": statistics.stdev(accs) if len(accs) > 1 else float("nan"),
            "seeds": len(accs),
            "n": len(rs) // max(1, len(accs)),
            "tokens": statistics.mean(r["tokens"] for r in rs),
            "time": statistics.mean(r["time_s"] for r in rs),
            "mem": max(r["peak_mem_gb"] for r in rs),
        }
    return out


def table(summary: dict, bench: str, main_m: int) -> str:
    lines = [
        "| Condition | Accuracy (%) | Output tokens | Time / question (s) | Peak GPU memory (GB) |"
        " Seeds × questions |",
        "|---|---|---|---|---|---|",
    ]
    for (cond, m), label in LABELS.items():
        s = summary.get((bench, cond, main_m if m is None else m))
        if s:
            lines.append(f"| {label} | {s['acc']:.1f} ± {s['sd']:.1f} | {s['tokens']:.0f} | "
                         f"{s['time']:.2f} | {s['mem']:.1f} | {s['seeds']} × {s['n']} |")
    sweep = sorted(m for (b, c, m) in summary if b == bench and c == "latent")
    if len(sweep) > 2:
        lines += ["", "| LatentMAS m | " + " | ".join(map(str, sweep)) + " |",
                  "|---" * (len(sweep) + 1) + "|",
                  "| Accuracy (%) | " + " | ".join(
                      f"{summary[(bench, 'latent', m)]['acc']:.1f} ± "
                      f"{summary[(bench, 'latent', m)]['sd']:.1f}" for m in sweep) + " |"]
    return "\n".join(lines)


def plot(summary: dict, benches: list[str], main_m: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(LABELS.items())
    fig, axes = plt.subplots(1, len(benches), figsize=(5.2 * len(benches), 4.2), squeeze=False)
    highlight = {"LatentMAS": "#2f6db3", "Wrong cache": "#d9822b", "No cache": "#8a8a8a"}
    for ax, bench in zip(axes[0], benches, strict=True):
        labels, accs, sds, colors = [], [], [], []
        for (cond, m), label in keys:
            s = summary.get((bench, cond, main_m if m is None else m))
            if s:
                labels.append(label.replace(" (prompt cache only)", ""))
                accs.append(s["acc"])
                sds.append(0 if s["sd"] != s["sd"] else s["sd"])
                colors.append(highlight.get(label, "#c7cfd9"))
        # Zoom on the informative range; collapsed arms (noise cache) are drawn off scale.
        sane = [a for a in accs if a >= 50] or accs
        lo, hi = max(0, min(sane) - 5), min(100, max(sane) + 3)
        ax.bar(range(len(accs)), accs, yerr=sds, color=colors, capsize=3)
        ax.set_xticks(range(len(accs)), labels, rotation=35, ha="right")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{bench} (Qwen3-4B; m = {main_m} unless labeled)")
        ax.set_ylabel("Accuracy (%), mean ± SD over seeds")
        for i, a in enumerate(accs):
            if a < lo:
                ax.text(i, lo + 0.4, f"{a:.1f}\n(off scale)", ha="center", fontsize=7)
            else:
                ax.text(i, a + sds[i] + 0.2, f"{a:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-m", type=int, default=40)
    ap.add_argument("--plot", default=str(RESULTS.parent / "accuracy_by_condition.png"))
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    files = a.files or sorted(p for p in LATENTMAS_DIR.glob("*.jsonl") if ".texts." not in p.name)
    summary = summarize(load_rows(files))
    benches = sorted({b for b, _, _ in summary})
    for bench in benches:
        print(f"\n### {bench}\n\n{table(summary, bench, a.main_m)}")
    if benches:
        plot(summary, benches, a.main_m, Path(a.plot))
        print(f"\nplot: {a.plot}")


if __name__ == "__main__":
    main()
