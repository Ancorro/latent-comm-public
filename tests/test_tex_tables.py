"""tex_tables: the notebook's numbers from the committed result files, and well-formed output."""

import re

import pytest

from gradproposal.latent import tex_tables as T
from gradproposal.latent.bench import load
from gradproposal.latent.report import load_rows
from gradproposal.latent.run import partners

# latent/NOTEBOOK.md, Results: accuracy mean +/- SD over seeds (%)
GRID = {
    "GsmSingle": ("89.0", "0.6"), "GsmTextMAS": ("92.4", "0.5"), "GsmLatentMAS": ("89.1", "1.0"),
    "GsmWrong": ("85.1", "0.8"), "GsmNoCache": ("89.9", "0.5"), "GsmNoise": ("1.6", "0.7"),
    "GsmMZero": ("92.2", "0.8"), "GsmWrongMZero": ("93.0", "0.7"),
    "ArcSingle": ("90.9", "0.8"), "ArcLatentMAS": ("90.7", "1.2"), "ArcWrong": ("84.4", "0.7"),
}
SWEEP = {"Zero": "92.2", "Ten": "90.9", "Twenty": "91.1", "Forty": "89.1"}
# latent/NOTEBOOK.md, paired differences: contrast -> (GSM8K, ARC-Challenge), each (diff, lo, hi)
NOTEBOOK_PAIRED = {
    "LatentMASSingle": ((0.1, -2.7, 2.8), (-0.2, -2.4, 2.1)),
    "LatentMASNoCache": ((-0.8, -3.4, 1.8), (-0.8, -2.9, 1.3)),
    "LatentMASWrong": ((4.0, 0.1, 7.7), (6.2, 2.9, 9.8)),
    "WrongNoCache": ((-4.8, -8.3, -1.2), (-7.0, -10.3, -3.8)),
    "MZeroNoCache": ((2.3, 0.7, 4.2), (0.4, -1.1, 2.1)),
    "MZeroLatentMAS": ((3.1, 1.0, 5.6), (1.2, -0.8, 3.3)),
    "TextMASSingle": ((3.4, 1.4, 5.6), (2.4, 1.1, 4.0)),
    "MZeroWrongMZero": ((-0.8, -2.4, 0.9), (-0.8, -2.2, 0.8)),
    "WrongMZeroNoCache": ((3.1, 1.1, 5.1), (1.2, -0.7, 3.1)),
    "WrongMZeroWrong": ((7.9, 4.8, 11.2), (8.2, 5.1, 11.4)),
}
CI_TOLERANCE = 0.3  # points; the notebook's resampling code was not committed


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    return out, {k: v for k, (v, _) in T.write_all(out).items()}


def val(M, name):
    """Macro body without \\ensuremath{...}."""
    v = M[name]
    return v[len("\\ensuremath{"):-1] if v.startswith("\\ensuremath{") else v


def interval(M, name):
    lo, hi = val(M, name + "CI").strip("[]").split(", ")
    return float(lo), float(hi)


def test_latentmas_grid(gen):
    _, M = gen
    for stem, (acc, sd) in GRID.items():
        assert (M[stem + "Acc"], M[stem + "Sd"]) == (acc, sd), stem
    for word, acc in SWEEP.items():
        assert M[f"GsmSweep{word}Acc"] == acc, word


def test_paired_differences(gen):
    _, M = gen
    assert val(M, "GsmDiffLatentMASWrong") == "+4.0"
    assert val(M, "ArcDiffLatentMASWrong") == "+6.2"
    assert val(M, "GsmDiffMZeroNoCache") == "+2.3"
    assert val(M, "GsmDiffLatentMASWrongSeeds") == "+3.3,\\,+5.3,\\,+3.3"
    assert val(M, "ArcDiffLatentMASWrongSeeds") == "+7.0,\\,+7.3,\\,+4.3"
    for contrast, per_bench in NOTEBOOK_PAIRED.items():
        for bs, (diff, lo, hi) in zip(("Gsm", "Arc"), per_bench, strict=True):
            name = f"{bs}Diff{contrast}"
            assert float(val(M, name)) == pytest.approx(diff, abs=1e-9), name
            ours = interval(M, name)
            assert abs(ours[0] - lo) <= CI_TOLERANCE and abs(ours[1] - hi) <= CI_TOLERANCE, \
                (name, ours, (lo, hi))
    assert val(M, "ArcDiffTextMASLatentMAS") == "+2.7"  # notebook: 2.7 [+1.0, +4.4]


def test_partner_answers(gen):
    _, M = gen
    assert (M["GsmPartnerWrongHits"], M["GsmPartnerWrongErrors"]) == ("22", "134")
    assert (M["ArcPartnerWrongHits"], M["ArcPartnerWrongErrors"]) == ("38", "140")
    assert (M["GsmPartnerWrongMZeroHits"], M["GsmPartnerWrongMZeroErrors"]) == ("0", "63")
    assert (M["ArcPartnerWrongMZeroHits"], M["ArcPartnerWrongMZeroErrors"]) == ("9", "66")


def test_recorded_partners_follow_run_partners():
    _, partner = T.partner_answers(load_rows(T.latentmas_files()))
    for bench in T.BENCHES:
        rows = load(bench)
        mates = partners(len(rows), 32)
        assert all(partner[bench, r["id"]] == rows[j]["id"]
                   for r, j in zip(rows, mates, strict=True))


def test_split_probe(gen):
    _, M = gen
    assert [M[f"Split{s}Acc"] for s in ("Right", "Wrong", "NoCache", "Text")] == \
        ["91.0", "0.0", "0.0", "92.0"]
    assert (val(M, "SplitRightMinusWrong"), val(M, "SplitRightMinusWrongCI")) == \
        ("+91.0", "[+85.0, +96.0]")
    assert (val(M, "SplitRightMinusText"), val(M, "SplitRightMinusTextCI")) == \
        ("-1.0", "[-7.0, +5.0]")
    assert (M["SplitKVMiB"], M["SplitKVPositions"], M["SplitTextTokens"]) == ("25.0", "178", "111")
    assert (M["SplitRightTotalTF"], M["SplitTextTotalTF"]) == ("6.12", "14.44")
    assert (M["TimingRightSender"], M["TimingTextSender"]) == ("0.79", "15.0")
    assert (M["SplitN"], M["SplitSkipped"]) == ("100", "88")


def test_generated_files_are_clean(gen):
    out, M = gen
    files = sorted(out.glob("*.tex"))
    assert len(files) == 14
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"\b(nan|inf)\b", text, re.IGNORECASE), f.name
        assert "DO NOT EDIT" in text and "Sources (sha256" in text, f.name
        if f.name == "numbers.tex":
            continue
        body = text.split("\\midrule\n", 1)[1].split("\\bottomrule")[0]
        for line in body.splitlines():  # every data row is terminated
            assert line.startswith(("\\midrule", "\\cmidrule")) or line.endswith("\\\\"), line
    assert all(name.isalpha() for name in M)
