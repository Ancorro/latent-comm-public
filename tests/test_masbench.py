"""MASBench Reasoning on the latent channel: data split, MASBench-identical scoring, arms on a
tiny Qwen3."""

import json

import pytest
import torch

from gradproposal.latent import masbench as M
from gradproposal.latent.latentmas import alignment_matrix
from gradproposal.latent.split import kv_bytes
from tests.test_latentmas import tiny_model
from tests.test_split import ChatTok

# ---------------------------------------------------------------- data


def test_load_two_agent_split():
    rows = M.load()
    assert len(rows) == 300 and len({r["id"] for r in rows}) == 300
    assert (
        sum(r["level"] == "easy" for r in rows) == 200
        and sum(r["level"] == "hard" for r in rows) == 100
    )
    for r in rows:
        # every supporting fact appears in exactly one agent's evidence, so neither half holds all
        # of them (two released questions list a few empty-string facts; those carry no evidence)
        facts = [f for f in r["facts_a"] + r["facts_b"] if f.strip()]
        in_a, in_b = [f in r["a"] for f in facts], [f in r["b"] for f in facts]
        assert all(x != y for x, y in zip(in_a, in_b, strict=True)), r["id"]
        assert r["facts_a"] and r["facts_b"], r["id"]
    assert sum(any(not f.strip() for f in r["facts_a"] + r["facts_b"]) for r in rows) == 2
    assert M.load(limit=5)[0]["id"] == rows[0]["id"] and len(M.load(level="hard")) == 100


def test_render_context_matches_masbench_format():
    text = M.render_context([["T1", ["p one", " ", "p two"]], ["T2", ["p three"]]])
    assert text == "Article: T1\np one\np two\n\nArticle: T2\np three"


# ---------------------------------------------------------------- scoring, checked against
# MASBench's own code


@pytest.mark.parametrize(
    "ref,pred,want",
    [
        ("The Sunwing Sky-Sailor", "the sunwing sky-sailor", 1.0),
        ("The Sunwing Sky-Sailor", "Sunwing", 2 * (1 / 1) * (1 / 2) / (1 / 1 + 1 / 2)),
        ("1999", "In 1999", 1.0),  # "in" is not an article, so pred has 2 tokens: precision 1/2
        ("Paris", "London", 0.0),
        ("", "", 1.0),
        ("x", "", 0.0),
    ],
)
def test_f1_matches_masbench(ref, pred, want):
    if ref == "1999":
        want = 2 * (1 / 2) * 1 / (1 / 2 + 1)
    assert M.f1(ref, pred) == pytest.approx(want)
    try:  # when the official repo is checked out, compare bit for bit
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "private" / "ext" / "MASBench" / "Reasoning"
        if src.exists():
            sys.path.insert(0, str(src))
            from src.metrics import cal_f1_score

            assert M.f1(ref, pred) == pytest.approx(cal_f1_score(ref, pred))
    finally:
        sys.path = [p for p in sys.path if "MASBench" not in p]


def test_parse_answer_like_masbench():
    assert M.parse_answer("blah <A>answer</A> then <A>The Sky-Sailor.</A>") == "The Sky-Sailor"
    assert M.parse_answer("<A>Final Answer</A>") is None
    assert M.parse_answer("no tag here") is None
    assert M.parse_answer("<a>lower</a>") == "lower"
    assert M.score("reasoning... <A>Paris</A>", "Paris") == ("Paris", 1.0)
    assert M.score("gave up", "Paris") == (None, 0.0)  # the paper's rule: missing answer scores 0


def test_summarize_per_arm_and_level():
    rows = [
        {"condition": "latent", "level": "easy", "answer": "x", "f1": 1.0},
        {"condition": "latent", "level": "hard", "answer": None, "f1": 0.0},
        {"condition": "text", "level": "easy", "answer": "y", "f1": 0.5},
    ]
    s = M.summarize(rows)
    assert s["latent"] == {"n": 2, "answered": 1, "f1": 50.0, "f1_easy": 100.0, "f1_hard": 0.0}
    assert s["text"]["f1"] == 50.0 and "f1_hard" not in s["text"]


# ---------------------------------------------------------------- arms on a tiny model (CPU,
# seconds)


def tiny_runner():
    r = M.MASBenchRunner.__new__(M.MASBenchRunner)
    r.model, r.tok = tiny_model(), ChatTok()
    r.model.generation_config.eos_token_id = 1
    r.w_a, r.norm = alignment_matrix(r.model)
    r.max_new, r.temperature, r.top_p, r.think = 5, 0.0, 0.95, True
    return r


ITEMS = [
    {
        "id": f"q{i}",
        "level": "easy",
        "question": f"Which craft {i}?",
        "gold": "Sky",
        "a": f"Article: A{i}\nAlice fact {i}",
        "b": f"Article: B{i}\nBob fact {i}",
        "facts_a": [f"Alice fact {i}"],
        "facts_b": [f"Bob fact {i}"],
    }
    for i in range(3)
]


@torch.no_grad()
def test_arms_run_and_wrong_uses_partner_cache():
    r = tiny_runner()
    mates = ITEMS[1:] + ITEMS[:1]
    outs = {c: r.run(c, ITEMS, mates, 3, torch.Generator().manual_seed(0)) for c in M.ARMS}
    for c, o in outs.items():
        assert len(o) == 3 and all(x["tokens"] <= 5 for x in o), c
    assert outs["nocache"][0]["sender_prompt"] == 0 and outs["both"][0]["sender_prompt"] == 0
    lat = outs["latent"][0]
    assert lat["message_bytes"] == kv_bytes(r.model.config, lat["message_tokens"])
    assert "message" in outs["text"][0] and outs["text"][0]["sender_out"] <= 5
    # greedy: the wrong arm for question 0 equals the latent arm run on the partner's sender prompt
    # (Alice's cache is built from the partner's question and evidence; Bob keeps his own prompt)
    swapped = r.run(
        "latent", [dict(ITEMS[0], a=mates[0]["a"])], [mates[0]], 3, torch.Generator().manual_seed(0)
    )
    assert outs["wrong"][0]["sender_prompt"] == outs["latent"][1]["sender_prompt"]
    assert outs["wrong"][0]["solver_prompt"] == outs["latent"][0]["solver_prompt"]
    assert outs["wrong"][0]["text"] != ""
    assert swapped[0]["sender_prompt"] == outs["wrong"][0]["sender_prompt"]


@torch.no_grad()
def test_solver_prompts_hold_only_own_half():
    for latent in (True, False):
        msgs = M.solver_messages("Q?", "Bob's half", None if latent else "msg from Alice", latent)
        user = msgs[-1]["content"]
        assert "Bob's half" in user and "<A>answer</A>" in user
        assert ("latent KV" in user) == latent and ("msg from Alice" in user) == (not latent)
    send = M.sender_messages("Q?", "Alice's half")[-1]["content"]
    assert "Alice's half" in send and "Bob" in send and "<A>" not in send
    both = M.both_messages("Q?", "half a", "half b")[-1]["content"]
    assert "half a" in both and "half b" in both


def test_data_file_is_the_released_set():
    import gzip

    with gzip.open(M.DATA, "rt", encoding="utf-8") as f:
        items = json.load(f)
    assert len(items) == 300 and {i["level"] for i in items} == {"easy", "hard"}
    assert all(set(i["context"]) == {"agent_a", "agent_b"} for i in items)
