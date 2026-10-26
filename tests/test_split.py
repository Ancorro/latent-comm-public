"""Split-information probe: data split, compute accounting and the four arms on a tiny Qwen3."""

import torch

from gradproposal.latent.latentmas import Chain, alignment_matrix, extend, run_latent_agent
from gradproposal.latent.split import (
    SplitRunner,
    forward_flops,
    kv_bytes,
    latent_only,
    load_split,
    report,
    split_question,
    stage_flops,
    visible,
)
from tests.test_latentmas import TOK, V, tiny_model


def test_split_question():
    facts, ask = split_question(
        "Janet’s ducks lay 16 eggs per day. She eats three. How much does she make?")
    assert facts == "Janet’s ducks lay 16 eggs per day. She eats three."
    assert ask == "How much does she make?"
    assert split_question("He has 3 apples. How many after 2 more?") is None  # digit in ask
    assert split_question("He has 3 apples. Find the total.") is None  # not a question
    assert split_question("How many apples?") is None  # no facts


def test_load_split_fixed_subset():
    rows, skipped = load_split(100)
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    assert all(not any(ch.isdigit() for ch in r["ask"]) for r in rows)
    assert skipped >= 0 and rows[0]["id"] == "gsm8k-test-0"


def test_visible():
    assert visible("<think>x</think>\n\nmsg") == ("msg", True)
    assert visible("<think>cut off") == ("cut off", False)


def test_forward_flops_matches_brute_force_and_is_additive():
    cfg = tiny_model().config
    one = forward_flops(cfg, 1, 0) - 4 * cfg.num_hidden_layers * 4 * 16  # weights per position
    brute = sum(one + 4 * cfg.num_hidden_layers * 4 * 16 * (7 + i + 1) for i in range(5))
    assert forward_flops(cfg, 5, 7) == brute
    assert forward_flops(cfg, 9, 3) == forward_flops(cfg, 4, 3) + forward_flops(cfg, 5, 7)
    s = stage_flops(cfg, "latent", sender_prompt=10, m=4, sender_out=0, solver_prompt=6, answer=3)
    n = stage_flops(cfg, "nocache", sender_prompt=0, m=0, sender_out=0, solver_prompt=6, answer=3)
    assert n["flops_sender"] == 0 and s["flops_solver_prompt"] > n["flops_solver_prompt"]


@torch.no_grad()
def test_kv_bytes_equals_cache_size():
    model = tiny_model(dtype=torch.bfloat16)
    w_a, norm = alignment_matrix(model)
    chain = run_latent_agent(model, TOK, Chain.empty(model, 1), ["facts: 3 eggs"], 4, w_a, norm)
    real = sum(t.numel() * t.element_size() for layer in chain.cache.layers
               for t in (layer.keys, layer.values))
    assert real == kv_bytes(model.config, chain.length)


class ChatTok:
    """CharTokenizer plus the chat and decode methods the runner uses."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return TOK(text)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"][-40:] for m in messages) + "|A:"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + i % 90) for i in ids)


def tiny_runner():
    r = SplitRunner.__new__(SplitRunner)
    r.model, r.tok = tiny_model(), ChatTok()
    r.model.generation_config.eos_token_id = V - 1
    r.w_a, r.norm = alignment_matrix(r.model)
    r.max_new, r.temperature, r.top_p, r.think = 5, 0.0, 0.95, True
    return r


ITEMS = [{"id": f"q{i}", "facts": f"Tom has {i + 2} cats.", "ask": "How many cats?", "gold": "2"}
         for i in range(3)]


@torch.no_grad()
def test_arms_run_and_wrong_uses_partner_cache():
    r = tiny_runner()
    mates = ITEMS[1:] + ITEMS[:1]
    outs = {c: r.run(c, ITEMS, mates, 3, torch.Generator().manual_seed(0))
            for c in ("latent", "wrong", "nocache", "text")}
    for c, o in outs.items():
        assert len(o) == 3 and all(x["tokens"] <= 5 for x in o), c
    assert outs["nocache"][0]["sender_prompt"] == 0
    assert outs["latent"][0]["message_bytes"] == kv_bytes(r.model.config,
                                                         outs["latent"][0]["message_tokens"])
    assert "message" in outs["text"][0] and outs["text"][0]["sender_out"] <= 5
    # greedy: the wrong arm for question 0 equals the latent arm for its partner's cache
    alone = r.run("latent", [{**ITEMS[0], "facts": mates[0]["facts"]}], [mates[0]], 3,
                  torch.Generator().manual_seed(0))
    assert outs["wrong"][0]["text"] == alone[0]["text"]


@torch.no_grad()
def test_latent_solver_sees_sender_cache():
    """Right and no-cache solver logits differ: the cache reaches the solver."""
    model = tiny_model()
    w_a, norm = alignment_matrix(model)
    chain = run_latent_agent(model, TOK, Chain.empty(model, 1), ["Tom has 5 cats."], 3, w_a, norm)
    ids = torch.tensor([TOK("How many cats?")["input_ids"]])
    h_cache = extend(model, chain, torch.ones_like(ids), ids=ids)
    h_none = extend(model, Chain.empty(model, 1), torch.ones_like(ids), ids=ids)
    assert not torch.allclose(h_cache, h_none)
    assert V  # vocab shared with the LatentMAS tests


def test_report(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [{"meta": {}}]
    for i in range(20):
        for c, ok in (("latent", 1), ("wrong", i % 2), ("nocache", 0), ("text", 1)):
            row = {"qid": f"q{i}", "condition": c, "correct": ok, "tokens": 10,
                   "flops_sender": 1e12, "flops_solver_prompt": 1e12, "flops_solver_answer": 1e12}
            if c != "nocache":  # the runner writes no message fields for the floor arm
                row.update(message_bytes=2**20, message_tokens=50)
            rows.append(row)
    p.write_text("\n".join(map(__import__("json").dumps, rows)))
    out = report(str(p))
    assert out["latent-wrong"][0] == 50.0 and out["latent-nocache"][0] == 100.0


@torch.no_grad()
def test_latent_only_mask_and_hidden_slots_unused():
    """Per row, only the first real slot and the m latent slots stay visible (left padding
    included), and noise in the hidden prompt slots does not change the solver."""
    model = tiny_model()
    w_a, norm = alignment_matrix(model)
    m = 3
    chain = run_latent_agent(model, TOK, Chain.empty(model, 2),
                             ["Tom has 5 cats.", "Ann has 12 dogs and 3 birds."], m, w_a, norm)
    real = chain.mask.clone()
    latent_only(chain, m)
    assert chain.mask.sum(1).tolist() == [1 + m, 1 + m]
    for row in range(2):
        first = int(real[row].nonzero()[0])
        assert chain.mask[row, first] == 1 and chain.mask[row, -m:].all()
    noisy = chain.clone()
    hidden = (real.bool() & ~chain.mask.bool())[:, None, :, None]
    for layer in noisy.cache.layers:
        layer.keys = torch.where(hidden, torch.randn_like(layer.keys), layer.keys)
        layer.values = torch.where(hidden, torch.randn_like(layer.values), layer.values)
    ids = torch.tensor([TOK("How many?")["input_ids"]] * 2)
    h = extend(model, chain, torch.ones_like(ids), ids=ids)
    h_noisy = extend(model, noisy, torch.ones_like(ids), ids=ids)
    assert torch.allclose(h, h_noisy, atol=1e-5)


@torch.no_grad()
def test_only_arms_run_and_report_message_size():
    r = tiny_runner()
    mates = ITEMS[1:] + ITEMS[:1]
    for c in ("latentonly", "wrongonly"):
        o = r.run(c, ITEMS, mates, 3, torch.Generator().manual_seed(0))
        assert len(o) == 3 and o[0]["message_tokens"] == 1 + 3
        assert o[0]["message_bytes"] == kv_bytes(r.model.config, 4)
