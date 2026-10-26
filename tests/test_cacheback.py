"""CacheBack mechanics on tiny random Qwen3 / Qwen3.5 models (CPU, seconds)."""

import copy

import pytest
import torch

from gradproposal.latent import cacheback as cb
from gradproposal.latent import fanoutqa as fq
from gradproposal.latent.latentmas import _sample, extend
from tests.test_latentmas import tiny_hybrid, tiny_model

V = 96


def rand_ids(n, seed):
    return torch.randint(1, V, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@torch.no_grad()
def logits_after(model, chain, ids):
    """Logits at every position of `ids` read on top of `chain` (batch 1)."""
    t = torch.tensor([ids])
    pos = chain.next_pos[:, None] + torch.arange(t.shape[1])
    out = model.model(input_ids=t, position_ids=pos, past_key_values=chain.cache, use_cache=True,
                      attention_mask=torch.cat([chain.mask, torch.ones_like(t)], 1))
    return model.lm_head(out.last_hidden_state[0])


@torch.no_grad()
def test_full_keep_equals_full_cache_relay():
    """Brief test 1: keeping 100% of positions (ratio 1), the receiver that re-prefills the
    message rows sees exactly what a LatentMAS-style relay of the sender's own cache gives."""
    model = tiny_model()
    prompt, judger = rand_ids(70, 0), rand_ids(12, 1)
    sender = cb.produce(model, prompt, steps=5, chunk=16)
    keep = cb.select(torch.rand(sender.length), ratio=1, steps=5)
    assert keep == list(range(sender.length))
    rows = cb.message_rows(model, prompt, sender.latents, keep)
    relay = logits_after(model, sender.chain.clone(), judger)
    ours = logits_after(model, cb.receiver_chain(model, [rows], chunk=32), judger)
    torch.testing.assert_close(ours, relay, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_scores_match_explicit_attention():
    """The fused capture equals the formulas computed from eager attention weights over
    [sender inputs, receiver prompt rows], and leaves the sender cache unchanged."""
    model = tiny_model()
    prompt, judger, lo, hi = rand_ids(60, 2), rand_ids(14, 3), 4, 11
    sender = cb.produce(model, prompt, steps=3)
    before = logits_after(model, sender.chain.clone(), judger)
    got = cb.scores(model, sender, judger, lo, hi)
    assert sender.chain.length == sender.length
    torch.testing.assert_close(logits_after(model, sender.chain.clone(), judger), before)

    emb = model.get_input_embeddings()
    rows = torch.cat([emb(torch.tensor(prompt)), sender.latents, emb(torch.tensor(judger[:hi]))])
    model.set_attn_implementation("eager")
    att = model.model(inputs_embeds=rows[None], output_attentions=True).attentions
    model.set_attn_implementation("sdpa")
    T, R = sender.length, hi - lo
    H, kv = model.config.num_attention_heads, model.config.num_key_value_heads
    votes = sum(a[0, :, T + lo : T + hi, :T].sum((0, 1)) for a in att) / (H * R)
    E = torch.zeros(T)
    Rm = torch.zeros(T)
    for a in att:
        w = a[0, :, T + lo : T + hi, :T]
        for s in range(kv):
            b = w[s * (H // kv) : (s + 1) * (H // kv)].mean(0)  # [rows, T]
            E += (cb.pool(b.mean(0)) * 256) ** 2
            Rm += ((cb.pool(b) * 256) ** 2).mean(0)
    snap = cb.pool(votes)
    support = snap * (Rm / E) ** 2
    snap[0], support[0] = snap.max(), support.max()
    torch.testing.assert_close(got["snap"], snap, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(got["support"], support, atol=1e-6, rtol=1e-4)


def test_fixed_span_keep():
    s = torch.zeros(64)
    s[40:48] = 5.0  # span 2 (positions 32..47) has the top mean
    s[20] = 9.0  # span 1 has a single peak at 20
    keep = cb.fixed_span_keep(s, 16 + 1 + 3, protected=(0,), span=16)
    assert len(keep) == 20 and 0 in keep and set(range(32, 48)) <= set(keep)
    rest = sorted(set(keep) - set(range(32, 48)) - {0})
    assert rest == [19, 20, 21]  # partial span: contiguous run centred on its peak
    sel = cb.select(torch.rand(100), ratio=4, steps=10)
    assert len(sel) == 25 and {0, *range(90, 100)} <= set(sel)
    with pytest.raises(ValueError):
        cb.fixed_span_keep(torch.rand(10), 2, protected=(0, 1, 2))


@torch.no_grad()
def test_receiver_batched_and_chunked_match_single():
    """Left-padded, chunked receiver prefill gives each row what it gets alone."""
    model = tiny_model()
    emb = model.get_input_embeddings()
    msgs = [emb(torch.tensor(rand_ids(n, n))) for n in (37, 21)]
    judger = rand_ids(9, 7)
    chain = cb.receiver_chain(model, msgs, chunk=8)
    t = torch.tensor([judger, judger])
    hb = model.lm_head(extend(model, chain, torch.ones_like(t), ids=t))
    for i, m in enumerate(msgs):
        single = logits_after(model, cb.receiver_chain(model, [m]), judger)[-1]
        torch.testing.assert_close(hb[i], single, atol=1e-4, rtol=1e-4)


def test_top_k_one_is_greedy():
    logits = torch.randn(5, V)
    got = _sample(logits, 0.6, 0.95, torch.Generator().manual_seed(0), top_k=1)
    assert torch.equal(got, logits.argmax(-1))


def test_noise_rows_match_moments():
    x = torch.randn(50, 256) * 3 + 1
    y = cb.noise_rows(x, torch.Generator().manual_seed(0))
    torch.testing.assert_close(y.mean(-1), x.mean(-1), atol=0.6, rtol=0)
    assert not torch.allclose(x, y)


@torch.no_grad()
def test_hybrid_scores_use_attention_layers_and_keep_state():
    model = tiny_hybrid()
    prompt, judger = rand_ids(30, 4), rand_ids(10, 5)
    sender = cb.produce(model, prompt, steps=2)
    assert sender.chain.hybrid
    before = logits_after(model, copy.deepcopy(sender.chain), judger)
    got = cb.scores(model, sender, judger, 2, 8)
    assert got["support"].shape == (sender.length,) and torch.isfinite(got["support"]).all()
    torch.testing.assert_close(logits_after(model, copy.deepcopy(sender.chain), judger), before)


def test_fanoutqa_scoring_and_recall():
    groups = fq.gold_groups("synthetic", {"Alaska": "20,310", "Texas": 8751.0})
    assert groups == (("alaska",), ("texas",), ("20,310",), ("8751",))
    assert fq.score(groups, "Alaska 20310 ft; Texas: 8,751.")["strict"] == 1.0
    assert fq.score(groups, "Alaska 20310")["loose"] == 0.5
    assert fq.visible("<think>Texas</think> Alaska") == "Alaska"
    assert fq.visible("<think>unclosed") == ""
    words = ["Tex", "zz", "as", " 8751"]
    def dec(ids):
        return "".join(words[i] for i in ids)
    text = fq.retained_text(list(range(4)), [0, 2, 3, 9], dec)  # 9: a latent row, skipped
    assert not fq.group_hits(groups, text)[1]  # "Tex" + "as" never join across a dropped token
    assert fq.group_hits(groups, text)[3]


@torch.no_grad()
def test_growable_cache_decodes_identically():
    """In-place appends give the same tokens as transformers' concatenating cache, including
    batch row drops during decoding and a deep-copied chain."""
    from gradproposal.latent.latentmas import Chain, decode
    from tests.test_latentmas import TOK

    model = tiny_model()
    prompts = ["abc", "a much longer prompt", "mid prompt"]

    def run():
        chain = Chain.empty(model, 3)
        a = decode(model, TOK, chain.clone(), prompts, max_new_tokens=300, temperature=0.6,
                   top_p=0.95, top_k=20, gen=torch.Generator().manual_seed(0), eos_ids=[5, 7])
        msgs = [model.get_input_embeddings()(torch.tensor(rand_ids(n, n))) for n in (40, 40)]
        b = decode(model, TOK, cb.receiver_chain(model, msgs, chunk=16), None,
                   prompt_ids=[rand_ids(5, 1)] * 2, max_new_tokens=50, temperature=0,
                   eos_ids=[-1])
        return a, b

    ref = run()
    cb.growable_kv(True)
    try:
        got = run()
    finally:
        cb.growable_kv(False)
    assert got == ref
