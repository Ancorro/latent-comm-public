"""LatentMAS mechanics on a tiny random Qwen3 (CPU, seconds)."""

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3Config, Qwen3ForCausalLM

from gradproposal.latent.latentmas import (
    Chain,
    align,
    alignment_matrix,
    decode,
    extend,
    noise_like,
    run_latent_agent,
)
from gradproposal.latent.run import partners

V = 96


class CharTokenizer:
    """Stand-in tokenizer: one id per character, id 0 = padding."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + ord(c) % (V - 1) for c in text]}


def tiny_model(dtype=torch.float32, tied=True):
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=tied,
    )
    model = Qwen3ForCausalLM(cfg).to(dtype).eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


TOK = CharTokenizer()


def ids_of(text):
    return torch.tensor([TOK(text)["input_ids"]])


@torch.no_grad()
def test_m0_empty_cache_matches_plain_generation():
    """m = 0 and no predecessor: our decode path is plain greedy generation."""
    model = tiny_model()
    prompt = "Question: 2+3?"
    ours = decode(
        model, TOK, Chain.empty(model, 1), [prompt], max_new_tokens=12, temperature=0, eos_ids=[-1]
    )[0]
    ref = model.generate(ids_of(prompt), do_sample=False, max_new_tokens=12, min_new_tokens=12)
    assert ours == ref[0, len(prompt) :].tolist()

    # The first-step logits match a plain forward pass too.
    chain = Chain.empty(model, 1)
    ids = ids_of(prompt)
    h = extend(model, chain, torch.ones_like(ids), ids=ids)
    torch.testing.assert_close(model.lm_head(h), model(ids).logits[:, -1], atol=1e-5, rtol=1e-5)


@torch.no_grad()
def reference_full_pass(model, p1, p2, m, w_a, norm):
    """No cache: build A1's latent thoughts by re-running the whole sequence each step, then read
    [A1 prompt, thoughts, A2 prompt] in one forward pass (Theorem 3.3's right-hand side)."""
    emb = model.get_input_embeddings()
    seq = emb(ids_of(p1))
    for _ in range(m):
        h = model.model(inputs_embeds=seq).last_hidden_state[:, -1]
        seq = torch.cat([seq, align(h, w_a, norm)[:, None]], dim=1)
    seq = torch.cat([seq, emb(ids_of(p2))], dim=1)
    return model(inputs_embeds=seq).logits[:, -1].float()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
@pytest.mark.parametrize("m", [0, 4])
@torch.no_grad()
def test_theorem_3_3_cache_transfer_equals_concatenation(dtype, tol, m):
    """A2 continuing from A1's cache (prompt + m latent thoughts) gives the same next-token logits
    as one pass over A1's input, A1's thoughts and A2's prompt concatenated."""
    model = tiny_model(dtype)
    w_a, norm = alignment_matrix(model)
    p1, p2 = "Plan: add the numbers.", " Critic: check it."
    chain = run_latent_agent(model, TOK, Chain.empty(model, 1), [p1], m, w_a, norm)
    assert chain.length == len(p1) + m
    ids = ids_of(p2)
    h = extend(model, chain, torch.ones_like(ids), ids=ids)
    ours = model.lm_head(h).float()
    ref = reference_full_pass(model, p1, p2, m, w_a, norm)
    scale = ref.abs().max()
    assert (ours - ref).abs().max() <= tol * scale
    assert ours.argmax() == ref.argmax()


@torch.no_grad()
def test_batched_rows_match_unbatched():
    """Left padding inside the running cache must not change any row's result."""
    model = tiny_model()
    w_a, norm = alignment_matrix(model)
    qs = ["short q", "a much longer question here"]
    roles = [["P1 ", "C1 "], ["Planner prompt one ", "Critic two "]]

    def run(rows):
        chain = Chain.empty(model, len(rows))
        for r in range(2):
            chain = run_latent_agent(
                model, TOK, chain, [roles[r][i] + qs[i] for i in rows], 3, w_a, norm
            )
        prompts = ["Solve: " + qs[i] for i in rows]
        return decode(model, TOK, chain, prompts, max_new_tokens=8, temperature=0, eos_ids=[-1])

    both = run([0, 1])
    assert both[0] == run([0])[0]
    assert both[1] == run([1])[0]


@torch.no_grad()
def test_batched_decode_with_early_stops_matches_unbatched():
    """Rows that stop early are dropped from the cache mid-decode; the rest must be unaffected."""
    model = tiny_model()
    qs = ["Q: one", "Q: a longer second question", "Q: third", "Q: and the fourth one"]

    def run(rows, eos):
        return decode(model, TOK, Chain.empty(model, len(rows)), [qs[i] for i in rows],
                      max_new_tokens=10, temperature=0, eos_ids=eos)

    free = [run([i], [-1])[0] for i in range(4)]
    eos = [free[0][2], free[2][4]]  # rows 0 and 2 stop early
    both = run([0, 1, 2, 3], eos)
    assert both == [run([i], eos)[0] for i in range(4)]
    assert len(both[0]) < 10 and len(both[2]) < 10


@torch.no_grad()
def test_noise_matches_moments():
    model = tiny_model()
    w_a, norm = alignment_matrix(model)
    qs = ["first", "second one"]
    chain = run_latent_agent(model, TOK, Chain.empty(model, 2), qs, 2, w_a, norm)
    noisy = noise_like(chain, torch.Generator().manual_seed(0))
    real = chain.mask[1].bool()
    for layer, nlayer in zip(chain.cache.layers, noisy.cache.layers, strict=True):
        x, y = layer.values[1][:, real], nlayer.values[1][:, real]
        assert y.shape == x.shape and not torch.allclose(x, y)
        assert abs(float(x.std() - y.std())) < 0.1 * float(x.std())
        assert abs(float(x.mean() - y.mean())) < 0.25 * float(x.std())  # ~400 samples


def test_wrong_cache_partners_are_a_derangement():
    assert partners(5, 32) == [1, 2, 3, 4, 0]
    p = partners(300, 32)
    assert sorted(p) == list(range(300)) and all(i != j for i, j in enumerate(p))
    assert all(i // 32 == j // 32 for i, j in enumerate(p))  # partner stays in its group


def test_alignment_is_identity_for_tied_embeddings():
    """With W_out = W_in (tied, as in Qwen3-4B) the least-squares map collapses to about I."""
    w_a, _ = alignment_matrix(tiny_model(tied=True))
    torch.testing.assert_close(w_a, torch.eye(64), atol=1e-2, rtol=0)  # ridge shrinks slightly
    w_a, _ = alignment_matrix(tiny_model(tied=False))
    assert (w_a - torch.eye(64)).norm() > 1.0


# ---------------------------------------------------------------- hybrid (Qwen3.5-style) path


def tiny_hybrid(dtype=torch.float32):
    """Three Gated DeltaNet layers (recurrent state, no KV) and one full-attention layer."""
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        tie_word_embeddings=True,
    )
    model = Qwen3_5ForCausalLM(cfg).to(dtype).eval()
    model.generation_config.eos_token_id = None
    return model


@pytest.mark.parametrize("m", [0, 4])
@torch.no_grad()
def test_hybrid_theorem_3_3(m):
    """Handing A2 both the attention KV and the recurrent states equals one pass over the
    concatenated input embeddings."""
    model = tiny_hybrid()
    w_a, norm = alignment_matrix(model)
    p1, p2 = "Plan: add the numbers.", " Critic: check it."
    chain = run_latent_agent(model, TOK, Chain.empty(model, 1), [p1], m, w_a, norm)
    assert chain.hybrid
    ids = ids_of(p2)
    ours = model.lm_head(extend(model, chain, torch.ones_like(ids), ids=ids))
    ref = reference_full_pass(model, p1, p2, m, w_a, norm)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_hybrid_rejects_padding_and_noise_covers_recurrent_state():
    model = tiny_hybrid()
    w_a, norm = alignment_matrix(model)
    with pytest.raises(ValueError, match="padding"):
        run_latent_agent(model, TOK, Chain.empty(model, 2), ["a", "longer"], 1, w_a, norm)
    chain = run_latent_agent(model, TOK, Chain.empty(model, 1), ["plan"], 2, w_a, norm)
    noisy = noise_like(chain, torch.Generator().manual_seed(0))
    for layer, nlayer in zip(chain.cache.layers, noisy.cache.layers, strict=True):
        if hasattr(layer, "recurrent_states"):
            x, y = layer.recurrent_states[0], nlayer.recurrent_states[0]
            assert y.shape == x.shape and not torch.allclose(x, y)
