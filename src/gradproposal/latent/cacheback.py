"""Minimal CacheBack (Rossi et al., arXiv 2609.32046) on Hugging Face transformers.

A sender reads its private evidence, then takes 40 latent steps (LatentMAS-style; for Qwen the
hidden state is only rescaled to the mean token-embedding norm, no alignment matrix). The receiver's
query q (its own rendered prompt) is appended to a copy of the sender's cache; attention from the
question rows of q to the sender positions scores every position:

    Q_t = Pool7( sum_layers mean_{heads, rows} a )                       mean query attention
    E_t = sum_{layers, kv heads} Pool7(mean_rows b)^2                    b = a averaged over the
    R_t = sum_{layers, kv heads} mean_rows Pool7(b)^2                      query heads of a KV head
    S_t = Q_t (R_t / E_t)^2                                              CacheBack (p = alpha = 2)

Position 0 and the 40 latent positions are always kept; the rest of a budget ceil(T / r) is filled
with 16-wide spans ranked by mean score. The message is continuous rows: the input embeddings of the
kept positions (token embeddings for prompt tokens, the fed vectors for latent steps). The receiver
prefills [sender 0 rows][sender 1 rows][sender 2 rows][its prompt] from scratch at positions 0..N-1;
no key-value (KV) state crosses agents. Paper Appendix C; official code github.com/maxr0ssi/rclc
(tag paper-v1: `rcc/models/qwen/{engine,capture}.py`, `rcc/transforms/select/query_support/`).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from gradproposal.latent.latentmas import Chain, extend

LATENT_STEPS = 40
SPAN = 16
POOL = 7
MOMENT_SCALE = 256.0  # official SUPPORT_MOMENT_SCALE: keeps squared weights off fp32 underflow
QWEN_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32_768}


def load_model(name: str, dtype=torch.bfloat16, yarn: bool = True, device=None):
    """Qwen3 with the paper's served window: YaRN x4 over 32,768 (131,072 positions)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(name)
    if yarn:
        config.rope_parameters = {**config.rope_parameters, **QWEN_YARN}
        config.max_position_embeddings = 131_072
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return AutoModelForCausalLM.from_pretrained(
        name, config=config, dtype=dtype, attn_implementation="sdpa").to(device).eval()


def _growable_update(self, key_states, value_states, *args, **kwargs):
    """DynamicLayer.update that appends into a preallocated buffer (grown by 25%) instead of
    concatenating: `torch.cat` copies the whole cache on every decoding step. `keys`/`values`
    stay views of the filled part, so crop, batch selection and deepcopy keep working (a
    replaced tensor is detected by its data pointer and re-buffered)."""
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    n = self.keys.shape[-2] if self.keys.numel() else 0
    add = key_states.shape[-2]
    kb = getattr(self, "_kbuf", None)
    if (kb is None or n + add > kb.shape[-2] or kb.shape[0] != key_states.shape[0]
            or (n and kb.data_ptr() != self.keys.data_ptr())):
        cap = n + add + max(256, (n + add) // 4)
        b, h, _, d = key_states.shape
        kb, vb = (key_states.new_empty(b, h, cap, d), value_states.new_empty(b, h, cap, d))
        if n:
            kb[:, :, :n], vb[:, :, :n] = self.keys, self.values
        self._kbuf, self._vbuf = kb, vb
    self._kbuf[:, :, n : n + add] = key_states
    self._vbuf[:, :, n : n + add] = value_states
    self.keys, self.values = self._kbuf[:, :, : n + add], self._vbuf[:, :, : n + add]
    return self.keys, self.values


def growable_kv(enable: bool = True) -> None:
    """Switch every transformers DynamicLayer to in-place appends (or back)."""
    from transformers.cache_utils import DynamicLayer

    if not hasattr(DynamicLayer, "_cat_update"):
        DynamicLayer._cat_update = DynamicLayer.update
    DynamicLayer.update = _growable_update if enable else DynamicLayer._cat_update


def target_norm(model) -> torch.Tensor:
    w = model.get_input_embeddings().weight
    return torch.linalg.vector_norm(w.detach(), dim=-1, dtype=torch.float32).mean()


def rescale(h: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    """Official `apply_realign` with the identity map, in the hidden state's dtype."""
    return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6) * norm.to(h)


@torch.no_grad()
def extend_chunked(model, chain: Chain, seg_mask, ids=None, embeds=None, chunk: int | None = None):
    """`extend` over column chunks of one segment (bounds activation memory on long prompts)."""
    width = seg_mask.shape[1]
    if not chunk or width <= chunk:
        return extend(model, chain, seg_mask, ids=ids, embeds=embeds)
    for s in range(0, width, chunk):
        h = extend(model, chain, seg_mask[:, s : s + chunk],
                   ids=None if ids is None else ids[:, s : s + chunk],
                   embeds=None if embeds is None else embeds[:, s : s + chunk])
    return h


@dataclass
class Sender:
    chain: Chain  # cache over prompt + latent positions (batch 1)
    prompt_ids: list[int]
    latents: torch.Tensor  # [steps, d], the fed latent vectors

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + self.latents.shape[0]


@torch.no_grad()
def produce(model, prompt_ids: list[int], steps: int = LATENT_STEPS, chunk: int = 8192) -> Sender:
    """Sender: prefill the prompt (last token held back as the seed, as in the official
    `produce_hop`), feed the seed, then take `steps` latent steps."""
    return roll(model, prefill_head(model, prompt_ids, chunk), prompt_ids, steps)


@torch.no_grad()
def prefill_head(model, prompt_ids: list[int], chunk: int | None = 8192) -> Chain:
    """The sender prompt minus its last token (the official extraction prefill)."""
    ids = torch.tensor([prompt_ids[:-1]], device=model.device)
    chain = Chain.empty(model, 1)
    if ids.shape[1]:
        extend_chunked(model, chain, torch.ones_like(ids), ids=ids, chunk=chunk)
    return chain


@torch.no_grad()
def roll(model, chain: Chain, prompt_ids: list[int], steps: int = LATENT_STEPS) -> Sender:
    """Feed the held-back seed token, then take `steps` norm-matched latent steps."""
    dev = model.device
    seed = torch.tensor([prompt_ids[-1:]], device=dev)
    h = extend(model, chain, torch.ones_like(seed), ids=seed)
    norm, fed, one = target_norm(model), [], torch.ones(1, 1, dtype=torch.long, device=dev)
    for _ in range(steps):
        e = rescale(h, norm)[:, None]
        fed.append(e[0, 0])
        h = extend(model, chain, one, embeds=e)
    lat = torch.stack(fed) if fed else torch.zeros(0, model.config.hidden_size, device=dev)
    return Sender(chain, list(prompt_ids), lat)


# ---------------------------------------------------------------- receiver-conditioned scoring

_IMPL = "cacheback_capture"


class _Fold:
    """Per-layer reduction of the question rows' attention over the T sender positions."""

    def __init__(self, memory: int, lo: int, hi: int):
        self.T, self.lo, self.hi = memory, lo, hi
        self.votes = self.energy = self.row_energy = None
        self.heads, self.layers = 0, 0

    def __call__(self, query, key, scaling):
        T, lo, hi = self.T, self.lo, self.hi
        rows, heads, kv = hi - lo, query.shape[1], key.shape[1]
        groups = heads // kv
        k = key[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2)  # repeat_kv
        logits = (query[:, :, lo:hi] @ k.transpose(2, 3) * scaling)[0]  # [H, rows, T + hi]
        q_pos = torch.arange(lo, hi, device=logits.device)[:, None] + T
        k_pos = torch.arange(logits.shape[-1], device=logits.device)[None]
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
        w = torch.softmax(logits, dim=-1, dtype=torch.float32)[..., :T]  # [H, rows, T]
        g = w.view(kv, groups, rows, T).mean(1)  # b_{s,r,t}
        votes = w.sum((0, 1))
        energy = _moment(pool(g.sum(1) / rows)).sum(0)
        row_energy = _moment(pool(g)).sum((0, 1)) / rows
        self.votes = votes if self.votes is None else self.votes + votes
        self.energy = energy if self.energy is None else self.energy + energy
        self.row_energy = row_energy if self.row_energy is None else self.row_energy + row_energy
        self.heads, self.layers = heads, self.layers + 1


_active: _Fold | None = None
_full_attention_layers: set[int] | None = None


def _capture_attention(module, query, key, value, attention_mask, **kwargs):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    layer = getattr(module, "layer_idx", None)
    if _active is not None and (_full_attention_layers is None or layer in _full_attention_layers):
        _active(query, key, kwargs.get("scaling") or module.scaling)
    return out


ALL_ATTENTION_FUNCTIONS.register(_IMPL, _capture_attention)
ALL_MASK_ATTENTION_FUNCTIONS.register(_IMPL, sdpa_mask)


def pool(x: torch.Tensor) -> torch.Tensor:
    """Max-pool the last (position) axis over 7 neighbours, per leading stream."""
    n = x.shape[-1]
    y = F.max_pool1d(x.reshape(-1, 1, n), POOL, stride=1, padding=POOL // 2)
    return y[..., :n].reshape(x.shape)


def _moment(x: torch.Tensor) -> torch.Tensor:
    return (x * MOMENT_SCALE).pow(2.0)


@torch.no_grad()
def scores(model, sender: Sender, judger_ids: list[int], lo: int, hi: int) -> dict:
    """Mean query attention ("snap") and CacheBack ("support") scores over the sender's T
    positions. The receiver prompt rows [0, hi) are appended to the sender cache and attention
    from the question rows [lo, hi) is folded; the cache is cropped back afterwards."""
    global _active, _full_attention_layers
    chain, T = sender.chain, sender.chain.length
    assert T == sender.length
    types = getattr(model.config, "layer_types", None)
    _full_attention_layers = (None if types is None else
                              {i for i, t in enumerate(types) if t == "full_attention"})
    ids = torch.tensor([judger_ids[:hi]], device=model.device)
    pos = chain.next_pos[:, None] + torch.arange(hi, device=model.device)
    impl = model.config._attn_implementation
    # Recurrent (hybrid) states cannot be cropped back, so score a copy; KV is cropped in place.
    cache = copy.deepcopy(chain.cache) if chain.hybrid else chain.cache
    _active = fold = _Fold(T, lo, hi)
    try:
        model.set_attn_implementation(_IMPL)
        model.model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, T + hi, dtype=torch.long, device=model.device))
    finally:
        _active = None
        model.set_attn_implementation(impl)
        if not chain.hybrid:
            chain.cache.crop(-hi)  # drop the appended receiver-prompt rows
    assert fold.layers > 0
    snap = pool(fold.votes / (fold.heads * (hi - lo)))
    tiny = torch.finfo(torch.float32).tiny
    ratio = (fold.row_energy / fold.energy.clamp_min(tiny)).clamp_min(tiny)
    ratio = torch.where((fold.row_energy == 0) & (fold.energy == 0), 1.0, ratio)
    out = {"snap": snap, "support": snap * ratio.pow(2.0)}
    for v in out.values():
        v[0] = v.max()  # pin the first position (official _pin_sink)
    return {k: v.float().cpu() for k, v in out.items()}


# ---------------------------------------------------------------- selection and message


def budget(length: int, ratio: int) -> int:
    return math.ceil(length / ratio)


def fixed_span_keep(scores: torch.Tensor, budget: int, protected, span: int = SPAN) -> list[int]:
    """Exactly `budget` positions: protected ones, then whole spans of a fixed grid in descending
    mean score (ties: earlier span); the overshooting span contributes the contiguous run of free
    positions nearest its peak. Port of the official `fixed_span_keep`."""
    n = int(scores.numel())
    target = min(n, max(1, int(budget)))
    keep = {int(p) for p in protected if 0 <= int(p) < n}
    if len(keep) > target:
        raise ValueError("protected positions exceed the budget")
    if target == n:
        return list(range(n))
    vals = scores.float()
    bounds = [(s, min(n, s + span)) for s in range(0, n, span)]
    ranked = sorted(range(len(bounds)),
                    key=lambda i: (-float(vals[bounds[i][0] : bounds[i][1]].mean()), i))
    for i in ranked:
        room = target - len(keep)
        if room == 0:
            break
        s, e = bounds[i]
        free = [p for p in range(s, e) if p not in keep]
        if not free:
            continue
        if len(free) <= room:
            keep.update(free)
            continue
        peak = s + int(torch.argmax(vals[s:e]))
        best = min((abs((free[o] + free[o + room - 1]) / 2 - peak), free[o + room - 1] - free[o],
                     free[o]) for o in range(len(free) - room + 1))
        keep.update(range(best[2], best[2] + best[1] + 1))
        break
    assert len(keep) == target
    return sorted(keep)


def select(score: torch.Tensor, ratio: int, steps: int = LATENT_STEPS) -> list[int]:
    """Relative budget ceil(T / r) with position 0 and the latent tail protected."""
    n = int(score.numel())
    protected = (0, *range(max(1, n - steps), n))
    return fixed_span_keep(score, budget(n, ratio), protected)


@torch.no_grad()
def message_rows(model, prompt_ids: list[int], latents: torch.Tensor, keep: list[int]):
    """One sender's continuous rows: kept prompt tokens' embeddings in source order, then all
    latent vectors (official `interleaved_blocks`, flat-interleave layout)."""
    emb = model.get_input_embeddings()
    kept = [prompt_ids[p] for p in keep if p < len(prompt_ids)]
    tok = emb(torch.tensor(kept, dtype=torch.long, device=emb.weight.device))
    return torch.cat([tok, latents.to(tok)], 0)


def noise_rows(rows: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """Gaussian rows matching each row's mean and standard deviation over the hidden axis."""
    x = rows.float()
    mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, correction=0)
    z = torch.randn(x.shape, generator=gen, device=x.device)
    return (mean + std * z).to(rows.dtype)


@torch.no_grad()
def receiver_chain(model, messages: list[torch.Tensor], chunk: int = 8192) -> Chain:
    """Prefill each row's message (left-padded across the batch) into a fresh chain; the
    receiver prompt is then read and decoded by `latentmas.decode`."""
    dev, d = model.device, model.config.hidden_size
    width = max(m.shape[0] for m in messages)
    embeds = torch.zeros(len(messages), width, d, dtype=model.dtype, device=dev)
    mask = torch.zeros(len(messages), width, dtype=torch.long, device=dev)
    for i, m in enumerate(messages):
        embeds[i, width - m.shape[0] :] = m.to(dev, model.dtype)
        mask[i, width - m.shape[0] :] = 1
    chain = Chain.empty(model, len(messages))
    if width:
        extend_chunked(model, chain, mask, embeds=embeds, chunk=chunk)
    return chain
