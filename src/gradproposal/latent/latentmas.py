"""Minimal LatentMAS (Zou et al., arXiv 2511.20639), Sections 3.1-3.3, on Hugging Face transformers.

Every agent is the same model. An agent reads its prompt, then takes `m` latent steps: the
post-norm last-layer hidden state h at the newest position is mapped to an input embedding
e = align(h) and fed back in, so each step adds one position to the key-value (KV) cache and no
token is sampled. The next agent continues from that cache (its positions follow on), adds its own
prompt and latent steps, and so on; only the last agent decodes text.

Batching: rows are left-padded per segment, so padding sits in the middle of the running cache.
`Chain.mask` tracks real (1) versus padded (0) cache slots and `Chain.next_pos` the next rotary
position per row, so a padded row sees exactly what it would see alone. The official code
instead attends to earlier agents' padding (see NOTEBOOK.md).

Hybrid models (Qwen3.5: Gated DeltaNet layers with a fixed-size recurrent and conv state, plus
full-attention layers with a KV cache) use the same code: transformers keeps both kinds of state in
one `DynamicCache`, so "continuing from A1's cache" hands A2 the KV and the recurrent states.
Padding would leak into the recurrent states, so hybrid chains must be unpadded (batch size 1).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin


@dataclass
class Chain:
    """Batched state handed from agent to agent: the cache plus per-row masks and positions."""

    cache: DynamicCache
    mask: torch.Tensor  # [B, T] long, 1 = real cache slot, 0 = padding
    next_pos: torch.Tensor  # [B] long, rotary position of the next real token

    @classmethod
    def empty(cls, model, batch: int) -> Chain:
        dev = model.device
        return cls(
            DynamicCache(config=model.config),
            torch.zeros(batch, 0, dtype=torch.long, device=dev),
            torch.zeros(batch, dtype=torch.long, device=dev),
        )

    @property
    def length(self) -> int:
        return self.mask.shape[1]

    def clone(self) -> Chain:
        return Chain(copy.deepcopy(self.cache), self.mask.clone(), self.next_pos.clone())

    def keep_rows(self, rows: torch.Tensor) -> None:
        """Drop all rows but `rows` (in place), e.g. finished sequences during decoding."""
        self.cache.batch_select_indices(rows)
        self.mask, self.next_pos = self.mask[rows], self.next_pos[rows]

    @property
    def hybrid(self) -> bool:
        return any(isinstance(layer, LinearAttentionCacheLayerMixin) for layer in self.cache.layers)


# ---------------------------------------------------------------- alignment (Section 3.1, App. A)


@torch.no_grad()
def alignment_matrix(model, ridge: float = 1e-5, realign: bool = True):
    """W_a = (W_out^T W_out + ridge I)^-1 W_out^T W_in in fp32, and the mean input-embedding norm.

    Same closed form and ridge as the official `_build_latent_realign_matrix`. With
    realign=False, W_a = I (the official default when --latent_space_realign is not passed);
    the norm rescale in `align` is applied either way, as in the official code.
    """
    w_in = model.get_input_embeddings().weight.detach().float()
    w_out = model.get_output_embeddings().weight.detach().float()
    target_norm = w_in.norm(dim=1).mean()
    if not realign:
        return torch.eye(w_in.shape[1], device=w_in.device), target_norm
    gram = w_out.T @ w_out + ridge * torch.eye(w_out.shape[1], device=w_out.device)
    return torch.linalg.solve(gram, w_out.T @ w_in), target_norm


def align(h: torch.Tensor, w_a: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Map last-layer hidden states [B, d] to input embeddings, rescaled to the mean token norm."""
    e = h.float() @ w_a
    e = e * (target_norm / e.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return e.to(h.dtype)


# ---------------------------------------------------------------- forward passes


def left_pad(tokenizer, texts: list[str] | None, device, rows: list[list[int]] | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded ids and mask for `texts`, or for already tokenized `rows`."""
    if rows is None:
        rows = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    width = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r) :] = torch.tensor(r)
        mask[i, width - len(r) :] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def extend(model, chain: Chain, seg_mask: torch.Tensor, ids=None, embeds=None) -> torch.Tensor:
    """Append a segment (token ids or input embeddings) to the chain; return last hidden [B, d]."""
    if chain.hybrid and not bool(seg_mask.all()):
        raise ValueError("hybrid (recurrent) caches cannot hold padding: use batch size 1")
    pos = chain.next_pos[:, None] + seg_mask.cumsum(1) - 1
    full_mask = torch.cat([chain.mask, seg_mask], dim=1)
    out = model.model(
        input_ids=ids,
        inputs_embeds=embeds,
        attention_mask=full_mask,
        position_ids=pos.clamp_min(0),
        past_key_values=chain.cache,
        use_cache=True,
    )
    chain.cache = out.past_key_values
    chain.mask = full_mask
    chain.next_pos = chain.next_pos + seg_mask.sum(1)
    return out.last_hidden_state[:, -1, :]


@torch.no_grad()
def latent_steps(model, chain: Chain, h: torch.Tensor, m: int, w_a, target_norm,
                 trace: list | None = None) -> torch.Tensor:
    """m latent steps (Section 3.1): feed align(h) back as the next input embedding.
    If `trace` is a list, the fed embeddings [B, 1, d] are appended to it (for checks)."""
    ones = torch.ones(h.shape[0], 1, dtype=torch.long, device=h.device)
    for _ in range(m):
        e = align(h, w_a, target_norm)[:, None, :]
        if trace is not None:
            trace.append(e)
        h = extend(model, chain, ones, embeds=e)
    return h


@torch.no_grad()
def run_latent_agent(model, tokenizer, chain: Chain, prompts: list[str], m: int, w_a, target_norm,
                     trace: list | None = None):
    """One non-final agent: read its prompt on top of the inherited cache, then think m steps."""
    ids, seg = left_pad(tokenizer, prompts, model.device)
    h = extend(model, chain, seg, ids=ids)
    latent_steps(model, chain, h, m, w_a, target_norm, trace)
    return chain


def _sample(logits: torch.Tensor, temperature: float, top_p: float, gen,
            top_k: int | None = None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float()
    if top_k:  # keep the k largest logits (applied before top-p, as in vLLM)
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    drop = sorted_p.cumsum(-1) - sorted_p > top_p  # keep the smallest prefix reaching top_p
    sorted_p = sorted_p.masked_fill(drop, 0.0)
    choice = torch.multinomial(sorted_p, 1, generator=gen)
    return idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def decode(
    model,
    tokenizer,
    chain: Chain,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    gen: torch.Generator | None = None,
    eos_ids: list[int] | None = None,
    top_k: int | None = None,
    prompt_chunk: int | None = None,
    prompt_ids: list[list[int]] | None = None,
    stats: dict | None = None,
) -> list[list[int]]:
    """Final agent: read the prompt on top of the chain, then sample (temperature 0 = greedy).
    `prompt_chunk` reads long prompts in column chunks to bound activation memory; `prompt_ids`
    gives the prompts already tokenized (`prompts` is then ignored). If `stats` is a dict, the
    prompt-reading time (seconds, synchronized) is stored under "prefill_s".

    Returns the generated ids per row, without the stop token.
    """
    if eos_ids is None:
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
    eos_t = torch.tensor(eos_ids, device=model.device)
    t0 = time.perf_counter()
    ids, seg = left_pad(tokenizer, prompts, model.device, prompt_ids)
    for s in range(0, ids.shape[1], prompt_chunk or ids.shape[1]):
        e = s + (prompt_chunk or ids.shape[1])
        h = extend(model, chain, seg[:, s:e], ids=ids[:, s:e])
    if stats is not None:
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        stats["prefill_s"] = time.perf_counter() - t0
    batch = ids.shape[0]
    out = torch.zeros(batch, max_new_tokens, dtype=torch.long, device=model.device)
    keep = torch.zeros(batch, max_new_tokens, dtype=torch.bool, device=model.device)
    active = torch.arange(batch, device=model.device)  # original row of each chain row
    dead = torch.zeros(batch, dtype=torch.bool, device=model.device)  # stopped, not yet dropped
    for step in range(max_new_tokens):
        tok = _sample(model.lm_head(h), temperature, top_p, gen, top_k)
        stop = torch.isin(tok, eos_t) & ~dead
        out[active, step] = tok
        keep[active, step] = ~dead & ~stop
        dead |= stop
        if bool(dead.all()):
            break
        # Finished rows cost a full forward each step; drop them once a quarter have finished.
        if not chain.hybrid and int(dead.sum()) * 4 >= len(active):
            alive = (~dead).nonzero().flatten()
            chain.keep_rows(alive)
            active, tok, dead = active[alive], tok[alive], dead[alive]
        h = extend(model, chain, torch.ones_like(tok)[:, None], ids=tok[:, None])
    out, keep = out.cpu(), keep.cpu()
    return [out[i][keep[i]].tolist() for i in range(batch)]


# ---------------------------------------------------------------- cache interventions


def _moment_noise(x: torch.Tensor, real: torch.Tensor, gen) -> torch.Tensor:
    """Gaussian noise shaped like x; mean and std match x per row (dim 0) over `real` entries."""
    dims = tuple(range(1, x.ndim))
    xf, real = x.float(), real.expand_as(x)
    count = real.sum(dim=dims).clamp_min(1)
    mean = (xf * real).sum(dim=dims) / count
    view = (-1,) + (1,) * (x.ndim - 1)
    std = ((((xf - mean.view(view)) * real) ** 2).sum(dim=dims) / count).sqrt()
    z = torch.randn(x.shape, generator=gen, device=x.device, dtype=torch.float32)
    return (mean.view(view) + std.view(view) * z).to(x.dtype)


@torch.no_grad()
def noise_like(chain: Chain, gen: torch.Generator | None = None) -> Chain:
    """Replace all transferred state with Gaussian noise of the same shape, matching the true
    tensor's mean and std per row, layer and tensor: every K and V (over real, unpadded slots) and,
    for hybrid models, every recurrent and conv state."""
    out = chain.clone()
    real_kv = out.mask.bool()[:, None, :, None]  # [B, 1, T, 1]
    for layer in out.cache.layers:
        for name in ("keys", "values"):
            x = getattr(layer, name, None)
            if x is not None and x.numel():
                setattr(layer, name, _moment_noise(x, real_kv, gen))
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for states in (layer.conv_states, layer.recurrent_states):
                for i, x in states.items():
                    if x is not None:
                        states[i] = _moment_noise(x, torch.ones_like(x, dtype=torch.bool), gen)
    return out
