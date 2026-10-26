"""Real-model checks for the LatentMAS implementation (run on a GPU node):

1. Theorem 3.3 in bf16: solver logits after the planner -> critic -> refiner cache chain versus one
   forward pass over the same input embeddings (prompts and latent thoughts) concatenated.
2. Batched rows (left padding inside the cache) versus each row alone.
3. Against the official code (github.com/Gen-Verse/LatentMAS, `--ref` = a clone): its
   `ModelWrapper.generate_latent_batch` on the same weights, batch size 1, same prompts and m;
   compares the final caches and the solver's next-token logits.

    python -m gradproposal.latent.check --ref /path/to/LatentMAS --out latent/results/check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import torch

from gradproposal.latent import bench as B
from gradproposal.latent.latentmas import Chain, align, extend, left_pad, run_latent_agent
from gradproposal.latent.run import Runner


def logit_gap(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    kl = torch.sum(torch.softmax(b, -1) * (torch.log_softmax(b, -1) - torch.log_softmax(a, -1)))
    return {"max_abs": float((a - b).abs().max()), "ref_max_abs": float(b.abs().max()),
            "argmax_agree": bool(a.argmax() == b.argmax()), "kl": float(kl)}


@torch.no_grad()
def solver_logits(r: Runner, q: str, bench: str, m: int) -> torch.Tensor:
    chain = r.chain([q], bench, m)
    ids = r.tok(r.chat(B.latent_messages("judger", q, bench), r.think), add_special_tokens=False,
                return_tensors="pt")["input_ids"].to(r.model.device)
    h = extend(r.model, chain, torch.ones_like(ids), ids=ids)
    return r.model.lm_head(h)[0], chain  # chain now ends with the judger prompt


@torch.no_grad()
def theorem_3_3(r: Runner, q: str, bench: str, m: int) -> dict:
    """Right-hand side rebuilt without any cache reuse: thoughts recomputed by full passes, so in
    bf16 the two paths can feed slightly different thoughts (see theorem_3_3_exact)."""
    model, emb = r.model, r.model.get_input_embeddings()

    def embed(text):
        ids = r.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return emb(ids.to(model.device))

    seq = None
    for role in B.ROLES[:-1]:
        e = embed(r.chat(B.latent_messages(role, q, bench), r.think))
        seq = e if seq is None else torch.cat([seq, e], 1)
        for _ in range(m):
            h = model.model(inputs_embeds=seq).last_hidden_state[:, -1]
            seq = torch.cat([seq, align(h, r.w_a, r.norm)[:, None]], 1)
    seq = torch.cat([seq, embed(r.chat(B.latent_messages("judger", q, bench), r.think))], 1)
    ref = model(inputs_embeds=seq, logits_to_keep=1).logits[0, -1]
    ours, _ = solver_logits(r, q, bench, m)
    return logit_gap(ours, ref) | {"positions": seq.shape[1]}


@torch.no_grad()
def theorem_3_3_exact(r: Runner, q: str, bench: str, m: int) -> dict:
    """Theorem 3.3 proper: the reference reads exactly the embeddings the cached path fed (prompts
    and recorded latent thoughts) in one pass; compare logits at every solver-prompt position."""
    model, emb = r.model, r.model.get_input_embeddings()

    def ids_of(text):
        return r.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            model.device)

    chain, parts = Chain.empty(model, 1), []
    for role in B.ROLES[:-1]:
        prompt, trace = r.chat(B.latent_messages(role, q, bench), r.think), []
        chain = run_latent_agent(model, r.tok, chain, [prompt], m, r.w_a, r.norm, trace)
        parts += [emb(ids_of(prompt)), *trace]
    judger = ids_of(r.chat(B.latent_messages("judger", q, bench), r.think))
    n = judger.shape[1]
    pos = chain.next_pos[:, None] + torch.arange(n, device=model.device)
    out = model.model(input_ids=judger, attention_mask=torch.cat([chain.mask, torch.ones_like(
        judger)], 1), position_ids=pos, past_key_values=chain.cache, use_cache=True)
    ours = model.lm_head(out.last_hidden_state[0]).float()
    seq = torch.cat([*parts, emb(judger)], 1)
    ref = model(inputs_embeds=seq, logits_to_keep=n).logits[0].float()
    kl = (torch.softmax(ref, -1) * (torch.log_softmax(ref, -1) - torch.log_softmax(ours, -1))
          ).sum(-1)
    return {"positions_compared": n, "max_abs": float((ours - ref).abs().max()),
            "ref_max_abs": float(ref.abs().max()), "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "argmax_agree_frac": float((ours.argmax(-1) == ref.argmax(-1)).float().mean())}


@torch.no_grad()
def batched(r: Runner, qs: list[str], bench: str, m: int) -> list[dict]:
    chain = r.chain(qs, bench, m)
    prompts = [r.chat(B.latent_messages("judger", q, bench), r.think) for q in qs]
    ids, seg = left_pad(r.tok, prompts, r.model.device)
    hb = r.model.lm_head(extend(r.model, chain, seg, ids=ids))
    return [logit_gap(hb[i], solver_logits(r, q, bench, m)[0]) for i, q in enumerate(qs)]


@torch.no_grad()
def official(r: Runner, ref_dir: str, q: str, bench: str, m: int) -> dict:
    sys.path.insert(0, ref_dir)
    import models as ref_models  # noqa: I001  official models.py (its vLLM import is optional)
    import prompts as ref_prompts

    # The official code targets transformers 4.x: `past[0][0]` fails on 5.x caches. Shim it.
    ref_models._past_length = lambda past: past.get_seq_length() if past is not None else 0
    args = types.SimpleNamespace(latent_space_realign=True, model_name="Qwen/Qwen3-4B",
                                 task=bench, think=r.think)
    w = ref_models.ModelWrapper.__new__(ref_models.ModelWrapper)
    w.model, w.tokenizer, w.device, w.args = r.model, r.tok, r.model.device, args
    w.latent_space_realign, w._latent_realign_matrices, w.pre_aligned = True, {}, None
    w_a_ref, norm_ref = w._ensure_latent_realign_matrix(r.model, r.model.device, args)
    past = None
    for role in B.ROLES[:-1]:
        msgs = ref_prompts.build_agent_message_sequential_latent_mas(
            role, q, method="latent_mas", args=args)
        assert msgs == B.latent_messages(role, q, bench)
        prompt = w.render_chat(msgs) + ("<think>" if r.think else "")
        enc = r.tok([prompt], return_tensors="pt", add_special_tokens=False)
        past = w.generate_latent_batch(enc["input_ids"].to(w.device),
                                       attention_mask=enc["attention_mask"].to(w.device),
                                       latent_steps=m, past_key_values=past)
    chain = r.chain([q], bench, m)  # before the judger prompt is appended
    ours_logits, _ = solver_logits(r, q, bench, m)
    cache_len = [chain.length, past.get_seq_length()]
    cache_diff = max(float((a.keys.float() - b.keys.float()).abs().max())
                     for a, b in zip(chain.cache.layers, past.layers, strict=True))
    judger = w.render_chat(ref_prompts.build_agent_message_sequential_latent_mas(
        "judger", q, method="latent_mas", args=args)) + ("<think>" if r.think else "")
    ids = r.tok([judger], return_tensors="pt", add_special_tokens=False)["input_ids"].to(w.device)
    pos = torch.arange(past.get_seq_length(), past.get_seq_length() + ids.shape[1],
                       device=w.device)[None]
    ref_logits = r.model(input_ids=ids, past_key_values=past, position_ids=pos,
                         logits_to_keep=1).logits[0, -1]
    return {"w_a_max_abs_diff": float((w_a_ref - r.w_a).abs().max()),
            "target_norm_diff": float(norm_ref - r.norm),
            "cache_len": cache_len,
            "cache_keys_max_abs_diff": cache_diff} | logit_gap(ours_logits, ref_logits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ref", default="", help="clone of github.com/Gen-Verse/LatentMAS")
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = Runner(a.model, realign=True, max_new_tokens=0, temperature=0.6, top_p=0.95, think=True,
               dtype=getattr(torch, a.dtype))
    cfg = r.model.config
    report = {"model": a.model, "m": a.m, "dtype": a.dtype,
              "tie_word_embeddings": cfg.tie_word_embeddings,
              "w_a_minus_I_fro": float((r.w_a - torch.eye(r.w_a.shape[0], device=r.w_a.device))
                                       .norm()),
              "target_norm": float(r.norm)}
    hybrid = Chain.empty(r.model, 1).hybrid
    report["hybrid"] = hybrid
    for bench in B.BENCHES:
        qs = [x["question"] for x in B.load(bench, limit=3)]
        report[bench] = {
            "theorem_3_3_exact": [theorem_3_3_exact(r, q, bench, a.m) for q in qs[:2]],
            "theorem_3_3_exact_m0": theorem_3_3_exact(r, qs[0], bench, 0),
            "theorem_3_3": [theorem_3_3(r, q, bench, a.m) for q in qs[:2]],
            "theorem_3_3_m0": theorem_3_3(r, qs[0], bench, 0),
        }
        if not hybrid:  # hybrid chains are unpadded (batch 1); the official code is dense-only
            report[bench]["batched_vs_single"] = batched(r, qs, bench, a.m)
            report[bench]["official"] = [official(r, a.ref, q, bench, a.m) for q in qs]
        print(json.dumps(report[bench], indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
