"""CacheBack against the official code on the same weights (run on a GPU node).

For a few panel questions and each sender: our prompts versus the official prompt builders; our
latent roll versus `rcc.latent.rollout.latent_rollout` from the same prefilled cache; our scores
versus `capture_qwen_scores`; our keeps versus `qwen_w16_keep`; our message rows versus
`interleaved_blocks`; and the receiver's next-token logits after [message rows][receiver prompt]
(official: `torch.cat((payload.rows, manager))` as input embeddings) versus ours (chunked prefill
of the rows, then the prompt ids).

    python -m gradproposal.latent.cb_check --ref RCLC_PAPER_V1 --out latent/results/cb-check.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import torch

from gradproposal.latent import bench as B
from gradproposal.latent import cacheback as cb
from gradproposal.latent.latentmas import extend


def rel(a, b) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def logit_gap(a, b) -> dict:
    a, b = a.float(), b.float()
    kl = torch.sum(torch.softmax(b, -1) * (torch.log_softmax(b, -1) - torch.log_softmax(a, -1)))
    return {"kl": float(kl), "argmax_agree": bool(a.argmax() == b.argmax()),
            "max_abs": float((a - b).abs().max())}


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="checkout of github.com/maxr0ssi/rclc @ paper-v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--evidence-chars", type=int, default=0, help="truncate evidence (0 = full)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, f"{a.ref}/src")
    from rcc.benchmarks.fanoutqa import prompts as P
    from rcc.latent.rollout import build_realign, latent_rollout
    from rcc.models.qwen.capture import capture_qwen_scores, qwen_w16_keep
    from rcc.topologies.fanout.layout import flat_interleave_layout, interleaved_blocks
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = cb.load_model(a.model)
    emb = model.get_input_embeddings()
    report = {"model": a.model, "evidence_chars": a.evidence_chars, "items": []}
    for item in B.load_fanoutqa(a.n):
        q = item["question"]
        rec = {"qid": item["id"], "workers": []}
        mgr_text = B.render(tok, B.manager_body(q))
        ref_mgr = P.manager_prompt(tok, q, "", enable_thinking=True)
        mgr_ids, lo, hi = B.question_span(tok, mgr_text, q)
        qids = P.question_token_ids(tok, ref_mgr, q)
        rec["manager_prompt_equal"] = ref_mgr == mgr_text
        rec["question_rows_equal"] = list(qids) == mgr_ids[lo:hi]
        ours_rows, keeps_ours, keeps_ref, prompts, rolled = [], [], [], [], []
        for w in item["workers"]:
            text = w["text"][: a.evidence_chars] if a.evidence_chars else w["text"]
            ids = tok(B.render(tok, B.worker_body(q, text)), add_special_tokens=False)["input_ids"]
            ev = tok(text, add_special_tokens=False)["input_ids"]
            ref_prompt = P.worker_prompt(tok, q, ev, report_style="natural", enable_thinking=True)
            head = cb.prefill_head(model, ids)
            past = copy.deepcopy(head.cache)
            sender = cb.roll(model, head, ids)
            ref = latent_rollout(model, torch.tensor([ids[-1:]], device=model.device),
                                 latent_steps=cb.LATENT_STEPS,
                                 realign=build_realign(model, enabled=False), past=past,
                                 record_embeds=True)
            ref_lat = ref.embeds[0, 1:]
            key_diff = max(float((x.keys.float() - y.keys.float()).abs().max())
                           for x, y in zip(sender.chain.cache.layers, ref.past.layers, strict=True))
            ours = cb.scores(model, sender, mgr_ids, lo, hi)
            theirs, found = capture_qwen_scores(
                model, ref.past, torch.tensor([mgr_ids], device=model.device),
                torch.ones(1, len(mgr_ids), dtype=torch.long, device=model.device), qids)
            wr = {"prompt_equal": ref_prompt == B.render(tok, B.worker_body(q, text)),
                  "T": sender.length, "question_found": bool(found),
                  "latent_max_abs_diff": float((sender.latents - ref_lat).float().abs().max()),
                  "cache_keys_max_abs_diff": key_diff,
                  "snap_rel_diff": rel(ours["snap"], theirs["snap"]),
                  "support_rel_diff": rel(ours["support"], theirs["support"])}
            for r in (4, 16):
                k_ours = cb.select(ours["support"], r)
                k_ref = list(qwen_w16_keep(theirs["support"], r))
                wr[f"keep{r}_port_equal"] = cb.select(theirs["support"], r) == k_ref
                wr[f"keep{r}_overlap"] = len(set(k_ours) & set(k_ref)) / len(k_ref)
                if r == 4:
                    keeps_ours.append(k_ours)
                    keeps_ref.append(tuple(k_ref))
            prompts.append(torch.tensor(ids))
            rolled.append(ref_lat)
            ours_rows.append(cb.message_rows(model, ids, sender.latents, keeps_ours[-1]))
            rec["workers"].append(wr)
            del head, past, sender, ref
            torch.cuda.empty_cache()
        layout = flat_interleave_layout(tuple(keeps_ref), tuple(len(p) for p in prompts))
        blocks, _, _ = interleaved_blocks(
            tuple(prompts), layout, tuple(keeps_ref), tuple(rolled),
            lambda t: emb(torch.tensor(t, device=model.device)))
        ref_payload = torch.cat(blocks).to(torch.bfloat16)
        our_payload = torch.cat(ours_rows)
        rec["keeps4_equal"] = [list(k) for k in keeps_ref] == keeps_ours
        if rec["keeps4_equal"]:
            diff = (our_payload.cpu() - ref_payload.cpu()).float().abs().max()
            rec["payload_max_abs_diff"] = float(diff)
        # Receiver: official input-embedding prefill of [official payload][manager prompt] ...
        mgr = emb(torch.tensor(mgr_ids, device=model.device))
        rows = torch.cat((ref_payload.to(mgr), mgr))[None]
        ref_logits = model(inputs_embeds=rows, logits_to_keep=1).logits[0, -1]
        # ... versus ours on the same payload (chunked receiver prefill, then the prompt ids).
        chain = cb.receiver_chain(model, [ref_payload])
        ids = torch.tensor([mgr_ids], device=model.device)
        ours_logits = model.lm_head(extend(model, chain, torch.ones_like(ids), ids=ids))[0]
        rec["receiver_positions"] = rows.shape[1]
        rec["receiver_logits"] = logit_gap(ours_logits, ref_logits)
        print(json.dumps(rec, indent=1), flush=True)
        report["items"].append(rec)
        del chain
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
