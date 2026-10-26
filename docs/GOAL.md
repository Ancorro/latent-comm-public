# Goal: beat the published multi-agent score on MASBench Reasoning with a latent channel

Set 2026-10-26. A long arc (days to weeks), worked as a series of self-contained work packages. Not reaching the number is an acceptable outcome; the record of what was tried is the deliverable either way.

## The target

**Benchmark:** MASBench, Reasoning track (Chu et al., arXiv [2610.04672](https://arxiv.org/abs/2610.04672), 3 Oct 2026; code and data [BUPT-GAMMA/MASBench](https://github.com/BUPT-GAMMA/MASBench), data CC BY-SA 4.0, code MIT). Two agents, Alice and Bob, each hold a different part of a HotpotQA-derived evidence chain for the same question and cannot answer alone; they exchange messages and give one final answer. 300 questions: 200 simple ("easy"), 100 complex ("hard"). Scored by word-level F1 against the reference answer, reported x100.

**Number to beat: 59.24** (Qwen-Plus, Protocol mechanism; simple 61.37, complex 54.98). Other published rows: DeepSeek-v3.2 55.23, Gemini-2.5-Flash-Lite 55.53, GPT-5-nano 49.66, Llama-3.1-8B 33.45. No model is near saturation; the authors note backbone capability remains the deciding factor.

**Win condition: accuracy only.** Mean F1 (x100) over all 300 questions, paper scoring, above 59.24 with our method. Communication cost (bytes, positions, wall-clock, FLOPs) is recorded and reported beside every number but decides nothing.

## Fixed measurement rules

Set before any run so numbers stay comparable across tasks.

- **Paper settings for text runs:** Protocol style, at most 10 individual speaking turns, 4,096 output tokens per turn, temperature 0, full private context (`--mode context`), all 300 questions in dataset order.
- **Scoring:** a question with no recognized final answer scores 0 (the paper's rule). The public script omits such questions from its mean; we report both, the paper's rule is the headline. Scoring code: MASBench's `cal_f1_score` (SQuAD normalization), unchanged.
- **Our runner:** `private/masbench/run_reasoning.py` imports MASBench's prompts, conversation loop, loader and metric and only adds concurrency. Any method that changes what crosses between agents replaces the conversation loop, keeps the prompts, the turn cap and the scorer.
- **Subsets:** smoke runs use the stratified 30 (20 easy, 10 hard, seed 42). Decisions are made on all 300. Three seeds only where sampling is stochastic (temperature 0 runs need one).
- **Baselines on the same backbone, same training:** single agent with both halves (upper bound); single agent with one half (must be low); text Protocol as published; same-size text message; random cache (the causal-audit control). Anything our channel is trained on, the text baseline is trained on too.

## Model scale

- **Main backbone: Qwen3.8-27B** (dense; FP8 copy in the shared cache, 29 GB; bf16 52 GB). FP8 serves on one A40/L40S (48 GB); bf16 and all training need one H100/H200 80 GB. Chosen because it is the largest open model already cached and sits in the 27-32B band the goal asks for.
- **Second backbone, if time allows: Gemma-3-27B** (a different family, to show the channel is not Qwen-specific).
- **Development backbone: Qwen3-8B** (fast iteration; results on it never count toward the goal).
- Larger (Qwen3-32B dense at 130K context, 70B) only if 27B shows a clear gain and compute is free.

## Compute plan

- **OSU HPC first** (account eecs): A40/L40S for FP8 serving and evaluation; H100/H200 (dgxh, preempt) for bf16 and LoRA training. Jobs: `private/masbench/bench.sbatch` (serve + evaluate), `private/hpc/latent.sbatch` (manual forward passes, training).
- **AWS fallback:** `g6e.2xlarge` (L40S 48 GB) about $2/hour when the cluster queue stalls; a full 300-question text evaluation is under an hour.
- **Budget per evaluation (to be replaced by measured numbers from the smoke run):** 300 questions x about 3K generated tokens; with 8-16 questions in flight, 15-30 min of GPU time plus model load.
- Weights stay in the shared HF cache; never redownloaded.

## Milestones (one work package each, a few hours of focused work, CPU-checkable "Done when")

Each package has a CPU-only completion check (the test suite on tiny random models) so it can be verified anywhere. GPU runs happen between packages, and their results feed the next one.

| # | Task | Done when (CPU, tiny models) | GPU run after |
|---|---|---|---|
| 0 | **Text baselines** on Qwen3.8-27B: single-agent both halves, one half, Protocol as published. Measurement only, no code package; sets the local number to beat and the upper bound. | | 300 questions x 3 arms |
| 1 | **MASBench adapter**: load the two-agent split into our runner (`latent/`), run the LatentMAS-style full-cache hand-off and the wrong-cache control on it; score with MASBench's F1. | tests on tiny models: split loads, both agents see only their half, wrong-cache partners are a derangement, scorer matches MASBench on fixed pairs | full-cache vs text, 300 questions |
| 2 | ***N*-position latent summaries** per agent per turn (untrained): after reading its half, each agent writes N latent positions; only their KV crosses. Sweep N. | summary arm runs; receiver sees only the N positions plus the sink; mask correct under padding | N in {8, 32, 128} vs text |
| 3 | **Trained summary writer**: LoRA on the summary positions, frozen reader, trained on HotpotQA split the MASBench way (`papers/BENCHMARKS.md` augmentation plan). Same-size text baseline gets the same LoRA budget. | training loop runs on a tiny model and lowers receiver loss; receiver answers from the summary alone on a toy task | train on H100, evaluate on the 300 |
| 4 | **Two-way exchange**: alternate latent turns under the 10-turn cap, with a learned "ask" (query positions) so the partner's summary is conditioned on the need (CacheBack's idea, trained). | multi-turn loop on tiny models; turn cap honoured; query changes the partner's kept positions | full evaluation |
| 5 | **Report**: paired comparison of every arm against the published 59.24 and our own text baseline, tables generated from results files, communication cost beside each row. | `tex_tables.py` builds the tables from committed results; tests pin the numbers | |

Beyond 5 if the number is not reached: GRPO on the writer with team reward (`swe/` already has the GRPO loop), the second backbone, Scheduling track (5 agents) once routing exists.

## Record keeping

- Results: `latent/results/masbench/<arm>-<model>-<job>.jsonl`, one line per question plus a summary line; registry in `latent/results/runs.jsonl`; W&B project `latent-comm`, group `masbench`.
- Notebook: `latent/NOTEBOOK.md` gets a MASBench section, dated, every number linked to its file.
