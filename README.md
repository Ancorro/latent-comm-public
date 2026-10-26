# latent-comm
Latent communication between LLM (large language model) agents: LatentMAS reproduction, cache-content controls, split-information probe and CacheBack. Code moved here from `gradProposal` on 2026-10-26; the GRFP and PhD writing stays there.
LLM (large language model) architecture experiments. What "state transfer" means here is still open; first we watch LLM agents act in open RL (reinforcement learning) environments.

## Decisions

- **Environments: GEM** ([github.com/axon-rl/gem](https://github.com/axon-rl/gem), [arXiv 2510.01051](https://arxiv.org/abs/2510.01051)). One gym-style `reset`/`step` API for multi-turn games and single-turn Reasoning Gym puzzles.
- **Agent: Claude through the Claude Code CLI** (`claude -p`, tools off), using the CLI login. No API key; inference is remote, so nothing heavy runs locally.
- **State between turns:** `render_prompt` in `src/gradproposal/run.py` replays the full history each turn. That function is the hand-off point for architecture experiments.
- **Runs in WSL (Windows Subsystem for Linux), not native Windows.** Reasoning Gym needs `pycosat`, which only ships as C source; Windows here has no compiler.
- **Compute:** anything running a model locally (training, local model rollouts) goes to the HPC (high-performance computing) cluster, never the desktop.
- **Open models run on HPC:** `private/hpc/run.sbatch` serves the model with vLLM (a fast inference server) on one GPU and plays episodes in the same job. Default model: Qwen3.5-4B.
- **Hosted open models: Amazon Bedrock** (`--agent bedrock --model google.gemma-3-4b-it`), serverless and paid per token; text in and out only, so no model internals. Needs `aws login` in WSL.
- **Thinking:** a reasoning model's `<think>` text is logged and shown folded, but stripped before the environment and later turns see it, as Qwen's own chat template does.

Goal and milestones: `docs/GOAL.md` (beat 59.24 F1 on MASBench Reasoning with a latent channel).

## Setup and check (CPU, any Linux)

```bash
bash setup.sh        # .venv/ with CPU torch, about 2 minutes; Python 3.12
bash check.sh        # test suite (tiny random Qwen3 models, stub agents) and ruff, about 30 s
```

Everything under `src/` and `tests/` runs on a CPU; real-model runs (Qwen3-4B) need a GPU and the cluster scripts in `private/`.

## Run (from WSL)

```bash
cd /mnt/c/<path to>/latent-comm
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/gradproposal   # keeps the Linux venv off the Windows folder
uv sync
uv run python -m gradproposal.run --envs game:GuessTheNumber-v0,game:Wordle-v0-easy --n 2
uv run python -m gradproposal.viewer --port 8790          # web page of agent I/O (reads runs/*.jsonl)
uv run pytest && uv run ruff check .
```

Open model on HPC (from Git Bash on Windows; deploys the committed code, submits, streams the log into `runs/`):

```bash
private/scripts/hpc_run.sh --envs game:GuessTheNumber-v0,game:Wordle-v0-easy --n 2 [--no-think]
```

- Swap models via the `MODEL`/`NAME` defaults in `private/hpc/run.sbatch`; the model must already be in the shared Hugging Face cache (jobs run offline).
- The job shares the `eecs` group GPU cap with other work; it waits as `QOSGrpGRES` while that cap is full.

- Episodes append to `runs/episodes.jsonl` (gitignored): one snapshot per turn, same `id`; readers keep the last.
- Interactive reinforcement-learning lesson built on the logged episodes: http://localhost:8790/lesson (`src/gradproposal/lesson.html`, plain HTML/JS plus KaTeX from a CDN).
- Software-engineering RL (reinforcement learning) on the desktop GPU: `swe/NOTEBOOK.md` (setup, decisions, results), code in `src/gradproposal/swe/`, episodes shown live on the I/O page.
- Papers on passing hidden state between agents, with notes: `papers/NOTES.md` (PDFs alongside).
- RL visuals page: http://localhost:8790/visuals (`src/gradproposal/visuals.html`): in-browser Q-learning maze, REINFORCE CartPole, and embedded videos of game-playing RL agents.
- Environment ids: `game:Wordle-v0-easy`, `game:Mastermind-v0`, `game:Hangman-v0`, `game:Minesweeper-v0-easy`, `game:Sudoku-v0-easy`, `rg:<reasoning gym task>`; full list in `gem.envs.registration.ENV_REGISTRY`.

## Tracking (W&B)

- **Hosted W&B cloud only.** GPU runs of `gradproposal.latent.run`, `split` and `cb_run` log to https://api.wandb.ai. `gradproposal.tracking.require_wandb()` stops a run before any data or model loads unless `WANDB_MODE` is unset or `online`, `WANDB_BASE_URL` is unset or `https://api.wandb.ai`, and credentials exist (`WANDB_API_KEY`, or a netrc entry for `api.wandb.ai`; `NETRC` picks the file). A failed W&B call fails the run. `private/hpc/latent.sbatch` exports the mode and URL; keys stay in `~/.netrc`. No model runs (exempt): `split --report`, `report`, `cb_report`, `wandb_backfill --dry-run`.
- **Layout** (one place: the layout section of `src/gradproposal/tracking.py`):
  - project `latent-comm` for real runs, `latent-comm-dev` for smoke runs (`run --limit`, or an out file in `latent/results/dev/` or named `smoke*`);
  - group = experiment = local folder `latent/results/<group>/`: `latentmas-qwen3-4b`, `split-probe`, `cacheback-fanoutqa`. A run refuses an `--out` outside its folder;
  - job type `eval`, `timing` (split probe at batch size 1) or `smoke`;
  - run name `<group>.<bench>.<conditions or arm>[.m<m>].s<seed>[.<job type>]`, for example `latentmas-qwen3-4b.gsm8k.latent+wrong+noise+nocache.m40.s0`;
  - tags `git-<sha7>`, model, bench, `paper-p1` (LatentMAS grid and split probe), `smoke`, `backfill`;
  - run id = hash of project and results path, so a requeued job continues its W&B run. The run URL goes into the results file's meta row.
- **Logged:** config = CLI arguments plus run metadata (the file's meta row); per batch: running accuracy, batch seconds, peak GPU memory; at the end: summary (accuracy per condition and m, split-probe paired differences), a per-question table without texts, and the results file as a new version of artifact `results-<group>-<bench>` (type `results`, alias = run name). Solver texts (`*.texts.jsonl`) are not uploaded.
- **Registry:** `latent/results/runs.jsonl`, one entry per results file: date, files, commit, Slurm job, host, GPU, command, status, W&B project, group, job type, name and URL. Rebuild it after fetching results: `python -m gradproposal.latent.wandb_backfill --build-registry`. Runs from before tracking: `python -m gradproposal.latent.wandb_backfill --dry-run`, then without `--dry-run` to log them (tag `backfill`) and record their URLs.
