#!/usr/bin/env bash
# CPU-only setup for everything that runs without a GPU: the LatentMAS, split-probe and CacheBack mechanics on tiny
# random models, the GEM episode runner against stub agents, the viewer, tracking and table generation.
#   bash setup.sh            -> .venv/ (Python 3.11+), then: bash check.sh
# Versions match uv.lock; `uv sync` gives the same environment if uv is installed. Needs network access for pip.
# Real-model runs (Qwen3-4B on a GPU) are not covered here; see README.md.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 11), f"Python 3.11+ needed, found {sys.version.split()[0]}"'

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
# CPU wheel of the pinned torch (the default index pulls several GB of CUDA libraries).
.venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
# gem-llm 0.0.3 declares python<=3.12, which pip reads as 3.12.0 only; it runs fine on any 3.12.x.
.venv/bin/python -m pip install -q --ignore-requires-python "gem-llm==0.0.3"
.venv/bin/python -m pip install -q -e . --group dev --group latent
.venv/bin/python -c 'import torch, transformers, gem, wandb; print(f"ok: torch {torch.__version__}, transformers {transformers.__version__}")'
