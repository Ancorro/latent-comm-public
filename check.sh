#!/usr/bin/env bash
# Completion check, CPU only, about 30 s: the full test suite (tiny random Qwen3 models, stub agents) and lint.
#   bash setup.sh && bash check.sh
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m pytest -q tests
.venv/bin/python -m ruff check .
