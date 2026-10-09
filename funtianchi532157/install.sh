#!/usr/bin/env bash
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

uv sync
uv pip install --python .venv/bin/python -e "$project_dir/lm-evaluation-harness"
