#!/usr/bin/env bash
# Run once, before any detached preparation run.
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-/root/userdata/MappingNetworks}"; VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-finewebedu}"
mkdir -p "$ROOT_DIR/.cache/finewebedu/pip" "$ROOT_DIR/.tmp/finewebedu"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/finewebedu/pip" TMPDIR="$ROOT_DIR/.tmp/finewebedu"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade "pip==24.3.1"
"$VENV_DIR/bin/python" -m pip install "datasets==3.2.0" "tiktoken==0.8.0" "numpy==2.1.3"
