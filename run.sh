#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  /usr/bin/python -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/streamlit run projectai.py "$@"
