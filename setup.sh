#!/usr/bin/env sh
set -eu
cd -- "$(dirname -- "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "${1:-}" = "--neural" ]; then
  .venv/bin/python -m pip install -e '.[neural]'
else
  .venv/bin/python -m pip install -e .
fi
.venv/bin/python -m oibc_portable doctor
