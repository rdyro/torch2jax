#!/usr/bin/env bash

set -e

PYTHON_VER=$1
JAX_VERSION=$2
JAX_OPT=$3

pyenv global $PYTHON_VER

pip install -U uv > /dev/null 2> /dev/null
uv pip install --system --cache-dir $HOME/.cache/uv torch pytest ruff \
  "jax[$JAX_OPT]==$JAX_VERSION" 2> /dev/null
uv pip install --system "numpy==1.26.4" 2> /dev/null # downgrade numpy to ver 1
uv pip install --system .  2> /dev/null

echo "JAX_VERSION=$JAX_VERSION" "JAX_OPT=$JAX_OPT" "########################"
python3 -m pytest -p no:warnings tests