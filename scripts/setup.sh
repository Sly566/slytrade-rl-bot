#!/usr/bin/env bash
# Setup SlyTrade dev environment on Linux (Pop!_OS / Ubuntu).
# Creates .venv with python3.12 and installs the project in editable mode.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Using Python: $(python3.12 --version 2>/dev/null || echo 'python3.12 NOT FOUND — install with: sudo apt install python3.12 python3.12-venv')"

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv in .venv/"
  python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "==> Installing slytrade (editable) + dev dependencies"
pip install -e ".[dev]"

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .venv/bin/activate"
echo "Then run:"
echo "  slytrade doctor"
