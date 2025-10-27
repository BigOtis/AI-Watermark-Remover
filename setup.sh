#!/usr/bin/env bash

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 not found. Install from https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi

echo "Activating virtual environment..."
set +u
source .venv/bin/activate
set -u

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
python -m pip install -r requirements.txt

echo "Downloading LaMa model..."
python -m iopaint download --model lama || iopaint download --model lama || true

if [ "$#" -gt 0 ]; then
  python remwm.py "$@"
else
  python remwmgui.py
fi

