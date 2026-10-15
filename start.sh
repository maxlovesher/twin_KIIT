#!/usr/bin/env bash
# GlucoTwin launcher for macOS / Linux:  bash start.sh
set -euo pipefail
cd "$(dirname "$0")"

echo
echo "  GlucoTwin - hybrid digital twin for Type 2 Diabetes"
echo "  ---------------------------------------------------"
echo

PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "  [!] Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/"
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "  [1/3] Creating a private Python environment in .venv ..."
    "$PY" -m venv .venv
fi
VPY=.venv/bin/python

if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
    echo "  [2/3] Installing packages - first run takes a few minutes ..."
    "$VPY" -m pip install --disable-pip-version-check -q --upgrade pip
    "$VPY" -m pip install --disable-pip-version-check -r requirements.txt
    cp requirements.txt .venv/installed-requirements.txt
else
    echo "  [2/3] Packages already installed."
fi

echo "  [3/3] Starting the dashboard at http://localhost:8501  (Ctrl+C to stop)"
echo "        The first page load takes ~30 s while the patient twins compile."
echo
exec "$VPY" -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
