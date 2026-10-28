#!/usr/bin/env bash
# Start the Strategy Intelligence dashboard on a machine that has nothing set up yet.
#
# Safe to run repeatedly, and safe after the working directory has been wiped of its
# untracked runtime files: it recreates `.venv` when it is missing, installs the package
# only when it has to, then hands over to `research demo`, which rebuilds the demo memory
# if that is gone too.
#
#   scripts/preview.sh              # dashboard on http://localhost:32909
#   PORT=4000 scripts/preview.sh    # another port
#   DIR=/tmp/demo scripts/preview.sh
#
# The demo bars are synthetic and every screen says so. Nothing here touches a broker,
# and the dashboard can only start a forward paper session.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-32909}"
HOST="${HOST:-0.0.0.0}"
DIR="${DIR:-$HOME/.tradingagents/demo}"
DOMAIN_FILE="tradingagents/research/demo.py"

venv_ready() {
    [ -x .venv/bin/python ] && .venv/bin/python -c "import pandas, typer" >/dev/null 2>&1
}

if ! venv_ready; then
    echo "==> creating .venv (first run, this takes a minute)"
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[dev]"
fi

# An editable install records the project path, so a directory restored by a checkout
# still resolves; a stale entry point is the one failure worth repairing here.
if ! .venv/bin/tradingagents --help >/dev/null 2>&1; then
    echo "==> repairing the install"
    .venv/bin/pip install --quiet -e ".[dev]"
fi

if [ ! -f "$DOMAIN_FILE" ]; then
    echo "error: $DOMAIN_FILE is missing — this checkout is not the branch under test" >&2
    exit 1
fi

echo "==> dashboard on http://localhost:${PORT}  (memory: ${DIR})"
exec .venv/bin/tradingagents research demo --dir "$DIR" --port "$PORT" --host "$HOST"
