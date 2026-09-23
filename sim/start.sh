#!/usr/bin/env bash
# Start the Reachy Mini daemon in MuJoCo simulation (3D window + API on http://localhost:8000).
# macOS needs mjpython (MuJoCo's viewer must own the main thread). Extra args are passed through,
# e.g. `sim/start.sh --scene minimal` or `--headless`.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=reachy_mini_env
if [[ "$(uname)" == "Darwin" ]]; then
  PY="$VENV/bin/mjpython"
else
  PY="$VENV/bin/python"
fi

exec "$PY" -m reachy_mini.daemon.app.main --sim --no-media "$@"
