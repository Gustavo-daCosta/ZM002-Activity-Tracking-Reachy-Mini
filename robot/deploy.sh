#!/usr/bin/env bash
# Deploy the wave recognizer to the robot: sync the code + models and set up the ~/wave_env venv.
# Idempotent: safe to re-run after code changes (the venv steps are skipped when already installed).
#
#   robot/deploy.sh                 # full setup / update (robot needs internet for pip)
#   robot/deploy.sh --code-only     # only re-sync code and models
#   robot/deploy.sh --fetch-wheels  # download the aarch64 wheels on this Mac (for --offline)
#   robot/deploy.sh --offline       # install from the wheels in robot/wheels/
#
# Rationale, wheel availability and the aarch64 constraints: docs/robot.md
set -euo pipefail

HOST=${HOST:-reachy}
REMOTE=${REMOTE:-/home/pollen/wave_app}
VENV=${VENV:-/home/pollen/wave_env}
APPS_SITE=${APPS_SITE:-/venvs/apps_venv/lib/python3.12/site-packages}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WHEELS="$REPO/robot/wheels"
PY="$REPO/reachy_mini_env/bin/python"

CODE_ONLY=0
OFFLINE=0
for arg in "$@"; do
  case "$arg" in
    --code-only) CODE_ONLY=1 ;;
    --offline) OFFLINE=1 ;;
    --fetch-wheels)
      mkdir -p "$WHEELS"
      # Pre-download for a robot without internet. reachy_mini cannot be fetched this way (its dependency
      # markers differ on Linux) - on the robot it comes from a wheel or from the apps venv (step 5).
      "$PY" -m pip download --only-binary :all: \
        --platform manylinux_2_28_aarch64 --python-version 3.12 --implementation cp \
        -d "$WHEELS" -r "$REPO/requirements/robot.txt"
      du -sh "$WHEELS"
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { echo; echo "=== $* ==="; }

step "1. SSH and robot state"
ssh "$HOST" 'cat ~/VERSION.txt; python3 -V; df -h / | tail -1; free -h | head -2'

step "2. Sync code and models to $REMOTE"
ssh "$HOST" "mkdir -p $REMOTE/core/models"
# COPYFILE_DISABLE: keep macOS from adding AppleDouble "._*" files to the archive.
# The payload is whole directories, not a file list: `core/` is what both environments share and `robot/`
# is what only runs here, so the robot gets exactly those two. `training/` is excluded by construction --
# it needs scikit-learn, which this venv deliberately does not have. An earlier version listed twelve
# individual paths and carried a comment warning that narrowing one could silently drop the action
# recognizer; directories cannot drop a new module.
# Models are synced separately below, so they are excluded here.
COPYFILE_DISABLE=1 tar czf - -C "$REPO" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'core/models' --exclude 'robot/wheels' \
  core robot requirements/robot.txt \
  | ssh "$HOST" "tar xzf - -C $REMOTE"
# Only the models the robot needs (int8 MoveNet 2.9 MB + the numpy forests; the fp32 MoveNet, vitpose-s and
# BlazePose stay on the Mac). Only the .npz exports go: the robot has no scikit-learn and cannot unpickle a
# .joblib.
COPYFILE_DISABLE=1 tar czf - -C "$REPO/core/models" movenet-lightning-int8.tflite wave_classifier.npz wave_classifier_ntu.npz action_classifier.npz \
  | ssh "$HOST" "tar xzf - -C $REMOTE/core/models"
ssh "$HOST" "ls -la $REMOTE $REMOTE/core/models | head -30"

if [ "$CODE_ONLY" = 1 ]; then
  echo; echo "Code synced. Skipping the venv steps (--code-only)."
  exit 0
fi

step "3. Create the venv $VENV (the daemon venvs are never touched)"
# Python 3.12, not the system python3 (3.13 on ReachyMiniOS v0.2.3): 3.12 is what the daemon/apps venvs use, so the
# .pth fallback of step 5 stays ABI-compatible and the pre-downloaded cp312 wheels fit.
if [ -z "${PYBIN:-}" ]; then
  PYBIN=$(ssh "$HOST" 'ls -d ~/.local/share/uv/python/cpython-3.12.*/bin/python3.12 2>/dev/null | head -1')
  [ -n "$PYBIN" ] || PYBIN=$(ssh "$HOST" 'command -v python3.12 || command -v python3')
fi
echo "interpreter: $PYBIN"
ssh "$HOST" "test -x $VENV/bin/python || $PYBIN -m venv $VENV; $VENV/bin/pip install -q -U pip; $VENV/bin/python -V"

step "4. Install onnxruntime / opencv"
if [ "$OFFLINE" = 1 ]; then
  test -d "$WHEELS" || { echo "No $WHEELS - run --fetch-wheels first" >&2; exit 1; }
  COPYFILE_DISABLE=1 tar czf - -C "$WHEELS" . | ssh "$HOST" 'mkdir -p ~/wave_wheels && tar xzf - -C ~/wave_wheels'
  ssh "$HOST" "$VENV/bin/pip install --no-index --find-links ~/wave_wheels -r $REMOTE/requirements/robot.txt"
else
  ssh "$HOST" "$VENV/bin/pip install -r $REMOTE/requirements/robot.txt"
fi

step "5. Make the reachy_mini SDK importable"
# Preferred: its own wheel. Fallback: reuse the copy already working in the apps venv through a .pth file
# (our venv comes first in sys.path, so our numpy/opencv win).
if ssh "$HOST" "$VENV/bin/pip install 'reachy_mini==1.10.0'"; then
  echo "reachy_mini installed in $VENV"
else
  echo "Wheel install failed - falling back to the apps venv via a .pth file"
  ssh "$HOST" "echo $APPS_SITE > $VENV/lib/python3.12/site-packages/reachy_mini_apps_venv.pth"
fi

step "6. Smoke test (imports only, no motion)"
ssh "$HOST" "cd $REMOTE && $VENV/bin/python -c '
import cv2, numpy, onnxruntime
import reachy_mini
from core.pose_backends import create_backend
from core.motion.detectors import ClassifierWaveDetector
from core.motion.actions.detector import ActionDetector
print(\"cv2\", cv2.__version__, \"onnxruntime\", onnxruntime.__version__, \"numpy\", numpy.__version__)
print(\"reachy_mini\", reachy_mini.__version__ if hasattr(reachy_mini, \"__version__\") else \"?\")
print(\"classifier available:\", ClassifierWaveDetector().available)
print(\"action classifier available:\", ActionDetector().available)
'"

cat <<MSG

Done. Next (on the robot, moves the antennas):
  ssh $HOST 'cd $REMOTE && $VENV/bin/python -m robot.apps.wave_antennas --seconds 60'
  ssh $HOST 'cd $REMOTE && $VENV/bin/python -m robot.apps.action_recognition --seconds 60'
MSG
