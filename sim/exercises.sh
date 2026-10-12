#!/usr/bin/env bash
# One command: the MuJoCo simulation plus the webcam window recognizing squat / arm raise / side bend,
# with the antennas answering each exercise. Extra args go to sim.body_tracking (e.g. --camera FaceTime).
# Reuses a daemon already running on :8000; otherwise starts one and stops it on exit.
set -euo pipefail
cd "$(dirname "$0")/.."

started=""
if ! curl -sf http://localhost:8000/api/daemon/status >/dev/null; then
  sim/start.sh >/tmp/reachy-sim.log 2>&1 &
  started=$!
  trap 'kill $started 2>/dev/null' EXIT
  echo "Starting the simulation (log: /tmp/reachy-sim.log)..."
  for _ in $(seq 60); do
    curl -sf http://localhost:8000/api/daemon/status >/dev/null && break
    kill -0 "$started" 2>/dev/null || { echo "Simulation failed to start:"; tail -20 /tmp/reachy-sim.log; exit 1; }
    sleep 1
  done
fi

# MoveNet: the pose model the recordings were made with, so live features match the training ones.
reachy_mini_env/bin/python -m sim.body_tracking --model movenet-lightning \
  --detect-actions --action-model core/models/exercise_classifier.npz "$@"
