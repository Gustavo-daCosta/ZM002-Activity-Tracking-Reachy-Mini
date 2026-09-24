#!/usr/bin/env bash
# Wave recognition demo on the real Reachy Mini: one command, from this Mac.
#
#   ./demo.sh                  # sync, preflight, run; Ctrl+C stops and parks the robot
#   ./demo.sh --no-follow      # antennas only, no head/body tracking (busy room)
#   ./demo.sh --stream-port N  # serve the annotated camera view on another port
#
# Env vars: REACHY_HOST (robot IP/hostname, for the ping and the browser URL) and SSH_HOST (the ssh alias
# or host used to reach it; defaults to the "reachy" alias). If DHCP moved the robot, set both.
#
# What to expect and what to do when it breaks: docs/demo.md
set -euo pipefail

SSH_HOST=${SSH_HOST:-reachy}
REMOTE=${REMOTE:-/home/pollen/wave_app}
VENV=${VENV:-/home/pollen/wave_env}
REPO="$(cd "$(dirname "$0")" && pwd)"
PY="$REPO/reachy_mini_env/bin/python"

FOLLOW="--follow"
STREAM_PORT=8080
while [ $# -gt 0 ]; do
  case "$1" in
    --no-follow) FOLLOW=""; shift ;;
    --stream-port)
      case "${2:-}" in
        ''|*[!0-9]*) echo "--stream-port needs a numeric port" >&2; exit 2 ;;
      esac
      STREAM_PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# The IP is for the browser URL; ssh keeps using the alias, which may resolve differently.
HOST="$("$PY" -c 'import robot.connect; print(robot.connect.resolve_host())')"

echo "=== 1/4  Robot at $HOST (ssh alias: $SSH_HOST)"
if ! ping -c 1 -t 2 "$HOST" >/dev/null 2>&1; then
  cat >&2 <<MSG
Robot is not answering at $HOST.
  - same Wi-Fi ("Reachy Mini")?  - powered on?  - IP changed (DHCP)?
  - try: REACHY_HOST=reachy-mini.local SSH_HOST=reachy-mini.local ./demo.sh
  - a dead battery looks exactly like this: plug the power cable.
MSG
  exit 2
fi

echo "=== 2/4  Syncing code and models (idempotent, seconds when nothing changed)"
HOST="$SSH_HOST" "$REPO/robot/deploy.sh" --code-only

echo "=== 3/4  Preflight (enables motors, wakes up, re-acquires the camera)"
"$PY" "$REPO/robot/preflight.py" --fix --need media

echo "=== 4/4  Starting the recognizer"
echo
echo "    Camera view with the model overlay:  http://$HOST:$STREAM_PORT/"
echo "    Wave at the robot. Ctrl+C here stops it and puts it to sleep."
echo

# -tt forces a remote TTY even when stdin here is not one, so Ctrl+C reaches the remote process as SIGINT;
# a dropped connection instead delivers SIGHUP, which wave_antennas.py also handles like Ctrl+C.
ssh -tt "$SSH_HOST" "cd $REMOTE && exec $VENV/bin/python -m robot.apps.wave_antennas \
  --model movenet-tflite --width 640 --rate 50 \
  --amplitude 20 --freq 1.5 --cooldown 0.5 \
  --stream-port $STREAM_PORT $FOLLOW \
  --body-gain 1.5 --max-body-yaw 110"
