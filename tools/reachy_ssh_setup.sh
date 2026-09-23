#!/usr/bin/env bash
# Idempotent SSH setup for the Reachy Mini: dedicated key + "reachy" alias.
# Usage: tools/reachy_ssh_setup.sh   (REACHY_HOST overrides the IP)
set -euo pipefail

HOST="${REACHY_HOST:-192.168.137.171}"
KEY="$HOME/.ssh/reachy_mini"
CONFIG="$HOME/.ssh/config"

mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -C "reachy_mini" -f "$KEY"

touch "$CONFIG" && chmod 600 "$CONFIG"
if ! grep -qE '^Host reachy$' "$CONFIG"; then
  printf '\nHost reachy\n  HostName %s\n  User pollen\n  IdentityFile %s\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n  ConnectTimeout 5\n' \
    "$HOST" "$KEY" >> "$CONFIG"
  echo "Added 'Host reachy' ($HOST) to $CONFIG"
fi

if ssh -o BatchMode=yes reachy true 2>/dev/null; then
  echo "OK: 'ssh reachy' works without a password"
  exit 0
fi

echo "Key not installed on the robot yet. Run once (password: root):"
echo "  ssh-copy-id -i $KEY.pub pollen@$HOST"
echo "If the robot IP changed, edit HostName under 'Host reachy' in $CONFIG."
exit 1
