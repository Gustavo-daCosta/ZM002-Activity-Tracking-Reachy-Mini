---
name: reachy-ssh
description: SSH access and robot OS knowledge for the Reachy Mini wireless (Raspberry Pi CM4, ReachyMiniOS). Use when connecting to the robot shell, reading daemon logs, checking systemd services, venvs and installed apps, camera (libcamera/rpicam, focus VCM), audio (ALSA), network/Wi-Fi, disk, clock, or troubleshooting an unreachable robot, backend errors, IK warnings, missing camera or a tracker without frames.
---

# Reachy Mini — SSH & robot OS

## Access

```bash
ssh reachy '<command>'          # key auth, alias from ~/.ssh/config (user pollen, IdentityFile ~/.ssh/reachy_mini)
tools/reachy_ssh_setup.sh       # idempotent setup/verification of key + alias
```

- Fallback: `ssh pollen@192.168.137.171` / `pollen@reachy-mini.local`, password `root`.
  macOS has no `sshpass`: if a password is unavoidable, use a temporary `SSH_ASKPASS` script in the scratchpad
  with `SSH_ASKPASS_REQUIRE=force`, and delete it right after.
- **IP may change** (DHCP): update `HostName` under `Host reachy` in `~/.ssh/config` and `ROBOT_IP` in
  `robot/connect.py`. The robot and this Mac must be on the same Wi-Fi (`Reachy Mini`; this Mac was `192.168.137.217`).
- `pollen` has **passwordless sudo** — be careful, every sudo action needs user confirmation (see below).
- Always non-interactive, bounded commands (`-n 50`, `--no-pager`, `head`). Never `journalctl -f` without a limit;
  bound loops in Python on the Mac (macOS has no `timeout`).

## System map (verified 2026-09)

| Item | Value |
|---|---|
| OS | ReachyMiniOS v0.2.3 (`~/VERSION.txt`), Debian 13 trixie, kernel 6.12 rpi-v8 aarch64 |
| Hardware | Raspberry Pi CM4, 3.7 GiB RAM, 14 G eMMC root (≈70 % used) |
| Daemon service | `reachy-mini-daemon.service` (User=pollen, Restart=on-failure) |
| Daemon launcher | `/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/daemon/app/services/wireless/launcher.sh` → `python -u -m reachy_mini.daemon.app.main --wireless-version --no-wake-up-on-start` |
| Consequence | **The robot always boots asleep with motors disabled** (no wake up on start) |
| Daemon venv | `/venvs/mini_daemon` (reachy_mini 1.10.0, motor controller 1.5.6, rust kinematics 1.0.3, onnxruntime, opencv) |
| Apps venv | `/venvs/apps_venv` (reachy_mini 1.10.0, `reachy_mini_conversation_app` 1.0.1, `reachy_mini_dances_library` 0.2.1) |
| Restore copy | `/restore/venvs/mini_daemon` (factory reset source) |
| Other services | `reachy-mini-bluetooth` (GATT, Wi-Fi provisioning), `gpio-shutdown-daemon`, `NetworkManager`, `avahi-daemon` (mDNS), `ssh` |
| Camera IPC socket | `/tmp/reachymini_camera_socket` (daemon media server → face tracker / local SDK) |
| Uploaded sounds | `/tmp/reachy_mini_sounds/` |
| Motors bus | `/dev/ttyAMA3` @ 1 Mbps (XL330/XC330, IDs 10–18) — owned by the daemon |
| Audio | USB `Pollen Robotics Reachy Mini Audio` (38fb:1001) = ALSA card 0; `~/.asoundrc` defines `reachymini_audio_sink` (dmix) and `reachymini_audio_src` (dsnoop), 16 kHz stereo |
| Camera | imx708_wide via libcamera (`/dev/video0`, `/dev/v4l-subdev0-1`); focus VCM `dw9807` on I2C `10-000c` |
| GStreamer | extra plugins `/opt/gst-plugins-rs` (webrtcsrc/webrtcsink), uv in `/opt/uv` |
| Clock | **not NTP-synchronized** on this hotspot (was ~3 days behind) → log timestamps are unreliable |
| Home | `~/foco_0..3.jpg` (our old focus test photos, safe to delete with user OK) |

## Safe read-only commands (no confirmation needed)

```bash
# health / versions
ssh reachy 'cat ~/VERSION.txt; uptime; df -h /; free -h'
ssh reachy 'systemctl is-active reachy-mini-daemon reachy-mini-bluetooth'
ssh reachy 'systemctl status reachy-mini-daemon --no-pager | head -15'
ssh reachy '/venvs/mini_daemon/bin/pip list 2>/dev/null | grep -i reachy'
ssh reachy '/venvs/apps_venv/bin/pip list 2>/dev/null | grep -i reachy'

# daemon logs (bounded)
ssh reachy 'journalctl -u reachy-mini-daemon -n 100 --no-pager'
ssh reachy 'journalctl -u reachy-mini-daemon -b --no-pager | grep -iE "error|warning|traceback" | tail -40'
ssh reachy 'journalctl -u reachy-mini-daemon --since "10 min ago" --no-pager | grep -v uvicorn.access | tail -60'

# kernel / hardware
ssh reachy 'dmesg | tail -40'
ssh reachy 'dmesg | grep -iE "dw9807|imx708|usb|ttyAMA" | tail -20'
ssh reachy 'lsusb; aplay -l; arecord -l'
ssh reachy 'ls /dev/video* /dev/v4l-subdev*'
ssh reachy 'v4l2-ctl -d /dev/v4l-subdev1 --list-ctrls'        # focus_absolute 0..1023

# network
ssh reachy 'ip -brief addr; nmcli -t -f NAME,TYPE,DEVICE connection show --active'
ssh reachy 'nmcli -t -f SSID,SIGNAL,ACTIVE device wifi list | head'   # cached scan list, no rescan

# processes / resources
ssh reachy 'ps aux --sort=-%cpu | head -8'
ssh reachy 'ls /tmp/reachy_mini_sounds; ls -la /tmp/reachymini_camera_socket'
```

Daemon logs can also be streamed from the Mac without SSH: `ws://<host>:8000/logs/ws/daemon` (see `reachy-api`).

## Actions that REQUIRE user confirmation

| Action | Command |
|---|---|
| Restart daemon (robot goes limp/asleep, clients disconnect, ~30 s) | `ssh reachy 'sudo systemctl restart reachy-mini-daemon'` (or REST `POST /api/daemon/restart`) |
| Stop / start daemon | `sudo systemctl stop|start reachy-mini-daemon` |
| Reboot / shutdown | `ssh reachy 'sudo reboot'` / `sudo poweroff` (robot unreachable for ~1 min / until power button) |
| Official health check | `reachyminios_check` — **not read-only**: sets speaker & mic volume to 100 %, plays a test tone, records, `alsactl store`, grabs the camera and pings motors on the serial bus (conflicts with a running daemon) |
| Direct camera tests | `rpicam-still` / `rpicam-hello` — first `POST /api/media/release`, afterwards `POST /api/media/acquire` |
| Direct audio tests | `aplay`/`arecord`/`gst-launch-1.0 alsasrc device=reachymini_audio_src ...` (competes with daemon media) |
| Install / update Python packages | `/venvs/apps_venv/bin/pip install ...`, `/venvs/mini_daemon/bin/pip ...` |
| Install an app offline | `scp -r my_app reachy:/tmp/my_app && ssh reachy '/venvs/apps_venv/bin/pip install /tmp/my_app'` |
| Edit any file on the robot | e.g. daemon sources, `~/.asoundrc`, systemd units |
| Wi-Fi changes | `nmcli ... connect/delete` (can make the robot unreachable) |
| Delete files | e.g. `rm ~/foco_*.jpg` |
| Motor tools | `python -m reachy_mini.tools.scan_motors --wireless`, `reachy-mini-reflash-motors` (stop the daemon first) |

After any restart/reboot: wait, then run `robot/preflight.py --fix` — the robot will be asleep.

## Troubleshooting playbooks

### Robot unreachable (preflight exit 2)
1. Same Wi-Fi on the Mac? (`Reachy Mini`) Conference/hotel networks isolate clients → use the hotspot.
2. Robot powered and booted (~1 min)? LED state. **Battery dead** is common: the robot silently drops off Wi-Fi in
   the middle of a task (and on a `192.168.137.x` hotspot even the gateway ping can fail from the Mac). Ask the user
   to plug the power cable; after reboot it comes back asleep with motors disabled.
3. `ping -c 2 192.168.137.171`; `ping -c 2 reachy-mini.local` — if mDNS answers with another IP, update `ROBOT_IP`
   and `~/.ssh/config`.
4. Scan the subnet for the daemon: `for i in $(seq 1 254); do curl -s -m 0.3 http://192.168.137.$i:8000/api/daemon/status >/dev/null && echo 192.168.137.$i; done`
5. API down but SSH works → `systemctl is-active reachy-mini-daemon` + logs; restart needs confirmation.

### Backend not ready / error in daemon status
`journalctl -u reachy-mini-daemon -n 200 --no-pager | grep -iE "error|traceback|motor" -A3` → motor bus or
power issues (7 V 5 A supply / battery). Motor scan tools need the daemon stopped (confirmation).

### Commands accepted but robot does not move
Motors disabled or asleep → `robot/preflight.py --fix`. Check running app (`/api/apps/current-app-status`).

### `IK error: WARNING: Collision detected or head pose not achievable!` in logs
The requested head pose is outside the workspace (limits: roll/pitch ±40°, head yaw vs body ±60°, z −4..+2.5 cm).
Reduce amplitude or combine with `body_yaw`.

### Camera "hidden" / tracker `ts: null`
`curl -s $H/api/media/status` → `released: true` → `POST /api/media/acquire` (preflight `--need media --fix`).
Still no frames → `ls -la /tmp/reachymini_camera_socket`, logs `grep -i "camera\|media\|tracker"`.
"Face tracker cannot reach the camera feed" = media released by a client (`no_media`).

### Blurry camera
Known hardware fault on this unit: `dmesg | grep dw9807` shows `I2C write CTL fail ret = -5` since boot → the
autofocus actuator does not respond. Software focus (`lens-position`, `af-mode`) has no effect. Fix = reseat the
lens-module connector / ribbon cable, or replace the Camera Module 3 **Wide**. Check again after hardware work: no
`dw9807 ... fail` lines in `dmesg`.

### No sound / mic silence
`aplay -l` shows `Reachy Mini Audio`? Volume `GET /api/volume/current`. Mic silent → flat cable upside down (docs).
Low volume → update ReachyMiniOS / audio firmware (confirmation).

### App crashes silently
Missing dependency in `/venvs/apps_venv`:
`ssh reachy "/venvs/apps_venv/bin/python3 -c 'from my_app.main import MyApp'"`; logs are in the daemon journal.
Conversation app broken after update → "Reset apps environment" (`POST /cache/reset-apps`, confirmation).

## Docs

- Wireless get started / dev workflow: https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started ,
  https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/development_workflow
- Troubleshooting FAQ: https://huggingface.co/docs/reachy_mini/troubleshooting
- Reset / reflash: https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset
