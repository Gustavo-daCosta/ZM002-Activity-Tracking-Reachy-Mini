# Wave recognition demo

    ./demo.sh

Then open the URL it prints (`http://<robot>:8080/`) and wave at the robot. The antennas wave back and the
head and body follow you. Ctrl+C stops it and puts the robot to sleep.

    ./demo.sh --no-follow    # antennas only — use this if the room is full of people moving

## What it does

Waves are recognized on the robot itself: MoveNet int8 (TFLite/XNNPACK, 34 ms per frame) turns the camera
frame into a COCO-17 skeleton, and a random forest exported to plain numpy scores a 1.5 s window of wrist
motion in ~3 ms. The forest was trained on NTU RGB+D 60 (944 waving clips, 40 subjects), and scores F1
0.951 on our own recordings, which it never saw. The loop runs at ~9.7 FPS, limited by the camera, not the
model.

## When it breaks

**"Robot is not answering"** — the IP is DHCP and moves. Try `REACHY_HOST=reachy-mini.local
SSH_HOST=reachy-mini.local ./demo.sh` (set both — `REACHY_HOST` only fixes the ping and the browser URL,
`ssh` uses `SSH_HOST`), or read the current IP in the Reachy Mini Control app and use
`REACHY_HOST=<ip> SSH_HOST=<ip>`. A dead battery looks exactly like a network failure: plug the power cable
before debugging anything else.

**Preflight says the robot is asleep or the motors are off** — expected after every boot; the daemon starts
with `--no-wake-up-on-start` and `demo.sh` already passes `--fix`, which wakes it. If it still fails, read
the ✖ line: it names the one thing that is wrong.

**"No camera frames"** — something else holds the camera. Check
`curl -s http://<robot>:8000/api/media/status`; `released=true` means another client took it. Closing that
client and re-running is enough.

**The stream page is blank** — the recognizer is up but has not produced a frame yet; give it a few
seconds. If it stays blank, the port is wrong or blocked: try `./demo.sh --stream-port 8090`.

**The camera picture is blurry** — hardware, not software. The dw9807 focus motor on this unit fails over
I2C. Do not try to fix it before the meeting.

**"4/4" fails with "no such file or directory" (venv/app not found)** — this is a robot that has never been
set up for the demo. `demo.sh` only syncs code (`deploy_wave_robot.sh --code-only`), it does not create the
venv. Run `robot/deploy.sh` once (no flags) to set it up, then `./demo.sh` works as usual.
