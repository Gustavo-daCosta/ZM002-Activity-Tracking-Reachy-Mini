# CMAS demo on the robot

The robot leads a short guided session (in English) and recognizes that four CMAS items were
**performed** (it does not score them). It announces each item, asks for it, and answers when it sees it:

| round | item | what to do in front of the robot |
|---|---|---|
| 1 | 12 chair rise | sit on the chair, then stand up **slowly**, once |
| 2 | 14 pick-up | pick a pencil up from the floor |
| 3 | 7 / 8 arm raise | both arms straight up above the head, hold as long as you can, then lower them |

Starting point: this Mac and the robot are on the same Wi-Fi (`Reachy Mini`), robot powered on (plug the
cable — a dead battery looks like a network failure). All commands run from the repo root.

## 1. Reach the robot

```bash
ping -c 1 192.168.137.53
ssh reachy 'echo ok'
```

No answer → the DHCP address moved. Try `ping -c 1 reachy-mini.local`, then put the new IP in `ROBOT_IP`
(`robot/connect.py`) and in `HostName` of `Host reachy` (`~/.ssh/config`).

## 2. Sync code and models

```bash
robot/deploy.sh --code-only
```

On a robot never set up for the demo, run `robot/deploy.sh` once without flags instead.

## 3. Preflight

```bash
reachy_mini_env/bin/python robot/preflight.py --fix --need media
```

Must exit 0 (motors on, awake, camera acquired). Otherwise read the ✖ line and fix that first.

## 4. Place the robot

The camera must see the **whole person**, the chair and the pencil, with room above the head for raised
arms. Put the robot at the edge of the table if needed. The robot does not turn to follow the person in
this session, so stand where it is already looking.

## 5. Run

```bash
ssh -t reachy 'cd ~/wave_app && ~/camera_env/bin/python -m robot.apps.exercise_session --cmas --stream-port 8090'
```

Open the annotated camera view at <http://192.168.137.53:8090/> (the stream needs ~10 s to come up).

The robot welcomes you, then runs the three rounds from the table:

- **Chair rise** and **pick-up** wait for one repetition, 30 s each; the robot says "Yay!" when it sees it,
  or "Time's up!" and moves on.
- **Arms up** is timed: the robot cheers every 5 s ("5 seconds! Keep going!"), and when the arms come down
  (or after 60 s) it says "Wow, you kept your arms up for N seconds!".

The session ends by itself (about 2 minutes) and puts the robot to sleep. Ctrl+C stops it early, also
putting the robot to sleep. Options: `--rounds pick_up,arms_up` picks a subset, `--reps-per-round 2` asks
for more repetitions, `--timeout 45` gives the chair rise and pick-up more time.

### Free recognition (no guidance)

To show the recognizer on its own, without rounds or speech: the head and body follow the person, the
antennas answer each recognized item, and the overlay times the arms-up hold. Ctrl+C prints the counts.

```bash
ssh -t reachy 'cd ~/wave_app && ~/camera_env/bin/python -m robot.apps.action_recognition --cmas --follow --stream-port 8090'
```

## What to say (and not say)

- It recognizes that an item was done; it gives no CMAS score.
- `chair_rise` means a sit/stand transition happened — sitting down can count too.
- Fast repeated sit/stands are not recognized; one slow rise at a time is.
- The arms-up seconds come from a geometric rule (wrists above the head), not from a trained model.
- Numbers to quote and their limits: `docs/action-recognition.md` (CMAS section).
