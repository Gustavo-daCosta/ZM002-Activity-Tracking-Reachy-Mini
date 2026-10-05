# Exercise recorder

Records you doing three physiotherapy exercises so we can train the Reachy Mini robot to recognize them.
The script tells you on screen what to do and for how long. At the end it writes one zip file: send that
file back. Thanks!

## What you need

- Python 3.10 or newer (Windows, macOS or Linux)
- A webcam (the laptop's built-in one is fine)
- About 2.5 m x 2 m of free floor, and a place for the laptop where the camera sees you **from head to feet**
  when you stand 2–3 m away (a chair or a shelf works)
- About 15 minutes

## Install (once)

Open a terminal in this folder.

macOS / Linux (on Ubuntu/Debian, first `sudo apt install python3-venv`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell or Command Prompt):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Record

Always run the script with the `.venv` Python, from this folder (on Windows write `.venv\Scripts\python`
instead of `.venv/bin/python`):

```bash
.venv/bin/python record_exercises.py --list-cameras     # which camera number is your webcam? usually 0
.venv/bin/python record_exercises.py --name "Your Name" --camera 0
```

1. Step back until the banner turns green (head, hips, knees and ankles visible) and stay there: after
   5 seconds fully in view the session starts by itself, no key needed. If you step out of view the
   5 seconds start over.
2. Do what the banner says until the time runs out. A 3-second countdown between rounds shows what comes
   next. One session is 24 rounds of 8 seconds, about 4.5 minutes.
3. When the session ends, change something — clothes, where you stand, lighting, or the room — and press
   **N** for a second session (click on the video window first: keys only work while it is selected),
   then walk back into view and it starts again by itself. Two sessions are ideal. Press **Q** to finish.
4. The zip `exercise_data_<name>_<date>.zip` appears in this folder. **Send it back.** It is roughly
   45 MB per session.

Press **Q** (or ESC) at any time to stop; what was recorded so far is kept. Running the script again with
the same name adds new sessions and the zip contains all of them.

## The exercises

| On screen | What to do |
|---|---|
| SQUAT | Feet shoulder-width apart, sit back as if onto a chair, stand up. Repeat at a calm pace. |
| ARM RAISE | Arms by your sides, raise both arms sideways to shoulder height, lower them. Repeat. |
| SIDE BEND | Hands on your hips, lean to the left, back to the middle, lean to the right. Repeat. |
| NO EXERCISE | Do what the hint says (stand still, check your phone, fix your hair...). Just be natural. |

Do them the way you normally would — small differences between people are exactly what we need. Stay
facing the camera and keep your whole body in view.

## What gets recorded

A video of each session and your body pose (17 points per frame) labeled with the exercise on screen.
Everything stays in the `recordings/` folder on your computer and in the zip you send. Nothing is uploaded.

## Troubleshooting

- **Could not open camera** — close Zoom/Teams/other apps using the camera, then try another number from
  `--list-cameras`.
- **macOS: black image or no camera** — System Settings → Privacy & Security → Camera: allow your terminal app.
- **Banner stays red** — step further back, and make sure the room is well lit.
- **`movenet-lightning.onnx not found`** — unzip the whole folder, and run the script from inside it.
