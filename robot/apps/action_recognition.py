"""Multi-action recognition on the real Reachy Mini: the antennas answer each recognized action with its
own signature.

On the robot (daemon and camera are local, no window). The camera is read through Picamera2, which only
`~/camera_env` (Python 3.13) can import; `~/wave_env` fails with "No module named 'picamera2'":

    ssh reachy 'cd ~/wave_app && ~/camera_env/bin/python -m robot.apps.action_recognition'

From this Mac (pose model here, frames over WebRTC, optional preview window):

    reachy_mini_env/bin/python -m robot.apps.action_recognition --host 192.168.137.171 --window

CMAS items (chair rise, pick-up, arms-up timer), annotated view at http://<robot>:8090/:

    ssh reachy 'cd ~/wave_app && ~/camera_env/bin/python -m robot.apps.action_recognition --cmas --follow --stream-port 8090'

Place the robot so its camera sees the whole person. Waves and clapping survive a waist-up framing, a
squat misses about 1 repetition in 7 with the legs in frame, and push-ups do not work at all. Per-class
F1 and what is honestly demonstrable: docs/action-recognition.md.

Deployment: robot/deploy.sh (see docs/robot.md).
"""

from core.motion.actions import CMAS_MODEL, DEFAULT_ACTION_MODEL, ActionDetector, ActionMonitor
from core.motion.cmas import OverheadHold
from robot.apps.common import make_parser, run


class CmasMonitor(ActionMonitor):
    """`ActionMonitor` plus the arms-up timer of CMAS items 7 and 8 on the overlay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold = OverheadHold()

    def update(self, now, keypoints):
        """Add a frame to the window and to the arms-up timer."""
        elapsed_ms = super().update(now, keypoints)
        self.hold.update(now, keypoints, self.min_score, self.window.aspect_ratio)
        return elapsed_ms

    def panel_lines(self, now):
        """The action rows, then the arms-up row."""
        yield from super().panel_lines(now)
        yield self.hold.panel_line()


def parse_args(argv=None):
    """Parse the command line."""
    parser = make_parser(__doc__)
    parser.add_argument("--classifier", default=None,
                        help="action model .npz (default: core/models/action_classifier.npz, "
                             "or core/models/cmas_classifier.npz with --cmas)")
    parser.add_argument("--cmas", action="store_true",
                        help="CMAS items: the chair rise / pick-up model, plus how long the arms stay straight "
                             "above the head (items 7 and 8)")
    return parser.parse_args(argv)


def main():
    """Entry point."""
    args = parse_args()
    # Fail before touching the robot when the model is missing; parsing the .npz once is deliberate.
    model_path = args.classifier or (CMAS_MODEL if args.cmas else DEFAULT_ACTION_MODEL)
    detector = ActionDetector(model_path)
    if not detector.available:
        raise SystemExit(f"No action model at {model_path}. Deploy it with robot/deploy.sh "
                         "--code-only, or train it with `python -m training.train_actions`.")
    print(f"Action model {model_path}")
    print("Place the robot so the camera sees the WHOLE person: squats and push-ups need the legs in frame.")

    def make_monitor(aspect_ratio, min_score):
        return (CmasMonitor if args.cmas else ActionMonitor)(aspect_ratio, min_score=min_score, detector=detector)

    def events(monitor):
        return {f"{action.upper()} recognized": count for action, count in monitor.counts.items()}

    def summary(monitor, recognized):
        # Recognitions, not antenna answers: an occurrence during another reaction is counted, not answered.
        counts = " ".join(f"{action}:{count}" for action, count in monitor.counts.items() if count)
        print(f"  actions recognized: {counts or '-'}")
        if recognized:
            print("  recognized at: " + ", ".join(f"{t:.1f}s {label.split()[0].lower()}" for t, label in recognized))
        if args.cmas:
            print(f"  longest arms-up hold: {monitor.hold.best_s:.1f} s")

    run(args, make_monitor, events, summary, "Reachy Mini - action recognition")


if __name__ == "__main__":
    main()
