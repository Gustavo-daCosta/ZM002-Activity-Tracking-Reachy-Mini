"""Multi-action recognition on the real Reachy Mini: the antennas answer each recognized action with its
own signature.

On the robot (daemon and camera are local, no window):

    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.action_recognition'

From this Mac (pose model here, frames over WebRTC, optional preview window):

    reachy_mini_env/bin/python -m robot.apps.action_recognition --host 192.168.137.171 --window

Place the robot so its camera sees the whole person. Waves and clapping survive a waist-up framing, a
squat misses about 1 repetition in 7 with the legs in frame, and push-ups do not work at all. Per-class
F1 and what is honestly demonstrable: docs/action-recognition.md.

Deployment: robot/deploy.sh (see docs/robot.md).
"""

from robot.apps.common import make_parser, run


def parse_args(argv=None):
    """Parse the command line."""
    parser = make_parser(__doc__)
    parser.add_argument("--classifier", default=None,
                        help="action model .npz (default: core/models/action_classifier.npz)")
    return parser.parse_args(argv)


def main():
    """Entry point."""
    args = parse_args()
    from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector
    from core.motion.actions.live import ActionMonitor

    # Fail before touching the robot when the model is missing; parsing the .npz once is deliberate.
    model_path = args.classifier or DEFAULT_ACTION_MODEL
    detector = ActionDetector(model_path)
    if not detector.available:
        raise SystemExit(f"No action model at {model_path}. Deploy it with robot/deploy.sh "
                         "--code-only, or train it with `python -m training.actions.train`.")
    print(f"Action model {model_path}")
    print("Place the robot so the camera sees the WHOLE person: squats and push-ups need the legs in frame.")

    def make_monitor(aspect_ratio, min_score):
        return ActionMonitor(aspect_ratio, min_score=min_score, detector=detector)

    def events(monitor):
        return {f"{action.upper()} recognized": count for action, count in monitor.counts.items()}

    def summary(monitor, recognized):
        # Recognitions, not antenna answers: an occurrence during another reaction is counted, not answered.
        counts = " ".join(f"{action}:{count}" for action, count in monitor.counts.items() if count)
        print(f"  actions recognized: {counts or '-'}")
        if recognized:
            print("  recognized at: " + ", ".join(f"{t:.1f}s {label.split()[0].lower()}" for t, label in recognized))

    run(args, make_monitor, events, summary, "Reachy Mini - action recognition")


if __name__ == "__main__":
    main()
