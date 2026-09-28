"""Wave recognition on the real Reachy Mini: the robot camera feeds the pose model, the antennas wave back.

On the robot (daemon and camera are local, no window):

    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.wave_antennas'

From this Mac (pose model here, frames over WebRTC, optional preview window):

    reachy_mini_env/bin/python -m robot.apps.wave_antennas --host 192.168.137.171 --window

Deployment: robot/deploy.sh (see docs/robot.md).
"""

from core.motion.reaction import antennas_with_neutral
from robot.apps.common import make_parser, run


def parse_args(argv=None):
    """Parse the command line."""
    parser = make_parser(__doc__)
    parser.add_argument("--wave-trigger", choices=["auto", "rules", "classifier"], default="auto",
                        help="detector the robot reacts to (auto: classifier if trained, else rules)")
    parser.add_argument("--classifier", default=None,
                        help="wave model .npz (default: core/models/wave_classifier_ntu.npz)")
    parser.add_argument("--amplitude", type=float, default=20.0, help="antenna wave amplitude in degrees")
    parser.add_argument("--freq", type=float, default=1.5, help="antenna wave frequency in Hz")
    parser.add_argument("--wave-seconds", type=float, default=1.2, help="how long the antennas wave back")
    parser.add_argument("--cooldown", type=float, default=0.5,
                        help="dead time after a wave before another one is answered")
    return parser.parse_args(argv)


def main():
    """Entry point."""
    args = parse_args()
    from core.motion.live import WaveMonitor

    # Built before touching the robot, so a missing classifier fails here.
    kwargs = {"model_path": args.classifier} if args.classifier else {}
    try:
        monitor = WaveMonitor(trigger=args.wave_trigger, **kwargs)
    except ValueError as exc:
        raise SystemExit(str(exc))
    monitor.reaction.amplitude_deg = args.amplitude
    monitor.reaction.freq_hz = args.freq
    monitor.reaction.duration_s = args.wave_seconds
    monitor.reaction.cooldown_s = args.cooldown
    monitor.antennas = lambda now: antennas_with_neutral(monitor.reaction.angles(now))
    print(f"Trigger {monitor.trigger_name}, antennas {args.amplitude:.0f} deg @ {args.freq:.1f} Hz, "
          f"wave {args.wave_seconds:.1f} s + {args.cooldown:.1f} s cooldown")

    def make_monitor(aspect_ratio, min_score):
        monitor.min_score = monitor.window.min_score = min_score
        monitor.window.aspect_ratio = aspect_ratio
        return monitor

    def summary(monitor, recognized):
        print(f"  wave detections: rules {monitor.triggers['rules']}, "
              f"classifier {monitor.triggers['classifier']}; antennas answered {monitor.reactions} "
              f"(trigger: {monitor.trigger_name}, the rest fell inside the cooldown)")
        if recognized:
            print("  reacted at: " + ", ".join(f"{t:.1f}s" for t, _ in recognized))

    run(args, make_monitor, lambda monitor: {"WAVE -> antennas": monitor.reactions}, summary,
        "Reachy Mini - wave recognition")


if __name__ == "__main__":
    main()
