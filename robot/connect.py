"""Shared configuration and helpers for Reachy Mini scripts.

The robot and this computer must be on the same Wi-Fi network ("Reachy Mini"). ROBOT_IP may change
(DHCP): if the robot stops answering, try reachy-mini.local or set REACHY_HOST=<new ip>.
"""

import os
from contextlib import contextmanager

ROBOT_IP = "192.168.137.171"
MDNS_HOST = "reachy-mini.local"
API_PORT = 8000


def resolve_host() -> str:
    """The robot host: REACHY_HOST if set, else ROBOT_IP."""
    return os.environ.get("REACHY_HOST") or ROBOT_IP


class RobotNotReady(RuntimeError):
    """Preflight found a problem it could not fix automatically."""


@contextmanager
def connect_robot(media: bool = True, sleep_on_exit: bool = False, needs=("motion",)):
    """Preflight (with auto-fix) + ReachyMini connection, as a context manager.

    Args:
        media: False skips WebRTC video/audio on this computer but keeps the camera on the daemon.
        sleep_on_exit: Put the robot to sleep when the block ends.
        needs: Preflight requirements ("motion", "media", "tracking").

    Yields:
        The connected `ReachyMini`.

    Raises:
        RobotNotReady: Preflight failed.
    """
    from robot import preflight as reachy_preflight  # lazy: preflight imports this module

    needs = set(needs)
    if media:
        needs.add("media")
    result = reachy_preflight.run_preflight(host=resolve_host(), fix=True, needs=tuple(needs))
    print(result.render())
    if not result.ready:
        raise RobotNotReady(result.render())

    from reachy_mini import ReachyMini

    backend = "default" if media else "no_media"
    with ReachyMini(host=result.host, connection_mode="network", media_backend=backend) as mini:
        if not media:
            # no_media makes the SDK release the camera; give it back to the daemon.
            mini.client.acquire_media()
            mini._media_released = False
        try:
            yield mini
        finally:
            try:
                mini.stop_head_tracking()
            except Exception:
                pass
            if sleep_on_exit:
                mini.goto_sleep()
