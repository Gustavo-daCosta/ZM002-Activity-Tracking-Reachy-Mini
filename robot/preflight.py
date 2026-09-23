"""Reachy Mini preflight: verify (and optionally fix) robot state before acting.

Uses only the daemon REST API and the Python standard library.
Exit codes: 0 ready, 1 not ready, 2 unreachable.

Usage:
    python robot/preflight.py [--host H] [--fix] [--json] [--need motion|media|tracking ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from robot.connect import API_PORT, MDNS_HOST, resolve_host  # noqa: E402

VALID_NEEDS = ("motion", "media", "tracking")

# Sleep pose observed on the robot: z ≈ -0.046 m, pitch ≈ 0.48 rad, antennas ≈ 2.6-3.0 rad
SLEEP_Z_M = -0.02
SLEEP_PITCH_RAD = 0.3
SLEEP_ANTENNA_RAD = 1.5

ICONS = {"ok": "✔", "fixed": "⚙", "warn": "!", "failed": "✖"}

NETWORK_HINT = (
    "Robot not reachable at {host} nor {mdns}. Check: robot powered on and booted (~1 min); "
    "computer and robot on the same Wi-Fi (e.g. 'Reachy Mini'); `ping {host}`; "
    "the IP may have changed (try `ping {mdns}` or the Reachy Mini Control app, then update "
    "ROBOT_IP in robot/connect.py or set REACHY_HOST)."
)


class ApiError(RuntimeError):
    """Network or HTTP failure talking to the daemon."""


class Api:
    """Minimal JSON client for the Reachy Mini daemon REST API."""

    def __init__(self, host: str, port: int = API_PORT, timeout: float = 3.0):
        self.host = host
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(self, method: str, path: str, body=None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise ApiError(f"{method} {path}: {exc}") from exc
        return json.loads(raw) if raw else None

    def get(self, path: str, timeout: float | None = None):
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, body=None, timeout: float | None = None):
        return self._request("POST", path, body, timeout)


def is_asleep(head_pose: dict, antennas: list) -> bool:
    """True when the head/antennas are in (or near) the sleep pose."""
    if (head_pose.get("z") or 0.0) < SLEEP_Z_M:
        return True
    if abs(head_pose.get("pitch") or 0.0) > SLEEP_PITCH_RAD:
        return True
    return any(abs(a) > SLEEP_ANTENNA_RAD for a in antennas or [])


@dataclass
class Check:
    name: str
    status: str  # ok | fixed | warn | failed
    detail: str = ""


@dataclass
class PreflightResult:
    host: str
    reachable: bool = False
    checks: list[Check] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.reachable and not any(c.status == "failed" for c in self.checks)

    @property
    def exit_code(self) -> int:
        if not self.reachable:
            return 2
        return 0 if self.ready else 1

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({c.name: {"status": c.status, "detail": c.detail} for c in self.checks})
        data["ready"] = self.ready
        data["exit_code"] = self.exit_code
        return data

    def render(self) -> str:
        lines = [f"Reachy Mini preflight @ {self.host}"]
        lines += [f"  {ICONS[c.status]} {c.name:<9} {c.detail}" for c in self.checks]
        verdict = {0: "READY", 1: "NOT READY", 2: "UNREACHABLE"}[self.exit_code]
        lines.append(f"=> {verdict}" + (f" (fixed: {', '.join(self.fixes_applied)})" if self.fixes_applied else ""))
        return "\n".join(lines)


@dataclass
class _Ctx:
    api: Api
    status: dict
    result: PreflightResult
    fix: bool
    interactive: bool
    sleep: Callable[[float], None]
    confirm: Callable[[str], bool]


def _ask(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes", "s", "sim")


def _connect(host: str, api_factory) -> tuple:
    candidates = [host] if host == MDNS_HOST else [host, MDNS_HOST]
    for candidate in candidates:
        api = api_factory(candidate)
        try:
            return api, api.get("/api/daemon/status")
        except ApiError:
            continue
    return None, None


def _wait_moves(ctx: _Ctx, timeout: float) -> bool:
    for _ in range(int(timeout / 0.5) + 1):
        if not ctx.api.get("/api/move/running"):
            return True
        ctx.sleep(0.5)
    return False


def _check_daemon(ctx: _Ctx) -> bool:
    state = ctx.status.get("state")
    if state == "running":
        ctx.result.add("daemon", "ok", f"running, version {ctx.status.get('version')}")
        return True
    if ctx.fix and ctx.interactive and ctx.confirm(f"Daemon state is '{state}'. Start it?"):
        ctx.api.post("/api/daemon/start?wake_up=false", timeout=30)
        ctx.result.fixes_applied.append("daemon start")
        ctx.sleep(5)
        ctx.status = ctx.api.get("/api/daemon/status")
        if ctx.status.get("state") == "running":
            ctx.result.add("daemon", "fixed", f"{state} -> running")
            return True
    ctx.result.add(
        "daemon", "failed",
        f"state={state}; starting it needs user confirmation: POST /api/daemon/start?wake_up=false",
    )
    return False


def _check_backend(ctx: _Ctx) -> bool:
    backend = ctx.status.get("backend_status") or {}
    error = backend.get("error") or ctx.status.get("error")
    if backend.get("ready") and not error:
        ctx.result.add("backend", "ok", "ready")
        return True
    ctx.result.add(
        "backend", "failed",
        f"ready={backend.get('ready')} error={error}; read the daemon logs (see reachy-ssh skill)",
    )
    return False


def _check_app(ctx: _Ctx) -> bool:
    app = ctx.api.get("/api/apps/current-app-status")
    lock = ctx.api.get("/api/daemon/robot-app-lock-status") or {}
    if app is None and lock.get("state") in (None, "free"):
        ctx.result.add("app", "ok", "no app running")
    else:
        ctx.result.add(
            "app", "warn",
            f"app={app} lock={lock}: an app may be controlling the robot; "
            "stopping it needs user confirmation (POST /api/apps/stop-current-app)",
        )
    return True


def _check_moves(ctx: _Ctx) -> bool:
    if _wait_moves(ctx, 5.0):
        ctx.result.add("moves", "ok", "no move running")
    else:
        ctx.result.add("moves", "warn", "a move is still running after 5 s")
    return True


def _check_motors(ctx: _Ctx) -> bool:
    mode = (ctx.api.get("/api/motors/status") or {}).get("mode")
    if mode == "enabled":
        ctx.result.add("motors", "ok", "enabled")
        return True
    if not ctx.fix:
        ctx.result.add("motors", "failed", f"mode={mode}; run with --fix")
        return False
    ctx.api.post("/api/motors/set_mode/enabled")
    ctx.result.fixes_applied.append("motors enabled")
    ctx.sleep(0.5)
    new_mode = (ctx.api.get("/api/motors/status") or {}).get("mode")
    if new_mode == "enabled":
        ctx.result.add("motors", "fixed", f"{mode} -> enabled")
        return True
    ctx.result.add("motors", "failed", f"still mode={new_mode} after set_mode/enabled")
    return False


WAKE_UP_TIMEOUT = 5.0


def _pose_detail(state: dict) -> str:
    pose = state.get("head_pose") or {}
    return (
        f"head z={pose.get('z', 0):+.3f} m pitch={pose.get('pitch', 0):+.2f} rad "
        f"antennas={[round(a, 2) for a in state.get('antennas_position') or []]}"
    )


def _check_awake(ctx: _Ctx) -> bool:
    state = ctx.api.get("/api/state/full") or {}
    if not is_asleep(state.get("head_pose") or {}, state.get("antennas_position") or []):
        ctx.result.add("awake", "ok", _pose_detail(state))
        return True
    if not ctx.fix:
        ctx.result.add("awake", "failed", f"sleep pose ({_pose_detail(state)}); run with --fix")
        return False
    ctx.api.post("/api/move/play/wake_up")
    ctx.result.fixes_applied.append("wake_up")
    ctx.sleep(1.0)
    _wait_moves(ctx, 10.0)
    # wake_up interpolates for ~2 s and "no move running" can be reported before the pose has settled,
    # so the pose is polled instead of read once (this used to abort scripts with a still-asleep robot).
    state = _wait_awake(ctx, WAKE_UP_TIMEOUT)
    if not is_asleep(state.get("head_pose") or {}, state.get("antennas_position") or []):
        ctx.result.add("awake", "fixed", f"woke up ({_pose_detail(state)})")
        return True
    ctx.result.add("awake", "failed", f"still in sleep pose after wake_up ({_pose_detail(state)})")
    return False


def _wait_awake(ctx: _Ctx, timeout: float) -> dict:
    """Poll the robot state until it leaves the sleep pose; returns the last state seen."""
    state = {}
    for attempt in range(int(timeout / 0.5) + 1):
        if attempt:
            ctx.sleep(0.5)
        state = ctx.api.get("/api/state/full") or {}
        if not is_asleep(state.get("head_pose") or {}, state.get("antennas_position") or []):
            break
    return state


def _check_media(ctx: _Ctx) -> bool:
    media = ctx.api.get("/api/media/status") or {}
    if media.get("no_media"):
        ctx.result.add(
            "media", "failed",
            "daemon runs with no_media; restarting it with media needs user confirmation",
        )
        return False
    if media.get("available") and not media.get("released"):
        ctx.result.add("media", "ok", "camera/audio held by daemon")
        return True
    if not media.get("released"):
        ctx.result.add("media", "failed", f"media not available: {media}")
        return False
    if not ctx.fix:
        ctx.result.add("media", "failed", "camera/audio released ('hidden'); run with --fix")
        return False
    ctx.api.post("/api/media/acquire", timeout=15)
    ctx.result.fixes_applied.append("media acquired")
    ctx.sleep(1.0)
    media = ctx.api.get("/api/media/status") or {}
    if media.get("available") and not media.get("released"):
        ctx.result.add("media", "fixed", "released -> acquired")
        return True
    ctx.result.add("media", "failed", f"acquire did not work: {media}")
    return False


def _face_ts(ctx: _Ctx):
    body = ctx.api.get("/api/media/tracking/face") or {}
    return (body.get("face_target") or {}).get("ts")


def _ts_advances(ctx: _Ctx, polls: int = 6) -> bool:
    first = _face_ts(ctx)
    for _ in range(polls):
        ctx.sleep(0.5)
        ts = _face_ts(ctx)
        if ts is not None and ts != first:
            return True
    return False


def _check_tracking(ctx: _Ctx) -> bool:
    if _ts_advances(ctx):
        ctx.result.add("tracking", "ok", "tracker already receiving frames")
        return True
    # Tracking was not active: enable it just for the test, then restore (disable).
    ctx.api.post("/api/media/tracking/enable", {"weight": 1.0})
    try:
        if _ts_advances(ctx):
            ctx.result.add("tracking", "ok", "tracker receives frames")
            return True
        if ctx.fix:
            ctx.api.post("/api/media/tracking/disable")
            ctx.api.post("/api/media/acquire", timeout=15)
            ctx.result.fixes_applied.append("media re-acquired for tracker")
            ctx.sleep(1.0)
            ctx.api.post("/api/media/tracking/enable", {"weight": 1.0})
            if _ts_advances(ctx):
                ctx.result.add("tracking", "fixed", "tracker receives frames after media re-acquire")
                return True
        ctx.result.add(
            "tracking", "failed",
            "tracker gets no frames (face_target.ts not advancing); see reachy-ssh troubleshooting",
        )
        return False
    finally:
        ctx.api.post("/api/media/tracking/disable")


def run_preflight(
    host: str | None = None,
    fix: bool = False,
    needs=("motion",),
    interactive: bool = False,
    api_factory=Api,
    sleep: Callable[[float], None] = time.sleep,
    confirm: Callable[[str], bool] | None = None,
) -> PreflightResult:
    """Run the ordered checks; stop at the first failing one."""
    needs = set(needs)
    unknown = needs - set(VALID_NEEDS)
    if unknown:
        raise ValueError(f"unknown needs: {sorted(unknown)}")
    host = host or resolve_host()

    api, status = _connect(host, api_factory)
    if api is None:
        result = PreflightResult(host=host)
        result.add("network", "failed", NETWORK_HINT.format(host=host, mdns=MDNS_HOST))
        return result

    result = PreflightResult(host=api.host, reachable=True)
    if api.host != host:
        result.add(
            "network", "warn",
            f"{host} unreachable but {api.host} answered (wlan_ip={status.get('wlan_ip')}); "
            "update ROBOT_IP in robot/connect.py",
        )
    else:
        result.add("network", "ok", f"{host} (wlan_ip={status.get('wlan_ip')})")

    steps = [_check_daemon, _check_backend, _check_app, _check_moves, _check_motors, _check_awake]
    if needs & {"media", "tracking"}:
        steps.append(_check_media)
    if "tracking" in needs:
        steps.append(_check_tracking)

    ctx = _Ctx(api, status, result, fix, interactive, sleep, confirm or _ask)
    try:
        for step in steps:
            if not step(ctx):
                break
    except ApiError as exc:
        result.add("api", "failed", f"daemon stopped answering: {exc}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify (and fix) Reachy Mini state before acting.")
    parser.add_argument("--host", help="robot host (default: REACHY_HOST or ROBOT_IP)")
    parser.add_argument("--fix", action="store_true", help="auto-fix safe problems (motors, wake up, media)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--need", action="append", choices=VALID_NEEDS, help="extra requirements (repeatable)")
    args = parser.parse_args(argv)

    interactive = sys.stdin.isatty() and not args.json
    result = run_preflight(args.host, args.fix, tuple(args.need or ["motion"]), interactive=interactive)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.render())
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
