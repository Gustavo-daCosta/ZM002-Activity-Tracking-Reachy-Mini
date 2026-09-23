from robot.preflight import ApiError, is_asleep, run_preflight

SLEEP_STATE = {
    "head_pose": {"x": -0.022, "y": 0.002, "z": -0.0456, "roll": 0.031, "pitch": 0.481, "yaw": -0.014},
    "antennas_position": [2.65, 2.99],
}
AWAKE_STATE = {
    "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "antennas_position": [0.0, 0.0],
}


class Seq:
    """Successive GET responses; the last one repeats forever."""

    def __init__(self, *items):
        self.items = list(items)

    def next(self):
        return self.items.pop(0) if len(self.items) > 1 else self.items[0]


def healthy_routes():
    return {
        "/api/daemon/status": {
            "state": "running", "version": "1.10.0", "wlan_ip": "192.168.137.171",
            "error": None, "backend_status": {"ready": True, "error": None},
        },
        "/api/apps/current-app-status": None,
        "/api/daemon/robot-app-lock-status": {"state": "free", "holder_name": None},
        "/api/move/running": Seq([]),
        "/api/motors/status": {"mode": "enabled"},
        "/api/state/full": AWAKE_STATE,
    }


def make_factory(routes, unreachable=()):
    posts = []

    class FakeApi:
        def __init__(self, host):
            self.host = host

        def get(self, path, timeout=None):
            if self.host in unreachable:
                raise ApiError(f"GET {path}: unreachable")
            value = routes[path]
            return value.next() if isinstance(value, Seq) else value

        def post(self, path, body=None, timeout=None):
            posts.append(path)
            return {"status": "ok"}

    return FakeApi, posts


def statuses(result):
    return {c.name: c.status for c in result.checks}


def test_is_asleep_detects_sleep_pose():
    assert is_asleep(SLEEP_STATE["head_pose"], SLEEP_STATE["antennas_position"])


def test_is_asleep_false_for_neutral_pose():
    assert not is_asleep(AWAKE_STATE["head_pose"], AWAKE_STATE["antennas_position"])


def test_is_asleep_antennas_alone():
    assert is_asleep(AWAKE_STATE["head_pose"], [0.0, 2.0])


def test_healthy_robot_is_ready_without_posts():
    factory, posts = make_factory(healthy_routes())
    result = run_preflight("robot", api_factory=factory, sleep=lambda s: None)
    assert result.ready and result.exit_code == 0
    assert posts == []


def test_unreachable_tries_mdns_and_exits_2():
    factory, _ = make_factory(healthy_routes(), unreachable=("robot", "reachy-mini.local"))
    result = run_preflight("robot", api_factory=factory, sleep=lambda s: None)
    assert result.exit_code == 2
    assert statuses(result) == {"network": "failed"}
    assert "same Wi-Fi" in result.checks[0].detail


def test_mdns_fallback_warns_but_is_ready():
    factory, _ = make_factory(healthy_routes(), unreachable=("robot",))
    result = run_preflight("robot", api_factory=factory, sleep=lambda s: None)
    assert result.ready
    assert result.host == "reachy-mini.local"
    assert statuses(result)["network"] == "warn"


def test_sleeping_robot_is_fixed():
    routes = healthy_routes()
    routes["/api/motors/status"] = Seq({"mode": "disabled"}, {"mode": "enabled"})
    routes["/api/state/full"] = Seq(SLEEP_STATE, AWAKE_STATE)
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, api_factory=factory, sleep=lambda s: None)
    assert result.ready
    assert posts == ["/api/motors/set_mode/enabled", "/api/move/play/wake_up"]
    assert statuses(result)["motors"] == "fixed"
    assert statuses(result)["awake"] == "fixed"


def test_wake_up_is_given_time_to_finish():
    # wake_up interpolates for ~2 s: the pose must be re-checked, not read once right after the call.
    routes = healthy_routes()
    routes["/api/state/full"] = Seq(SLEEP_STATE, SLEEP_STATE, SLEEP_STATE, AWAKE_STATE)
    factory, _ = make_factory(routes)
    result = run_preflight("robot", fix=True, api_factory=factory, sleep=lambda s: None)
    assert result.ready
    assert statuses(result)["awake"] == "fixed"


def test_wake_up_that_never_finishes_is_reported():
    routes = healthy_routes()
    routes["/api/state/full"] = SLEEP_STATE
    factory, _ = make_factory(routes)
    result = run_preflight("robot", fix=True, api_factory=factory, sleep=lambda s: None)
    assert not result.ready
    assert statuses(result)["awake"] == "failed"


def test_sleeping_robot_without_fix_is_not_ready():
    routes = healthy_routes()
    routes["/api/motors/status"] = {"mode": "disabled"}
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=False, api_factory=factory, sleep=lambda s: None)
    assert result.exit_code == 1
    assert posts == []


def test_stopped_daemon_not_started_when_non_interactive():
    routes = healthy_routes()
    routes["/api/daemon/status"] = {"state": "stopped", "backend_status": None, "error": None}
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, interactive=False, api_factory=factory, sleep=lambda s: None)
    assert result.exit_code == 1
    assert statuses(result)["daemon"] == "failed"
    assert posts == []


def test_stopped_daemon_started_after_confirmation():
    routes = healthy_routes()
    running = routes["/api/daemon/status"]
    routes["/api/daemon/status"] = Seq({"state": "stopped", "backend_status": None, "error": None}, running)
    factory, posts = make_factory(routes)
    result = run_preflight(
        "robot", fix=True, interactive=True, confirm=lambda q: True,
        api_factory=factory, sleep=lambda s: None,
    )
    assert result.ready
    assert posts == ["/api/daemon/start?wake_up=false"]


def test_backend_error_fails():
    routes = healthy_routes()
    routes["/api/daemon/status"] = {
        "state": "running", "error": None, "backend_status": {"ready": False, "error": "motor timeout"},
    }
    factory, _ = make_factory(routes)
    result = run_preflight("robot", fix=True, api_factory=factory, sleep=lambda s: None)
    assert statuses(result)["backend"] == "failed"
    assert "motor timeout" in result.checks[-1].detail


def test_running_app_only_warns():
    routes = healthy_routes()
    routes["/api/apps/current-app-status"] = {"name": "some_app", "state": "running"}
    routes["/api/daemon/robot-app-lock-status"] = {"state": "locked", "holder_name": "some_app"}
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, api_factory=factory, sleep=lambda s: None)
    assert result.ready
    assert statuses(result)["app"] == "warn"
    assert "/api/apps/stop-current-app" not in posts


def test_media_not_checked_for_motion_only():
    factory, _ = make_factory(healthy_routes())  # no media routes -> KeyError if queried
    result = run_preflight("robot", needs=("motion",), api_factory=factory, sleep=lambda s: None)
    assert "media" not in statuses(result)


MEDIA_OK = {"available": True, "released": False, "no_media": False}


def face(ts):
    return {"status": "ok", "face_target": {"detected": False, "x": None, "y": None, "roll": None, "ts": ts}}


def test_released_media_is_acquired():
    routes = healthy_routes()
    routes["/api/media/status"] = Seq({"available": False, "released": True, "no_media": False}, MEDIA_OK)
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, needs=("media",), api_factory=factory, sleep=lambda s: None)
    assert result.ready
    assert "/api/media/acquire" in posts
    assert statuses(result)["media"] == "fixed"


def test_no_media_daemon_fails_without_posts():
    routes = healthy_routes()
    routes["/api/media/status"] = {"available": False, "released": False, "no_media": True}
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, needs=("media",), api_factory=factory, sleep=lambda s: None)
    assert statuses(result)["media"] == "failed"
    assert posts == []


def test_tracking_already_running_is_left_alone():
    routes = healthy_routes()
    routes["/api/media/status"] = MEDIA_OK
    routes["/api/media/tracking/face"] = Seq(face(1.0), face(2.0))
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, needs=("tracking",), api_factory=factory, sleep=lambda s: None)
    assert statuses(result)["tracking"] == "ok"
    assert posts == []


def test_tracking_enabled_for_test_then_restored():
    routes = healthy_routes()
    routes["/api/media/status"] = MEDIA_OK
    # 8 reads without frames (7 idle polls + first read after enable), then frames
    routes["/api/media/tracking/face"] = Seq(*([face(None)] * 8), face(5.0), face(6.0))
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, needs=("tracking",), api_factory=factory, sleep=lambda s: None)
    assert statuses(result)["tracking"] == "ok"
    assert posts == ["/api/media/tracking/enable", "/api/media/tracking/disable"]


def test_tracking_without_frames_fails_and_restores():
    routes = healthy_routes()
    routes["/api/media/status"] = MEDIA_OK
    routes["/api/media/tracking/face"] = face(None)
    factory, posts = make_factory(routes)
    result = run_preflight("robot", fix=True, needs=("tracking",), api_factory=factory, sleep=lambda s: None)
    assert statuses(result)["tracking"] == "failed"
    assert posts[-1] == "/api/media/tracking/disable"
    assert "/api/media/acquire" in posts
