"""Runs `training.train_exercises` in a child process, so scikit-learn never loads into the camera loop.

`core/` must not import `training/`; a subprocess keeps that boundary and hands the memory back when the
fit ends. `nice` keeps the camera and head follow smooth on the Raspberry Pi CM4 while the forest trains.
The model is written to a temporary file and only replaces `out_path` after a successful fit, so a failed
run never destroys the child's previous model.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN_TIMEOUT_S = 15 * 60


class Trainer:
    """One training run.

    Attributes:
        out_path: Where the model ends up.
        log_path: The subprocess output (next to the model).
        state: "idle", "running", "done" or "failed".
    """

    def __init__(self, recordings_dir, out_path, timeout_s=TRAIN_TIMEOUT_S, python=sys.executable):
        self.recordings_dir = Path(recordings_dir)
        self.out_path = Path(out_path)
        self.log_path = self.out_path.with_suffix(".log")
        self._partial = self.out_path.with_suffix(".training.npz")
        self.timeout_s = timeout_s
        self.python = python
        self.state = "idle"
        self._process = None
        self._log = None
        self._started = None

    @property
    def elapsed(self):
        """Seconds since `start` (0 before)."""
        return 0.0 if self._started is None else time.monotonic() - self._started

    def start(self):
        """Launch the fit in the background."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._partial.unlink(missing_ok=True)
        self._log = open(self.log_path, "w")
        command = ["nice", "-n", "10", self.python, "-m", "training.train_exercises",
                   "--recordings", str(self.recordings_dir), "--out", str(self._partial)]
        self._process = subprocess.Popen(command, cwd=REPO, stdout=self._log, stderr=subprocess.STDOUT)
        self._started = time.monotonic()
        self.state = "running"

    def poll(self):
        """Current state; kills the fit once it runs past `timeout_s`."""
        if self.state != "running":
            return self.state
        code = self._process.poll()
        if code is None:
            if self.elapsed < self.timeout_s:
                return "running"
            self._process.kill()
            self._process.wait()
            self._log.write(f"\nKilled after {self.timeout_s:.0f} s.\n")
        self._log.close()
        if code == 0 and self._partial.exists():
            self._partial.replace(self.out_path)
            self.state = "done"
        else:
            self.state = "failed"
        return self.state

    def stop(self):
        """Kill a running fit (on abort)."""
        if self.state == "running":
            self._process.kill()
            self._process.wait()
            self._log.close()
            self.state = "failed"

    def log_tail(self, lines=20):
        """The last `lines` lines of the subprocess output."""
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
