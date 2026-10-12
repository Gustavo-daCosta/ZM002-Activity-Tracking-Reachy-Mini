"""State machine engine driving the interactive Reachy Mini exercise session.

Threading model
---------------
The camera loop runs in the main thread at ~10 FPS and must never block.
All blocking operations (goto_target(), voice.wait(), time.sleep() for rest)
are delegated to a single `_transition_thread` that runs one task at a time.
The camera loop only sets lightweight flags; the transition thread picks them up.
"""

from __future__ import annotations

from enum import Enum, auto
import logging
import queue
import threading
import time
from typing import Callable, Optional

from core.motion.actions import ActionMonitor
from core.motion.voice import ActionVoice
from core.session.config import (
    ExerciseRound,
    SessionPlan,
    get_outro_speech,
    get_rep_speech,
    get_round_complete_speech,
    get_round_intro_speech,
    get_welcome_speech,
)
from core.session.gestures import (
    OUTRO_CELEBRATION_GESTURE,
    ROUND_COMPLETE_GESTURE,
    ROUND_INTRO_GESTURE,
    WELCOME_GESTURE,
    play_gesture_with_speech,
)

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Lifecycle states of an exercise session."""

    IDLE = auto()
    INITIALIZING = auto()
    ROUND_INTRO = auto()
    ACTIVE = auto()
    ROUND_FEEDBACK = auto()
    SESSION_OUTRO = auto()
    FINISHED = auto()


class SessionEngine:
    """Orchestrates rounds, targeted recognition, real-time feedback, and gestures.

    The camera loop calls ``update()`` every frame — it never blocks.
    All gestures and state transitions are dispatched to ``_transition_thread``.

    Attributes:
        plan: The session configuration (rounds, reps per round, timeouts).
        state: Current ``SessionState``.
        current_round_idx: Index of current round (0-indexed).
        reps_this_round: Reps detected in the current round.
        total_reps_completed: Total reps counted across all finished rounds.
    """

    def __init__(
        self,
        plan: SessionPlan,
        monitor: ActionMonitor,
        voice: ActionVoice,
        mini=None,
        on_state_change: Optional[Callable[[SessionState, str], None]] = None,
    ) -> None:
        self.plan = plan
        self.monitor = monitor
        self.voice = voice
        self.mini = mini
        self.on_state_change = on_state_change

        self.state: SessionState = SessionState.IDLE
        self.current_round_idx: int = 0
        self.reps_this_round: int = 0
        self.total_reps_completed: int = 0
        self.rounds_completed: int = 0
        self.round_start_time: float = 0.0
        self._round_timed_out: bool = False

        # --- Thread safety -------------------------------------------------
        # A single background thread executes all blocking work (goto_target,
        # voice.wait, time.sleep). The camera loop only enqueues tasks.
        self._task_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._transition_thread = threading.Thread(
            target=self._run_transitions,
            name="session-transitions",
            daemon=True,
        )
        self._transition_thread.start()

    # ------------------------------------------------------------------
    # Public API (called from the main thread)
    # ------------------------------------------------------------------

    @property
    def current_round(self) -> ExerciseRound | None:
        """Return the current ExerciseRound or None if past end."""
        if 0 <= self.current_round_idx < len(self.plan.rounds):
            return self.plan.rounds[self.current_round_idx]
        return None

    @property
    def is_finished(self) -> bool:
        """Whether the workout session has completed."""
        return self.state == SessionState.FINISHED

    def start(self) -> None:
        """Kick off the session (non-blocking: returns immediately)."""
        self._transition(SessionState.INITIALIZING, "Starting session")
        self._enqueue(self._task_welcome)

    def update(self, now: float, keypoints) -> None:
        """Process one camera frame.  Called from the main loop — must never block."""
        if self.state != SessionState.ACTIVE:
            return

        cur = self.current_round
        if cur is None:
            return

        # Run pose → motion classification
        self.monitor.update(now, keypoints)

        # Check completion triggers (target reached or timeout)
        if self.reps_this_round >= cur.target_reps:
            self._trigger_round_complete(now, timed_out=False)
        elif now - self.round_start_time >= cur.timeout_s:
            self._trigger_round_complete(now, timed_out=True)

    def close(self) -> None:
        """Stop the background transition thread cleanly."""
        self._stop_event.set()
        self._transition_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Rep-detected callback (called from ActionMonitor inside update())
    # ------------------------------------------------------------------

    def _on_target_rep_detected(self, action: str, rep_count: int, now: float) -> None:
        """Called by ActionMonitor immediately after a valid target rep is counted.

        This runs in the main camera-loop thread, so it must be fast.
        Speak only (voice has its own async worker) and let the antenna
        reaction happen via the existing AntennaWave mechanism.
        """
        if self.state != SessionState.ACTIVE:
            return

        cur = self.current_round
        if cur is None:
            return

        self.reps_this_round = rep_count
        # Non-blocking: queued in the voice worker thread
        if self.voice is not None:
            rep_speech = get_rep_speech(rep_count, cur.target_reps)
            self.voice.speak(rep_speech, non_blocking=True)
        # Antenna reaction is already triggered by ActionMonitor.note() via
        # the existing AntennaWave mechanism sampled at 50 Hz — no extra call needed.

    # ------------------------------------------------------------------
    # Internal: round completion trigger (main thread, lightweight)
    # ------------------------------------------------------------------

    def _trigger_round_complete(self, now: float, timed_out: bool) -> None:
        """Switch state to ROUND_FEEDBACK and enqueue the blocking transition work."""
        if self.state != SessionState.ACTIVE:
            return  # already triggered

        cur = self.current_round
        if cur is None:
            return

        self._round_timed_out = timed_out
        self.total_reps_completed += self.reps_this_round
        self.rounds_completed += 1
        self.monitor.set_target_action(None)  # stop counting during feedback
        self._transition(
            SessionState.ROUND_FEEDBACK,
            f"Round {self.current_round_idx + 1} complete "
            f"({self.reps_this_round}/{cur.target_reps} reps, timed_out={timed_out})",
        )
        self._enqueue(self._task_round_feedback)

    # ------------------------------------------------------------------
    # Internal: background transition tasks (run in _transition_thread)
    # ------------------------------------------------------------------

    def _run_transitions(self) -> None:
        """Worker loop: execute one blocking task at a time."""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                task()
            except Exception as exc:
                logger.error("Transition task error: %s", exc, exc_info=True)
            finally:
                self._task_queue.task_done()

    def _enqueue(self, task: Callable) -> None:
        """Put a task on the background queue, discarding if already full."""
        try:
            self._task_queue.put_nowait(task)
        except queue.Full:
            logger.warning("Transition queue full; dropping task %s", task.__name__)

    def _task_welcome(self) -> None:
        """Blocking: play welcome gesture + speech, then intro first round."""
        welcome_text = get_welcome_speech(
            num_rounds=len(self.plan.rounds),
            reps_per_round=self.plan.reps_per_round,
        )
        play_gesture_with_speech(
            mini=self.mini,
            gesture=WELCOME_GESTURE,
            voice=self.voice,
            speech_text=welcome_text,
            wait_for_voice=True,
        )
        self._task_round_intro()

    def _task_round_intro(self) -> None:
        """Blocking: announce the current round, then activate recognition."""
        cur = self.current_round
        if cur is None:
            self._task_outro()
            return

        self.reps_this_round = 0
        self._round_timed_out = False
        self.monitor.set_target_action(cur.action, callback=self._on_target_rep_detected)

        intro_text = get_round_intro_speech(
            round_idx=self.current_round_idx + 1,
            total_rounds=len(self.plan.rounds),
            action=cur.action,
            target_reps=cur.target_reps,
        )
        self._transition(
            SessionState.ROUND_INTRO,
            f"Introducing round {self.current_round_idx + 1}: {cur.action}",
        )
        play_gesture_with_speech(
            mini=self.mini,
            gesture=ROUND_INTRO_GESTURE,
            voice=self.voice,
            speech_text=intro_text,
            wait_for_voice=True,
        )

        self.round_start_time = time.perf_counter()
        self._transition(
            SessionState.ACTIVE,
            f"Round {self.current_round_idx + 1} active: tracking {cur.action}",
        )

    def _task_round_feedback(self) -> None:
        """Blocking: congratulate, rest, then advance to next round or outro."""
        cur = self.plan.rounds[self.current_round_idx]  # capture before advancing
        feedback_text = get_round_complete_speech(
            action=cur.action,
            reps_done=self.reps_this_round,
            timed_out=self._round_timed_out,
        )
        play_gesture_with_speech(
            mini=self.mini,
            gesture=ROUND_COMPLETE_GESTURE,
            voice=self.voice,
            speech_text=feedback_text,
            wait_for_voice=True,
        )

        self.current_round_idx += 1
        if self.current_round_idx < len(self.plan.rounds):
            time.sleep(1.0)  # brief inter-round rest
            self._task_round_intro()
        else:
            self._task_outro()

    def _task_outro(self) -> None:
        """Blocking: celebrate the completed workout, then mark as finished."""
        self._transition(SessionState.SESSION_OUTRO, "Workout session complete")
        outro_text = get_outro_speech(
            total_reps=self.total_reps_completed,
            completed_rounds=self.rounds_completed,
            total_rounds=len(self.plan.rounds),
        )
        play_gesture_with_speech(
            mini=self.mini,
            gesture=OUTRO_CELEBRATION_GESTURE,
            voice=self.voice,
            speech_text=outro_text,
            wait_for_voice=True,
        )
        self._transition(SessionState.FINISHED, "Session finished")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: SessionState, detail: str = "") -> None:
        """Record state change and notify optional listener."""
        self.state = new_state
        logger.info("Session state -> %s: %s", new_state.name, detail)
        if self.on_state_change is not None:
            self.on_state_change(new_state, detail)
