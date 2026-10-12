"""The coach: intro, framing, guided recording, training on the device, then the workout with the new model.

The camera loop calls `update(now, keypoints)` every frame and it never blocks. Speech and gestures run one
at a time on a worker thread (the `SessionEngine` model); the worker moves the phase on when it is done.
Head follow streams through a `TargetSender`; `GestureMini` pauses it while a gesture owns the head.
"""

from __future__ import annotations

import enum
import logging
import queue
import random
import threading
import time
from pathlib import Path

from core.coach import phrases
from core.coach.head import THINKING_GESTURE, GestureMini
from core.coach.recording import missing_parts, next_session_index, sanitize
from core.motion.actions import EXERCISE_MODEL, ActionDetector, ActionMonitor
from core.motion.antennas import NEUTRAL_ANTENNAS, AntennaWave, TargetSender, antennas_with_neutral
from core.session.config import SessionPlan
from core.session.engine import SessionEngine
from core.session.gestures import WELCOME_GESTURE, play_gesture_with_speech

logger = logging.getLogger(__name__)

WORKOUT = ("squat", "arm_raise", "side_bend")
FRAMING_HOLD_S = 5.0  # whole body visible this long -> recording starts, as in the collect kit
HINT_EVERY_S = 6.0  # at most one "step back" line this often
GO_DELAY_S = 1.0  # after the round announcement, before "Go!"
THINKING_EVERY_S = 20.0  # a "still thinking" line + gesture while the model trains
WHITE, GREEN, YELLOW, CYAN = (255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0)


class Phase(enum.Enum):
    INTRO = enum.auto()
    FRAMING = enum.auto()
    ANNOUNCE = enum.auto()  # the worker is speaking; it sets the next phase
    ROUND = enum.auto()
    TRAINING = enum.auto()
    WORKOUT = enum.auto()
    DONE = enum.auto()


class Pipeline:
    """One coach session for one person.

    Attributes:
        phase: Current `Phase`.
        caption: The last line said, for the HUD (the session also works without sound).
        model_path: The model the workout runs (set when the workout starts).
        saved_path: The recording written, or None.
    """

    def __init__(self, mini, voice, name, recorder, trainer, recordings_dir, pose_model, follow,
                 workout_reps=3, workout_timeout_s=30.0, fallback_model=EXERCISE_MODEL, workout_model=None,
                 rate_hz=50.0):
        self.voice = voice
        self.name = name
        self.recorder = recorder
        self.trainer = trainer
        self.recordings_dir = Path(recordings_dir)
        self.pose_model = pose_model
        self.follow = follow
        self.workout_reps = workout_reps
        self.workout_timeout_s = workout_timeout_s
        self.fallback_model = Path(fallback_model)
        self.workout_model = workout_model
        self.phase = Phase.INTRO
        self.caption = ""
        self.model_path = None
        self.saved_path = None
        self.monitor = None
        self.engine = None
        self._rng = random.Random()
        self._praise = AntennaWave(amplitude_deg=25.0, freq_hz=2.0, duration_s=1.0)
        self._visible_since = None
        self._last_hint = -HINT_EVERY_S
        self._last_thinking = 0.0
        self._thinking_count = 0
        self._free_count = 0
        self._tasks = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run_tasks, name="coach-tasks", daemon=True)
        self.sender = self.gesture_mini = None
        if mini is not None:
            from reachy_mini.utils import create_head_pose  # lazy, like core/session/gestures.py
            self._head_pose = create_head_pose
            self.sender = TargetSender(mini, self.antennas, rate_hz)
            self.gesture_mini = GestureMini(mini, self.sender)

    @property
    def finished(self):
        return self.phase is Phase.DONE

    # --- main thread ------------------------------------------------------------------------------------

    def start(self):
        """Start the head sender and the worker, and say hello (returns at once)."""
        if self.sender is not None:
            self.sender.start()
        self._worker.start()
        if self.workout_model is not None:  # --workout-only: straight to playing with a saved model
            self.phase = Phase.ANNOUNCE
            self._enqueue(lambda: self._task_workout(Path(self.workout_model), phrases.lets_play(self.name)))
        else:
            self._enqueue(self._task_intro)

    def update(self, now, keypoints):
        """Process one camera frame (`now` = `time.perf_counter()`). Never blocks."""
        if self.sender is not None:
            yaw, pitch = self.follow(keypoints, now)
            self.sender.head = self._head_pose(yaw=yaw, pitch=pitch, degrees=True)
        phase = self.phase
        if phase is Phase.FRAMING:
            self._update_framing(now, keypoints)
        elif phase is Phase.ROUND:
            self._update_round(now, keypoints)
        elif phase is Phase.TRAINING:
            self._update_training(now)
        elif phase is Phase.WORKOUT:
            if keypoints is not None:
                self.engine.update(now, keypoints)
            if self.engine.is_finished:
                self.phase = Phase.DONE

    def antennas(self, now):
        """[right, left] antenna targets for the `TargetSender`."""
        if self.phase is Phase.WORKOUT and self.monitor is not None:
            return self.monitor.antennas(now)
        angles = self._praise.angles(now)
        return list(NEUTRAL_ANTENNAS) if angles is None else antennas_with_neutral(angles)

    def hud_lines(self, now, keypoints=None):
        """(text, colour) rows for the camera overlay."""
        lines = [(f"COACH {self.name} - {self.phase.name}", GREEN)]
        rec = self.recorder
        if self.phase in (Phase.FRAMING, Phase.ANNOUNCE, Phase.ROUND) and not rec.finished:
            lines.append((f"round {min(rec.done + 1, rec.total)}/{rec.total}", WHITE))
        if self.phase is Phase.ROUND and rec.current:
            lines.append((f"{rec.current.upper()}  {rec.time_left(now):.1f} s", YELLOW))
        if self.phase is Phase.FRAMING:
            missing = missing_parts(keypoints, rec.min_score)
            lines.append(("not visible: " + ", ".join(missing) if missing else "hold still...", YELLOW))
        if self.phase is Phase.TRAINING:
            lines.append((f"learning... {self.trainer.elapsed:.0f} s", YELLOW))
        if self.phase is Phase.WORKOUT and self.engine is not None:
            engine = self.engine
            lines.append((f"{engine.state.name}  round {engine.current_round_idx + 1}  "
                          f"reps {engine.reps_this_round}", YELLOW))
        if self.caption:
            lines.append((self.caption, CYAN))
        return lines

    def close(self):
        """Stop everything; keep what was recorded if the session ended before training."""
        if self.recorder.has_rounds and self.saved_path is None:
            try:
                print(f"Saved what was recorded: {self._save()}")
            except OSError:
                logger.exception("Could not save the partial recording")
        self.trainer.stop()
        if self.engine is not None:
            self.engine.close()
        self._stop.set()
        self._worker.join(timeout=2.0)
        if self.sender is not None:
            self.sender.stop()

    def _update_framing(self, now, keypoints):
        missing = missing_parts(keypoints, self.recorder.min_score)
        if missing:
            self._visible_since = None
            if now - self._last_hint >= HINT_EVERY_S:
                self._last_hint = now
                self._say_now(phrases.step_back(missing))
            return
        if self._visible_since is None:
            self._visible_since = now
        if now - self._visible_since >= FRAMING_HOLD_S:
            lead = phrases.welcome_back() if self.recorder.has_rounds else phrases.ready()
            self.phase = Phase.ANNOUNCE
            self._enqueue(lambda: self._task_next_round(lead))

    def _update_round(self, now, keypoints):
        result = self.recorder.add(now, keypoints)
        if result is None:
            return
        print(f"[coach] {result.name}: ok={result.ok} visible={result.visible:.2f} motion={result.motion:.2f}")
        if not result.ok and result.gone:
            self.phase = Phase.FRAMING
            self._visible_since = None
            self._last_hint = now  # let "where did you go" finish before any "step back"
            self._say_now(phrases.gone())
            return
        if result.ok:
            self._praise.trigger(now)
            lead = phrases.praise(self._rng)
            if self.recorder.done == self.recorder.total // 2:
                lead += " " + phrases.halfway()
        else:
            lead = phrases.redo() if result.retry else phrases.move_on()
        self.phase = Phase.ANNOUNCE
        self._enqueue(lambda: self._task_next_round(lead))

    def _update_training(self, now):
        state = self.trainer.poll()
        if state == "running":
            if now - self._last_thinking >= THINKING_EVERY_S:
                self._last_thinking = now
                text = phrases.thinking(self._thinking_count)
                self._thinking_count += 1
                self._enqueue(lambda: self._say(text, THINKING_GESTURE))
            return
        self.phase = Phase.ANNOUNCE
        if state == "done":
            model, lead = self.trainer.out_path, phrases.learned()
        else:
            print(f"Training failed, using {self.fallback_model}:\n{self.trainer.log_tail()}")
            model, lead = self.fallback_model, phrases.training_failed()
        self._enqueue(lambda: self._task_workout(model, lead))

    def _say_now(self, text):
        """Speak without waiting, from the main thread."""
        self.caption = text
        print(f"[coach] {text}")
        if self.voice is not None:
            self.voice.speak(text, non_blocking=True)

    # --- worker thread ----------------------------------------------------------------------------------

    def _enqueue(self, task):
        self._tasks.put(task)

    def _run_tasks(self):
        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                task()
            except Exception:
                logger.exception("Coach task failed")

    def _say(self, text, gesture=None, wait=True):
        """Speak (and play `gesture`), waiting for the speech to end unless `wait` is False."""
        self.caption = text
        print(f"[coach] {text}")
        if gesture is not None and self.gesture_mini is not None:
            play_gesture_with_speech(self.gesture_mini, gesture, self.voice, text, wait_for_voice=wait)
        elif self.voice is not None:
            self.voice.speak(text, non_blocking=True)
            if wait:
                self.voice.wait(timeout=20.0)

    def _task_intro(self):
        self._say(phrases.intro(self.name), WELCOME_GESTURE)
        self._visible_since = None
        self.phase = Phase.FRAMING

    def _task_next_round(self, lead):
        rec = self.recorder
        if rec.finished:
            self._say(lead)
            self._task_train()
            return
        name = rec.pending
        text = f"{lead} {phrases.round_intro(name, self._free_count)}"
        if name == "none":
            self._free_count += 1
        self._say(text)
        time.sleep(GO_DELAY_S)
        self._say(phrases.go(), wait=False)
        rec.start_round(time.perf_counter())
        self.phase = Phase.ROUND

    def _save(self):
        person = sanitize(self.name)
        person_dir = self.recordings_dir / person
        session = next_session_index(person_dir)
        self.saved_path = self.recorder.save(person_dir / f"session_{session}.npz", person, session,
                                             self.pose_model)
        return self.saved_path

    def _task_train(self):
        try:
            print(f"Recording saved: {self._save()}")
        except OSError:
            logger.exception("Could not save the recording")
            self._say(phrases.save_failed())
            self.phase = Phase.DONE
            return
        self._say(phrases.thanks_training())
        self.trainer.start()
        self._last_thinking = time.perf_counter()
        self.phase = Phase.TRAINING

    def _task_workout(self, model, lead):
        detector = ActionDetector(model)
        if not detector.available or not set(WORKOUT) <= set(detector.classes):
            print(f"{model} cannot run the workout, using {self.fallback_model}")
            detector = ActionDetector(self.fallback_model)
        if not detector.available:
            self._say(phrases.training_failed())
            self.phase = Phase.DONE
            return
        self.model_path = detector.path
        self.monitor = ActionMonitor(self.recorder.aspect_ratio, min_score=self.recorder.min_score,
                                     detector=detector, voice=False)
        plan = SessionPlan.create(exercises=WORKOUT, reps_per_round=self.workout_reps,
                                  timeout_s=self.workout_timeout_s)
        self.engine = SessionEngine(plan, self.monitor, self.voice, mini=self.gesture_mini)
        self._say(lead)
        self.phase = Phase.WORKOUT
        self.engine.start()
