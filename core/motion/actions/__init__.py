"""Recognition of several body actions from a window of COCO-17 keypoints.

Trained on COCO-17 (not BlazePose's 33 landmarks) so one model runs both on the Mac and on the robot,
whose CPU cannot execute mediapipe at all.
"""

NONE = "none"
ACTIONS = (NONE, "wave", "pushup", "squat", "clapping", "jumping_jacks")
# A push-up or a squat repetition takes 2-3 s and does not fit the wave model's 1.5 s window.
ACTION_WINDOW_S = 3.0
