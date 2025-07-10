import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
# keep a global Pose instance so MediaPipe loads only once
_POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


def extract_pose_keypoints(image: np.ndarray) -> np.ndarray:
    """
    Returns a (33,3) array of [x, y, visibility] normalized to image size.
    If no landmarks -> zeros.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _POSE.process(img_rgb)
    if not result.pose_landmarks:
        return np.zeros((33, 3), dtype=np.float32)

    h, w = image.shape[:2]
    kp = [
        [lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks.landmark
    ]
    return np.asarray(kp, dtype=np.float32)
