import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .result import FrameResult


class HolisticDetector:
    def __init__(self, hand_model, pose_model, num_hands=2):
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        self._last_ts = -1

    def detect(self, frame_bgr, timestamp_ms: int) -> FrameResult:
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_res = self.hand.detect_for_video(img, ts)
        pose_res = self.pose.detect_for_video(img, ts)

        hands = np.zeros((2, 21, 3), dtype=np.float32)
        present = np.zeros(2, dtype=bool)
        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            idx = 0 if handed[0].category_name == "Left" else 1
            if present[idx]:            # обе руки с одной меткой
                idx = 1 - idx
                if present[idx]:
                    continue
            hands[idx] = [[p.x, p.y, p.z] for p in lms]
            present[idx] = True

        pose = np.zeros((33, 3), dtype=np.float32)
        pose_present = len(pose_res.pose_landmarks) > 0
        if pose_present:
            pose[:] = [[p.x, p.y, p.z] for p in pose_res.pose_landmarks[0]]

        return FrameResult(hands, present, pose, pose_present, w / h)

    def close(self):
        self.hand.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()