import cv2
import numpy as np
import mediapipe as mp

from .result import FrameResult

HAND_CONNECTIONS = [
    (c.start, c.end)
    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

HAND_COLORS = [(0, 255, 0), (255, 128, 0)]   # слот 0 / слот 1 (BGR)


def _draw_skeleton(img, pts, connections, color, radius):
    h, w = img.shape[:2]
    px = (pts[:, :2] * [w, h]).astype(int)
    for a, b in connections:
        cv2.line(img, tuple(px[a]), tuple(px[b]), (255, 255, 255), 2)
    for x, y in px:
        cv2.circle(img, (x, y), radius, color, -1)


def draw_result(frame_bgr, res: FrameResult):
    img = np.copy(frame_bgr)
    if res.pose_present:
        pose = res.pose[[0, 11, 12, 13, 14, 15, 16]]
        idx_map = {0: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6}
        conns = [(idx_map[a], idx_map[b]) for a, b in POSE_CONNECTIONS]
        _draw_skeleton(img, pose, conns, (0, 0, 255), 5)
    for i in range(2):
        if res.hands_present[i]:
            _draw_skeleton(img, res.hands[i], HAND_CONNECTIONS, HAND_COLORS[i], 4)
    return img