import numpy as np
from .result import FrameResult

NOSE, L_SH, R_SH, L_EL, R_EL = 0, 11, 12, 13, 14

# 2 руки * 21 точка * 3 + 12 признаков тела + 2 маски = 140
FEATURE_DIM = 2 * 21 * 3 + 12 + 2


def extract_features(res: FrameResult) -> np.ndarray:
    a = res.aspect
    hands_feat = np.zeros((2, 21, 3), dtype=np.float32)
    wrists = np.zeros((2, 2), dtype=np.float32)

    for i in range(2):
        if not res.hands_present[i]:
            continue
        pts = res.hands[i].copy()
        pts[:, 0] *= a                       # убрать искажение от пропорций кадра
        wrists[i] = pts[0, :2]
        rel = pts - pts[0]                   # относительно запястья
        scale = np.linalg.norm(rel[9]) + 1e-6  # запястье -> основание среднего пальца
        hands_feat[i] = rel / scale

    body = np.zeros(12, dtype=np.float32)
    if res.pose_present:
        p = res.pose[:, :2].copy()
        p[:, 0] *= a
        center = (p[L_SH] + p[R_SH]) / 2
        width = np.linalg.norm(p[L_SH] - p[R_SH]) + 1e-6   # ширина плеч

        wrist_vs_shoulders = np.zeros((2, 2), dtype=np.float32)
        wrist_vs_nose = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            if res.hands_present[i]:
                wrist_vs_shoulders[i] = (wrists[i] - center) / width
                wrist_vs_nose[i] = (wrists[i] - p[NOSE]) / width
        elbows = (p[[L_EL, R_EL]] - center) / width

        body = np.concatenate(
            [wrist_vs_shoulders.ravel(), wrist_vs_nose.ravel(), elbows.ravel()]
        ).astype(np.float32)

    return np.concatenate(
        [hands_feat.ravel(), body, res.hands_present.astype(np.float32)]
    )