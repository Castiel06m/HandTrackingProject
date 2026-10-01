from dataclasses import dataclass
import numpy as np


@dataclass
class FrameResult:
    hands: np.ndarray          # слот 0 и слот 1, нули если руки нет
    hands_present: np.ndarray  # bool
    pose: np.ndarray           # нули если тела нет
    pose_present: bool
    aspect: float              # width / height кадра