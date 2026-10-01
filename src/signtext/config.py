from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
HAND_MODEL = ASSETS / "hand_landmarker.task"
POSE_MODEL = ASSETS / "pose_landmarker_full.task"