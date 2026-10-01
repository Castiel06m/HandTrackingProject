import time
import cv2

from signtext.config import HAND_MODEL, POSE_MODEL
from signtext.tracking.detector import HolisticDetector
from signtext.tracking.drawing import draw_result
from signtext.tracking.features import extract_features, FEATURE_DIM


def main():
    cap = cv2.VideoCapture(0)
    t0 = prev = time.perf_counter()
    fps = 0.0

    with HolisticDetector(HAND_MODEL, POSE_MODEL) as det:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            res = det.detect(frame, int((now - t0) * 1000))   # кадр НЕ зеркалим
            feats = extract_features(res)
            assert feats.shape == (FEATURE_DIM,)

            vis = cv2.flip(draw_result(frame, res), 1)        # зеркалим только для показа

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(vis, f"FPS {fps:.0f}  hands {res.hands_present.sum()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("SignText", vis)

            if cv2.waitKey(1) == 27:   # Esc
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()