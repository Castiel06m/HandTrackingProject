import numpy as np
from signtext.tracking.result import FrameResult
from signtext.tracking.features import extract_features, FEATURE_DIM


def fake(scale=1.0, shift=(0.0, 0.0)):
    rng = np.random.default_rng(0)
    hands = rng.random((2, 21, 3)).astype(np.float32) * scale
    pose = rng.random((33, 3)).astype(np.float32) * scale
    for arr in (hands, pose):
        arr[..., 0] += shift[0]
        arr[..., 1] += shift[1]
    return FrameResult(hands, np.array([True, True]), pose, True, 1.0)


def test_shape():
    assert extract_features(fake()).shape == (FEATURE_DIM,)


def test_scale_and_shift_invariance():
    a = extract_features(fake(1.0, (0, 0)))
    b = extract_features(fake(0.5, (0.2, 0.1)))   # дальше от камеры и сдвинуто
    assert np.allclose(a, b, atol=1e-4)


def test_no_hands():
    res = fake()
    res.hands_present[:] = False
    f = extract_features(res)
    assert np.all(f[:126] == 0) and np.all(f[-2:] == 0)