import os
import cv2
import numpy as np
import pytest

def _read(path):
    return None if not path or not os.path.exists(path) else cv2.imread(path, cv2.IMREAD_COLOR)

@pytest.fixture(scope="session")
def img_paths():
    return os.environ.get("TEST_IMG1"), os.environ.get("TEST_IMG2")

@pytest.fixture(scope="session")
def images(img_paths):
    p1, p2 = img_paths
    im1, im2 = _read(p1), _read(p2)
    if im1 is None or im2 is None:
        pytest.skip("Set TEST_IMG1 and TEST_IMG2 to valid image paths.")
    return im1, im2

@pytest.fixture(scope="session")
def approx_K(images):
    im1, _ = images
    h, w = im1.shape[:2]
    f = 1.2 * max(h, w)  # conservative focal estimate
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0, 1.0]], dtype=float)
