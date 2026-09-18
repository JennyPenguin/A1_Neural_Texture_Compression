# Unit tests written with Gemini

import numpy as np
import pytest
from main import sample

def test_single_pixel_exact():
    """Test sampling a 1x1 image at origin."""
    img = np.array([[[100, 150, 200]]], dtype=np.uint8)
    result = sample(0.5, 0.5, img)
    np.testing.assert_allclose(result, [100, 150, 200], atol=1e-3)

def test_boundary_pixel():
    """Test sampling a 1x1 image at corner."""
    img = np.array([[[100, 150, 200]]], dtype=np.uint8)
    result = sample(0.0, 0.0, img)
    np.testing.assert_allclose(result, [100, 150, 200], atol=1e-3)

def test_exact_pixel_center_sample():
    """
    In a 4x4 image:
    Pixel at index (x=1, y=2) has bounds x in [1.0, 2.0), y in [2.0, 3.0).
    Its center is at continuous (1.5, 2.5), corresponding to u = 1.5/4 = 0.375, v = 2.5/4 = 0.625.
    """
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[2, 1] = [255, 0, 0]  # Row 2 (y=2), Col 1 (x=1) set to pure Red

    # u = 1.5 / 4, v = 2.5 / 4
    result = sample(0.375, 0.625, img)
    np.testing.assert_allclose(result, [255, 0, 0], atol=1e-5)

def test_four_pixel_center_blend():
    """
    The point where 4 pixels meet in a 2x2 image is at continuous (1.0, 1.0).
    Corresponding to u = 0.5, v = 0.5.
    """
    img = np.array([
        [[100, 0, 0], [0, 100, 0]],
        [[0, 0, 100], [100, 100, 100]]
    ], dtype=np.float32)

    # u=0.5, v=0.5 maps to exact intersection (1.0, 1.0) of all 4 pixel centers
    result = sample(0.5, 0.5, img)
    np.testing.assert_allclose(result, [50, 50, 50], atol=1e-5)

def test_nonsquare_exact_pixel_center():
    """
    Test a 2x4 image (H=2, W=4).
    Targeting the exact center of pixel at row 1, col 2.
    Center is at continuous (x=2.5, y=1.5).
    u = 2.5 / 4 = 0.625
    v = 1.5 / 2 = 0.75
    """
    img = np.zeros((2, 4, 3), dtype=np.float32)
    img[1, 2] = [100, 150, 200]  # row 1 (y=1), col 2 (x=2)

    result = sample(0.625, 0.75, img)
    np.testing.assert_allclose(result, [100.0, 150.0, 200.0], atol=1e-5)

def test_nonsquare_fractional_blend():
    """
    Test a 2x3 image (H=2, W=3).
    Targeting continuous (x=1.75, y=0.75).
    u = 1.75 / 3 (approx 0.5833)
    v = 0.75 / 2 = 0.375
    
    This point lies between:
    - Top-Left   (x=1.5, y=0.5): [20, 20, 20]
    - Top-Right  (x=2.5, y=0.5): [30, 30, 30]
    - Bottom-Left(x=1.5, y=1.5): [50, 50, 50]
    - Bottom-Right(x=2.5,y=1.5): [60, 60, 60]
    
    Weights (x_offset = 0.25, y_offset = 0.25):
    TL weight: (1-0.25) * (1-0.25) = 0.5625
    TR weight: 0.25 * (1-0.25)     = 0.1875
    BL weight: (1-0.25) * 0.25     = 0.1875
    BR weight: 0.25 * 0.25         = 0.0625
    
    Expected result = (0.5625 * 20) + (0.1875 * 30) + (0.1875 * 50) + (0.0625 * 60)
                    = 11.25 + 5.625 + 9.375 + 3.75
                    = 30.0
    """
    img = np.array([
        [[10, 10, 10], [20, 20, 20], [30, 30, 30]],
        [[40, 40, 40], [50, 50, 50], [60, 60, 60]]
    ], dtype=np.float32)

    u = 1.75 / 3.0
    v = 0.375
    
    result = sample(u, v, img)
    np.testing.assert_allclose(result, [30.0, 30.0, 30.0], atol=1e-5)

def test_nonsquare_horizontal_only_blend():
    """
    Test a 2x5 image (H=2, W=5).
    Targeting continuous (x=3.8, y=1.5).
    u = 3.8 / 5 = 0.76
    v = 1.5 / 2 = 0.75
    
    y=1.5 is exactly the center of row 1, so y_offset = 0 (no vertical blending).
    x=3.8 is between col 3 (center 3.5) and col 4 (center 4.5).
    x_offset = 3.8 - 3.5 = 0.3
    
    Expected result interpolates 70% of col 3 and 30% of col 4:
    0.7 * [100, 100, 100] + 0.3 * [200, 200, 200] = [70 + 60] = [130, 130, 130]
    """
    img = np.zeros((2, 5, 3), dtype=np.float32)
    img[1, 3] = [100, 100, 100]
    img[1, 4] = [200, 200, 200]

    result = sample(0.76, 0.75, img)
    np.testing.assert_allclose(result, [130.0, 130.0, 130.0], atol=1e-5)

# A shared 2x2 image for these tests to make the math perfectly traceable.
# Top-Left (0.5, 0.5):     100
# Top-Right (1.5, 0.5):    200
# Bottom-Left (0.5, 1.5):  300
# Bottom-Right (1.5, 1.5): 400
@pytest.fixture
def test_image_2x2():
    return np.array([
        [[100, 100, 100], [200, 200, 200]],
        [[300, 300, 300], [400, 400, 400]]
    ], dtype=np.float32)

def test_horizontal_skew(test_image_2x2):
    """
    Test continuous (x=0.75, y=0.5).
    u = 0.75 / 2 = 0.375
    v = 0.5 / 2 = 0.25
    
    dx = 0.75 - 0.5 = 0.25
    dy = 0.5 - 0.5 = 0.0
    
    Weights:
    TL: (1 - 0.25) * (1 - 0) = 0.75
    TR: 0.25 * (1 - 0) = 0.25
    BL & BR: 0
    
    Expected: 0.75 * 100 + 0.25 * 200 = 75 + 50 = 125
    """
    result = sample(0.375, 0.25, test_image_2x2)
    np.testing.assert_allclose(result, [125.0, 125.0, 125.0], atol=1e-5)

def test_heavy_bottom_right_skew(test_image_2x2):
    """
    Test continuous (x=1.4, y=1.4).
    u = 1.4 / 2 = 0.7
    v = 1.4 / 2 = 0.7
    
    dx = 1.4 - 0.5 = 0.9
    dy = 1.4 - 0.5 = 0.9
    
    Weights:
    TL: (1 - 0.9) * (1 - 0.9) = 0.01
    TR: 0.9 * (1 - 0.9) = 0.09
    BL: (1 - 0.9) * 0.9 = 0.09
    BR: 0.9 * 0.9 = 0.81
    
    Expected: 1 + 18 + 27 + 324 = 370
    """
    result = sample(0.7, 0.7, test_image_2x2)
    np.testing.assert_allclose(result, [370.0, 370.0, 370.0], atol=1e-5)

def test_asymmetric_skew(test_image_2x2):
    """
    Test continuous (x=1.3, y=0.7).
    u = 1.3 / 2 = 0.65
    v = 0.7 / 2 = 0.35
    
    dx = 1.3 - 0.5 = 0.8
    dy = 0.7 - 0.5 = 0.2
    
    Weights:
    TL: (1 - 0.8) * (1 - 0.2) = 0.2 * 0.8 = 0.16
    TR: 0.8 * (1 - 0.2) = 0.8 * 0.8 = 0.64
    BL: (1 - 0.8) * 0.2 = 0.2 * 0.2 = 0.04
    BR: 0.8 * 0.2 = 0.16
    
    Expected: 16 + 128 + 12 + 64 = 220
    """
    result = sample(0.65, 0.35, test_image_2x2)
    np.testing.assert_allclose(result, [220.0, 220.0, 220.0], atol=1e-5)

def test_decimal_fraction_skew(test_image_2x2):
    """
    Test continuous (x=0.7, y=1.1).
    u = 0.7 / 2 = 0.35
    v = 1.1 / 2 = 0.55
    
    dx = 0.7 - 0.5 = 0.2
    dy = 1.1 - 0.5 = 0.6
    
    Weights:
    TL: (1 - 0.2) * (1 - 0.6) = 0.8 * 0.4 = 0.32
    TR: 0.2 * (1 - 0.6) = 0.2 * 0.4 = 0.08
    BL: (1 - 0.2) * 0.6 = 0.8 * 0.6 = 0.48
    BR: 0.2 * 0.6 = 0.12
    
    Expected: 32 + 16 + 144 + 48 = 240
    """
    result = sample(0.35, 0.55, test_image_2x2)
    np.testing.assert_allclose(result, [240.0, 240.0, 240.0], atol=1e-5)