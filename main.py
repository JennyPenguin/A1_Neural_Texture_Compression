import torch
from PIL import Image
import numpy as np
import numpy.typing as npt
import math

def get_device():
    # for Nvidia GPU
    if torch.cuda.is_available():
        return "cuda"
    # for Apple Silicon
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_image(path: str):
    return np.asarray(Image.open(path))

# Bilinear interpolation, img should have shape (w, h, 3)
def sample(u: float, v : float, img: np.ndarray):
    height, width, _ = img.shape
    u *= width
    v *= height
    # all coordinates are centered. I.e. x1 is center of top left pixel
    x1_org = math.floor(max(u - 0.5, 0.0))
    x1 = x1_org + 0.5
    # x2_org = max(x1_org + 2, width - 1)
    y1_org = math.floor(max(v - 0.5, 0.0))
    y1 = y1_org + 0.5
    # y2_org = max(y1_org + 2, height - 1)
    # Taking max for the edge corner where u, v is at the left/top edges
    s = max(u - x1, 0.0)
    t = max(v - y1, 0.0)
    interp_slice = img[y1_org: y1_org+2, x1_org: x1_org+2, :] 
    # Use -1 instead of 1 since may have only 1 row/col if at edge
    interp_y = interp_slice[0] * (1.0 - t) + interp_slice[-1] * t
    interp_x = interp_y[0] * (1.0 - s) + interp_y[-1] * s
    return np.reshape(interp_x, -1)

# Simple test to see if u,v coordinates are working correctly
def test_bilinear_downsample(img, down_size):
    height, width, _ = img.shape
    height //= down_size
    width //= down_size
    def sample_coord(r, c):
        return sample(c / width, r / height, img)
    # Inefficient but easy way of testing image
    downsampled = []
    for r in range(height):
        downsampled.append([])
        row = downsampled[-1]
        for c in range(width):
            row.append(sample_coord(r, c))
    return np.array(downsampled)

def compare_float_mat(m1, m2, e = 0.01):
    return np.allclose(m1, m2, rtol=e, atol=e)


img = load_image("textures/gradient.png")
print(img.shape)
# print(sample(0.2, 0.3, img))
downsampled = test_bilinear_downsample(img, 1)
converted = downsampled.astype(np.uint8)
print(compare_float_mat(downsampled, img))
Image.fromarray(converted).save("textures/gradient_downsampled_1.png")