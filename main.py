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
    dim = img.shape
    u *= dim[1]
    v *= dim[0]
    # all coordinates are centered. I.e. x1 is center of top left pixel
    x1_org = math.floor(max(u - 0.5, 0.0))
    x1 = x1_org + 0.5
    y1_org = math.floor(max(v - 0.5, 0.0))
    y1 = y1_org + 0.5
    # Taking max for the edge corner where u, v is at the left/top edges
    s = max(u - x1, 0.0)
    t = max(v - y1, 0.0)
    interp_slice = img[x1_org: x1_org + 2, y1_org: y1_org + 2, :] 
    interp_y = interp_slice[0] * t + interp_slice[1] * (1.0 - t)
    interp_x = interp_y[0] * s + interp_y[1] * (1.0 - s)
    return interp_x


img = load_image("textures/time spiral.png")
print(img.shape)
print(sample(0.2, 0.3, img))