import numpy as np
import numpy.typing as npt
import math

# Self Modules
import pytorch_helpers
from PIL import Image

###############################################################################
#`                              Image Helpers                                 #
###############################################################################
""" Load image with values all in [0, 255] range.
[in]    path:str    Path name of image to load   
[out]   Loaded image, each pixel is stored as a byte (0-255)
"""
def load_image(path: str):
    return np.asarray(Image.open(path))

###############################################################################
#`                       Bilinear Interpolation`                              #
###############################################################################

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
        return sample((c + 0.5) / width, (r + 0.5) / height, img)
    # Inefficient but easy way of testing image
    downsampled = []
    for r in range(height):
        downsampled.append([])
        row = downsampled[-1]
        for c in range(width):
            row.append(sample_coord(r, c))
    return np.array(downsampled)

###############################################################################
#`                           Color Converters                                 #
###############################################################################
""" Convert image from uint8 scale to [0, 1]. May loss some precision.
[in]    img:np.ndarry   Image to convert. All values should be in [0,255]
[out]   img with all values in range [0, 1]
"""
def normalize_image(img: np.ndarray):
    return img.astype(np.float32) / 255.0

def denormalize_image(img: np.ndarray):
    return (np.round(img * 255.0)).astype(np.uint8)

""" Quantize rgb color in [0, 1] to be stored in 2 bytes (565 encoding)
[in]    rgb:np.ndarry   The initial rgb colors, values all in [0, 1]
[out]   (2 byte encoding: np.uint16, rgb': new rgb colors in [0, 1])
"""
def quantize_rgb(rgb: np.ndarray):
    levels = (31, 63, 31) 
    q = np.round(rgb * levels).astype(np.uint16)
    new_rgb = q / levels # convert back to [0, 1]
    compressed = q[0] << 11 | q[1] << 5 | q[2]
    return (np.uint16(compressed), new_rgb)

def dequantize_rgb(color: np.uint16):
    levels = (31.0, 63.0, 31.0) 
    decoded = np.array([((color >> 11) & 0x1F), ((color >> 5) & 0x3F), (color & 0x1F)])
    C0 = decoded / levels
    return C0

###############################################################################
#`                              S3TC Compression                              #
###############################################################################

# pass in 4x4 block (r, c is top left corner of square)
def S3TC_encode_square(img, r, c):
    square = img[r:r+4, c:c+4]
    C0 = np.min(square, axis = (0, 1))
    C1 = np.max(square, axis = (0, 1))
    # Use the quantized C0 & C1 for more accurate comparison
    (C0_565, C0) = quantize_rgb(C0)
    (C1_565, C1) = quantize_rgb(C1)
    C2 = (2.0 * C0 + 1.0 * C1) / 3.0
    C3 = (1.0 * C0 + 2.0 * C1) / 3.0
    C_palette = np.array([C0, C1, C2, C3])

    indices = np.uint32(0)
    cnt = 0
    for r in range(4):
        for c in range(4):
            cell = square[r, c, :]
            best_diff = float('inf')
            best_index : np.uint32 = 0
            for i in range(4):
                diff = C_palette[i] - cell
                squared_err = np.dot(diff.T, diff)
                if (squared_err < best_diff):
                    best_index = np.uint32(i)
                    best_diff = squared_err
            # assert (best_index < 4, "Index out of bound?")
            # assert (best_index == float('inf'), "Never changed best")
            indices |= best_index << (2 * cnt)
            cnt += 1
    # assert (16 == cnt)

    color_encoded = np.uint32(C0_565) << 16
    color_encoded |= np.uint16(C1_565)
    return (color_encoded, indices)

# pass in 4x4 block (r, c is top left corner of square)
def S3TC_decode_square(color_encoded, indices_encoded):
    square = np.zeros((4, 4, 3))
    C0 = dequantize_rgb((color_encoded >> 16) & 0xFFFF)
    C1 = dequantize_rgb((color_encoded) & 0xFFFF)
    C2 = (2.0 * C0 + 1.0 * C1) / 3.0
    C3 = (1.0 * C0 + 2.0 * C1) / 3.0
    C_palette = [C0, C1, C2, C3]

    cnt = 0
    for r in range(4):
        for c in range(4):
            index = (indices_encoded >> (2 * cnt)) & 0x3
            square[r, c, :] = C_palette[index]
            cnt += 1

    return square

# TODO: Make this work for any dimension
# TODO: Write with faster numpy operations
""" Compress a normal image with values in range [0, 1] and only RGB channels
Only works on images that have dimensions which are multiples of 4. It not multiples of 4, gets cropped down to nearest multiple.
[in]    img:np.ndarry   Image to compress with values in range [0, 1]
[out]   Compressed image stored in S3TC format. Must decode using S3TC_decode
"""
def S3TC_encode(img: np.ndarray):
    shape = img.shape
    new_h, new_w = shape[0] // 4, shape[1] // 4
    res = np.zeros((new_h, new_w * 2)).astype(np.uint32)
    for r in range(new_h):
        for c in range(new_w):
            (color, indices) = S3TC_encode_square(img, r * 4, c * 4)
            res[r, 2 * c] = color
            res[r, 2 * c + 1] = indices
    return res

def S3TC_save_encoded(img: np.ndarray, path: str):
    img = Image.fromarray(img, mode="I")
    img.save(path)
    
""" Decompress an S3TC encoded image into a normal image with values in [0, 1]
[in]    img:np.ndarry   S3TC mage to decompress
[out]   Decompressed image
"""
def S3TC_decode(img: np.ndarray):
    shape = img.shape
    h, w = shape[0], shape[1] // 2
    reconstructed_img = np.zeros((h * 4, w * 4, 3))
    for r in range(h):
        for c in range(w):
            color_encoded, indices_encoded = img[r, 2 * c], img[r, 2 * c + 1]
            reconstructed_img[4*r:4*r+4, 4*c:4*c+4, :] = S3TC_decode_square(color_encoded, indices_encoded)
    return reconstructed_img

###############################################################################
#`                               Main Loop                                    #
###############################################################################

image = "gradient"

img = load_image(f"textures/{image}.png")
img = normalize_image(img)
encoded = S3TC_encode(img)
S3TC_save_encoded(encoded, f"textures/{image}.png")
decoded = S3TC_decode(encoded)
converted = denormalize_image(decoded)
# downsampled = test_bilinear_downsample(img, 1)
# converted = np.round(downsampled).astype(np.uint8)
Image.fromarray(converted).save(f"textures/{image}_compressed.png")