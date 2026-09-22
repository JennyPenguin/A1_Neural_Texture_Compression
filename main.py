import numpy as np
import numpy.typing as npt
import math
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

# To create training PSNR change plot
import matplotlib.pyplot as plt

###############################################################################
#`                              Image Helpers                                 #
###############################################################################
def get_device():
    # for Nvidia GPU
    if torch.cuda.is_available():
        return "cuda"
    # for Apple Silicon
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

""" Load image with values all in [0, 255] range.
[in]    path:str    Path name of image to load   
[out]   Loaded image, each pixel is stored as a byte (0-255)
"""
def load_image(path: str, rgba = False):
    img = Image.open(path)
    if rgba:
        img = img.convert('RGBA')
    else:
        img = img.convert("RGB")
    return np.asarray(img)

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
#`                           Calculate PSNR                                   #
###############################################################################

# Assumes both images are normalized images and have the same dimension
def calc_PSNR(img: np.ndarray, img2: np.ndarray):
    MSE = 0
    img = img.reshape(-1, 3)
    img2 = img2.reshape(-1, 3)
    N = img.shape[0]
    for i in range(N):
        diff = img[i] - img2[i]
        MSE += np.dot(diff.T, diff)
    MSE = MSE * (1.0 / N)
    return -10.0 * np.log10(MSE)

###############################################################################
#`                              S3TC Compression                              #
###############################################################################

# Returns C0, C1
def S3TC_pick_min_max(square: np.ndarray):
    C0 = np.min(square, axis = (0, 1))
    C1 = np.max(square, axis = (0, 1))
    return (C0, C1)

# Returns C0, C1
def S3TC_pick_longest_dist(square: np.ndarray):
    best_diff = float('-inf')
    best_C0, best_C1 = None, None
    square_flat = square.reshape(-1, 3)
    size = square_flat.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            C0 = square_flat[i]
            C1 = square_flat[j]
            diff = C1-C0
            norm = np.dot(diff.T, diff)
            if norm > best_diff:
                best_diff = norm
                best_C0 = C0
                best_C1 = C1
    return (best_C0, best_C1)

# pass in 4x4 block (r, c is top left corner of square)
def S3TC_encode_square(img, r, c, pick_min_max = False):
    square = img[r:r+4, c:c+4]
    C0, C1 = S3TC_pick_min_max(square) if pick_min_max else S3TC_pick_longest_dist(square)

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
            indices |= best_index << (2 * cnt)
            cnt += 1

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

def split_32_bits(org: np.uint32):
    mask8 = 0xFF
    res = []
    for _ in range(4):
        res.append(org & mask8)
        org = org >> 8
    return np.array(res).astype(np.uint8)

def combine_32_bits(org: np.ndarray):
    # size of ndarray should be 4
    res = np.uint32(0)
    for i in range(4):
        res |= np.uint32(org[i]) << (i * 8)
    return np.array(res)

""" Compress a normal image with values in range [0, 1] and only RGB channels
Only works on images that have dimensions which are multiples of 4. It not multiples of 4, gets cropped down to nearest multiple.
[in]    img:np.ndarry   Image to compress with values in range [0, 1]
[out]   Compressed image stored in S3TC format. Must decode using S3TC_decode
"""
def S3TC_encode(img: np.ndarray, pick_min_max = False):
    shape = img.shape
    new_h, new_w = shape[0] // 4, shape[1] // 4
    res = np.zeros((new_h, new_w * 2, 4)).astype(np.uint8)
    for r in range(new_h):
        for c in range(new_w):
            (color, indices) = S3TC_encode_square(img, r * 4, c * 4, pick_min_max)
            res[r, 2 * c] = split_32_bits(color)
            res[r, 2 * c + 1] = split_32_bits(indices)
    return res

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
            color_encoded = combine_32_bits(img[r, 2 * c])
            indices_encoded = combine_32_bits(img[r, 2 * c + 1])
            reconstructed_img[4*r:4*r+4, 4*c:4*c+4, :] = S3TC_decode_square(color_encoded, indices_encoded)
    return reconstructed_img

###############################################################################
#`                          Neural                                     #
###############################################################################

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()

        # one learnable grid per resolution, each shaped (1, feat_dim, R, R)
        self.grids = nn.ParameterList()
        for res in resolutions:
            # (batch size = 1 bc cannot process different res image dimesnion together, # channels = # features, height, width)
            res_grid = torch.randn(1, feat_dim, res, res, device=get_device()) * 0.01
            self.grids.append(res_grid)
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):   # uv: (N, 2) in [0, 1], N = batch size
        # bilinear-sample each grid at uv (see F.grid_sample, which wants
        # coords in [-1, 1]) and concatenate the features across resolutions

        # Transform to [-1, 1]
        uv = uv * 2.0 - 1.0
        N = uv.shape[0]
        batch_uv = uv.reshape(1, N, 1, 2)

        res = torch.tensor([]).to(get_device())
        for grid in self.grids:
            res_feature = F.grid_sample(grid, batch_uv ,mode='bilinear', padding_mode="border", align_corners=False)
            # Output dimensin : (1, F, N, 1)
            res_feature = (res_feature.squeeze()).T
            res = torch.cat([res, res_feature], dim=-1)
        return res # (N, out_dim)

class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 2 hidden layers of width 64 (Linear + ReLU), then Linear -> 3 and a Sigmoid (RGB in [0, 1])

        # Hidden layer 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class NeuralTexture(nn.Module):
    def __init__(self, resolutions, feature_dim):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feature_dim)
        self.mlp  = ColorMLP(self.grid.out_dim)

    def forward(self, uv):
        return self.mlp(self.grid(uv))

# Because texel centered coordinates, can just directly take color from image
# The output tensors lives on the GPU!
def build_input_ouput_pairs(img):
    h, w, _ = img.shape
    N = w * h
    coords = np.zeros((N, 2))
    target = np.zeros((N, 3))
    for r in range(h):
        for c in range(w):
            # want coordinate to be u, v
            coords[r * w + c] = [(c + 0.5) / w, (r + 0.5) / h]
            target[r * w + c] = img[r, c, :]
    return (torch.from_numpy(coords).float().to(get_device()), torch.from_numpy(target).float().to(get_device()))

BATCH_SIZE = 16384

# Input image should be normalized
def train_model(img: np.ndarray, coords: np.ndarray, target: np.ndarray, image: str, runs, color="blue"):
    for size, resolutions, feature_dim, line_style in runs:
        model = NeuralTexture(resolutions, feature_dim).to(get_device())
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
        # Size of image
        N = coords.shape[0]
        possible_indices = np.arange(N)
    
        criterion_MSE = nn.MSELoss()

        x = np.arange(2000)
        y = np.zeros((2000,))
    
        for i in range(2000):
            # Sample a minibatch of coords. Normally would cycle through a 
            # random permutation of train samples but writeup says to just take
            # a random sample.
            mini_batch_indices = np.random.choice(possible_indices, size=(BATCH_SIZE,), replace=(N < BATCH_SIZE))
            mini_batch_indices = torch.from_numpy(mini_batch_indices)
    
            batch_coords = coords[mini_batch_indices]
            batch_target = target[mini_batch_indices]
    
            # predict colors, outputs has dimensions (BATCH_SIZE, 2)
            predictions = model.forward(batch_coords)
    
            # MSE loss vs target 
            loss = criterion_MSE(predictions, batch_target)
            PSNR = -10.0 * torch.log10(loss)
            y[i] = PSNR.item()
            opt.zero_grad(); loss.backward(); opt.step()

        plt.plot(x, y, label=f"{image} - {size}", color=color, linestyle=line_style)
    
        # save non-quantized first and then quantized
        for q in range(2):
            if q == 1:
                quantize_model(model, quantize_mlp=False)
                print("!!!!!!!!!!!!!!!!!Quantized Results!!!!!!!!!!!!!!!!!!!!")

            # Get final PSNR
            predictions = model.forward(coords)
            final_loss = criterion_MSE(predictions, target)
        
            # PSNR from the loss:  psnr = -10 * torch.log10(loss)
            PSNR = -10.0 * torch.log10(final_loss)
            print(f"{size}\tgrid{"s" if len(resolutions) > 0 else ""}\t{resolutions}\tfeature_dim {feature_dim}\tMLP 2 x 64\tImage: {image}")
            print(f"PSNR: {PSNR}")

            # Copy to CPU and then convert to numpy
            predicted_colors = predictions.detach().cpu().numpy()
            reconstructed_img = np.zeros(img.shape)
            h, w, _ = img.shape
            for r in range(h):
                for c in range(w):
                    reconstructed_img[r, c, :] = predicted_colors[r * w + c]
            converted = denormalize_image(reconstructed_img)
            q_trail = "_q" if q == 1 else ""
            Image.fromarray(converted).save(f"textures/neural_compressed/{image}_NN_compressed_{size}{q_trail}.png")

###############################################################################
#                           Quantization                                      #
###############################################################################

def quantize_uint8(x):
    lo, hi = torch.min(x), torch.max(x)              # x = one array of float32 values
    scale  = (hi - lo) / 255             # 256 levels (8 bits)
    q = 0
    if scale > 0:
        q = torch.round((x - lo) / scale)     # integer index in [0, 255]
    x_hat  = lo + q * scale              # dequantized value used at decode
    return q, lo, scale, x_hat

def quantize_model(model, quantize_mlp=False):
    # each feature-grid level: its own lo/scale, stored as 8-bit q
    for grid in model.grid.grids:
        q, lo, scale, x_hat = quantize_uint8(grid.data)
        grid.data.copy_(x_hat)

    # the MLP is tiny -- quantizing it is optional
    if quantize_mlp:
        for p in model.mlp.parameters():
            q, lo, scale, x_hat = quantize_uint8(p.data)
            p.data.copy_(x_hat)

###############################################################################
#                        Neural Main Loop                                    #
###############################################################################

RUN_NEURAL = True

runs = [
    ("Small", (64,), 2, "-"),
    ("Medium", (16, 32, 64), 2, "--"),
    ("Large", (16, 32, 64, 128), 4, ":")
]
images = [
        # ("gradient", "blue"),
            ("bricks", "red"), 
        #    ("clouds", "purple")
        ]

if RUN_NEURAL:
    for (image, color) in images:
        img = load_image(f"textures/{image}.png")
        img = normalize_image(img)
        # build coords (N, 2) of texel centers in [0, 1] and target colors (N, # 3). Do this outside of train_model loop so do not have to recompute 
        # every time.
        coords, target = build_input_ouput_pairs(img)

        train_model(img, coords, target, image, runs, color=color)
    plt.title("PSNR over 2000 Training Loops")
    plt.xlabel("Training Step")
    plt.ylabel("PSNR (db)")
    plt.legend()
    plt.savefig('PSNR Over Training.png')
    plt.figure(figsize=(18, 6))

###############################################################################
#                           S3TC main Loop                                    #
###############################################################################

RUN_S3TC = False
if RUN_S3TC:
    images = ["gradient", "bricks", "clouds", "weird_image"]
    for image in images:
        img = load_image(f"textures/{image}.png")
        img = normalize_image(img)
        encoded = S3TC_encode(img, pick_min_max=False)
        (Image.fromarray(encoded, mode='RGBA')).save(f"textures/{image}_encoded.png")
        decoded = S3TC_decode(encoded)
        converted = denormalize_image(decoded)
        Image.fromarray(converted).save(f"textures/{image}_compressed.png")
        PSNR = calc_PSNR(img, decoded)

        print(f"PSNR For image {image}: {PSNR}")

        reload_decoded_img = load_image(f"textures/{image}_encoded.png", rgba=True)
        decoded2 = S3TC_decode(reload_decoded_img)
        converted2 = denormalize_image(decoded2)
        Image.fromarray(converted2).save(f"textures/{image}_compressed2.png")

        # Testing minmax instead
        encoded = S3TC_encode(img, pick_min_max=True)
        decoded = S3TC_decode(encoded)
        converted = denormalize_image(decoded)
        Image.fromarray(converted).save(f"textures/{image}_compressed_min_max.png")

###############################################################################
#                        Bilinear Main Loop                                   #
###############################################################################

RUN_DOWNSAMPLE = False
DOWNSAMPLE_RATIO = 16

if RUN_DOWNSAMPLE:
    image = "time_spiral"
    img = load_image(f"textures/{image}.png")
    downsampled = test_bilinear_downsample(img, DOWNSAMPLE_RATIO)
    converted = np.round(downsampled).astype(np.uint8)
    Image.fromarray(converted).save(f"textures/{image}_downsampled_{DOWNSAMPLE_RATIO}.png")



