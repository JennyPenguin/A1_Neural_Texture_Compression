import torch

def get_device():
    # for Nvidia GPU
    if torch.cuda.is_available():
        return "cuda"
    # for Apple Silicon
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"