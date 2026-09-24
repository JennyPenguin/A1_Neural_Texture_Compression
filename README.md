# A1 Neural Texture Compression

All code are in main.py. To ensure all python requirements are correctly installed, run
```
pip install -r requirements.txt
```

## P1. Texture sampler with bilinear interpolation

The bilinear sampling function is 
```python
def sample(u: float, v : float, img: np.ndarray)
```
We are using top left corner as u, v coordinate (0,0). img is the image to interpolate on and can have any number of channels and value ranges. The returned color format will be the same as given.

