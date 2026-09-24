# A1 Neural Texture Compression

All code are in main.py with the exception of the final evaluation plot which is in evaluation.py. To ensure all python requirements are correctly installed, run
```
pip install -r requirements.txt
```

## P1. Bilinear Interpolation

The bilinear sampling function is 
```python
def sample(u: float, v : float, img: np.ndarray)
```
We are using top left corner as u, v coordinate (0,0). img is the image to interpolate on and can have any number of channels and value ranges. The returned color format will be the same as given.

To run the code with downsampling, in main.py, look for 
```python
###############################################################################
#                        Bilinear Main Loop                                   #
###############################################################################
```
and set
```python
RUN_DOWNSAMPLE = True
```
Pay attention to the folder directory and image used.

## P2. S3TC
To run the S3TC code, in main.py, look for 
```python
###############################################################################
#                           S3TC main Loop                                    #
###############################################################################
```
and set 
```python
RUN_S3TC = True
```
You can change the image being compressed in the images array:
```python
if RUN_S3TC:
    images = ["gradient", "bricks", "clouds", "weird_image"]
```
Again, be aware of the folder directory that the images are saved to.

## P3-8 Neural Texture Compression
To run the neural compression, in main.py, look for
```python
###############################################################################
#                        Neural Main Loop                                     #
###############################################################################
```
and set
```python
RUN_NEURAL = True
```
Running this loop would generate for each image the compressed image based on the 3 architectures and the compressed image after quantizing each feature grid. It also generates the PSNR over training loop plot for all architectures together. The final PSNR value and number of parameters in the model is also printed in the terminal for each model (both original and quantized).

To change the architectures for the images, change
```python
runs = [
    ("Small", (64,), 2, "-"),
    ("Medium", (16, 32, 64), 2, "--"),
    ("Large", (16, 32, 64, 128), 4, ":")
]
```
Note that the architectures is shared across all images.

To change the images, change
```python
images = [
            ("gradient", "blue"),
            ("bricks", "red"), 
            ("clouds", "purple")
         ]
```
The first value in the tuple is the name of the image and the second value is its color in the training loop plot.

## P9 Evaluation
The evaluation is generated in evaluation.py. You can directly run it and it will produce both evaluation plots (one based on raw bytes and one based on compression factor).

Unfortunately, all values are hardcoded so there is no automated pipeline here to update the plots if you changed the images and architectures in the previous parts.