
# Black History Footage Colorization Project

## Overview
This project is focused on the colorization and enhancement of historical black and white footage, particularly emphasizing significant events and figures in Black history. The primary goal is to vividly bring these historical moments to life through color, while simultaneously denoising the footage for improved clarity and visual quality.

## Methodology
The approach involves taking black and white footage, which has been deliberately made noisy, and employing an AI model to perform both denoising and colorization. This project leverages a blend of computer vision and deep learning techniques for effective restoration and colorization.

## Dependencies
- Python 3.x
- OpenCV
- Torch
- torchvision
- PIL
- Numpy

## Installation
To set up the project, ensure that you have Python 3.x installed, then install the required dependencies:

```bash
pip install opencv-python torch torchvision Pillow numpy
```

## Usage
The project uses a pre-trained U-Net model for image processing. To run the colorization and denoising on your footage, follow these steps:

1. Place your black and white, noisy video footage in a designated directory.
2. Adjust the `video_path` variable in the script to point to your video file.
3. Run the script to process the video. The output will be a colorized, denoised version of the original footage.

### Inference Code
The main functionality of the project is encapsulated in the provided Python script. The script uses OpenCV for video processing, Torch and torchvision for model handling and transformations, and PIL for image manipulation.

```python
import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from unet import Unet
import numpy as np

# Model and transforms setup code ...

while(video.isOpened()):
    # Video processing loop ...

# Release resources code ...
```

## Notes
- The script provided in this README is a simplified representation of the project's code. It may require adjustments to fit specific use cases or to accommodate different video formats and quality.
- The effectiveness of the colorization and denoising might vary depending on the quality and characteristics of the input footage.

## License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).
