import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
#from model import AutoEncoder
from PIL import Image
from unet import Unet
import numpy as np

model = Unet(3, 3)
model = model.cuda()
model.load_state_dict(torch.load("weights/modelUnet_v2.pth"))
model.eval()

transforms = Compose([
    Resize((288, 512)),
    ToTensor(),
    Normalize([0.5], [0.5])
])

# Open video file
video_path = "/mnt/e/autoEncoder/thisHere8.mp4"
video = cv2.VideoCapture(video_path)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'x264' might also work
out = cv2.VideoWriter('/mnt/e/autoEncoder/testVidUpdated5.mp4', fourcc, 20.0, (512, 288))  # output path, codec, fps, size

def denormalize(tensor):
    tensor = tensor * torch.Tensor([0.5, 0.5, 0.5]).cuda() + torch.Tensor([0.5, 0.5, 0.5]).cuda()  # Denormalizing 
    tensor = tensor.clamp(0, 1)  # Clipping pixel values in the range [0, 1]
    return tensor

from torchvision.transforms import Grayscale

transforms = Compose([
    Resize((288, 512)),
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize([0.5], [0.5])
])

while(video.isOpened()):
    ret, frame = video.read()

    if not ret:
        break

    # Convert frame to PIL image and apply transforms
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_transformed = transforms(frame).unsqueeze(0).cuda()

    # Feed the frame through the model
    output = model(frame_transformed)

    # Postprocess output
    output = output.squeeze().permute(1, 2, 0)
    output = denormalize(output)
    output = output.cpu().detach().numpy()
    output = np.clip(output * 255, 0, 255).astype('uint8')  # convert float image to uint8
    print(output.shape)

    # Convert RGB to BGR (for opencv) and write output frame
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    out.write(output_bgr)

# Release everything when job is finished
video.release()
out.release()
