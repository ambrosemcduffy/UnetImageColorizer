import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import models

class CustomDataset(Dataset):
    def __init__(self, root_dir, input_transform=None, target_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        #self.transform = transform
        self.input_dir = os.path.join(root_dir, "blackAndWhite")
        self.target_dir = os.path.join(root_dir, "colorImages")
        self.input_filenames = os.listdir(self.input_dir)
    
    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, idx):
        input_filename = self.input_filenames[idx]
        input_path = os.path.join(self.input_dir, input_filename)
        target_filename = input_filename
        target_path = os.path.join(self.target_dir, target_filename)
        
        input_image = Image.open(input_path).convert("L")
        target_image = Image.open(target_path).convert("RGB")
        
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)
        return input_image, target_image
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        
        resnet = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-5])
        self.conv1 = nn.Conv2d(1, 128, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 8, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(512, 3, (3, 3), padding=1)
        #self.conv6 = nn.Conv2d(128, 3, (3, 3), padding=1)
        #self.t_conv1 = nn.ConvTranspose2d(4, 8, (2, 2), stride=2)
        #self.t_conv2 = nn.ConvTranspose2d(8, 16, (2, 2), stride=2, output_padding=1)
        #self.t_conv3 = nn.ConvTranspose2d(16, 3, (2, 2), stride=2, padding=(1, 2), output_padding=(1, 0))

    def forward(self, x):
        x = self.resnet(x)
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv5(x)
        #print(x.size())
        x = F.interpolate(x, size=(288, 512), mode='bilinear')

        #print(x.size())
        #x = torch.tanh(self.conv6(x))
        
        #x = F.relu(self.t_conv1(x))
        #x = F.relu(self.t_conv2(x))
        #x = F.sigmoid(self.t_conv3(x))
        #x = F.interpolate(x, size=(288, 512))  # resize the output
        return x

