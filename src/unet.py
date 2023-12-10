import torch.nn as nn
import torch


class Unet(nn.Module):
    def __init__(self, imageDim=3, outputDim=3):
        super(Unet, self).__init__()
        
        self.enc1 = self.conv_block(imageDim, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512, use_pooling=False)  # No pooling for the last encoder block

        
        self.dec1 = self.deconv_block(512+256, 256)
        self.dec2 = self.deconv_block(256+128, 128)
        self.dec3 = self.deconv_block(128+64, 64)
        
        self.final = nn.Conv2d(64, outputDim, kernel_size=1)
    
    def conv_block(self, inputDim, outputDim, use_pooling=True):
        layers = [
            nn.Conv2d(inputDim, outputDim, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputDim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outputDim, outputDim, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputDim),
            nn.LeakyReLU(inplace=True),
        ]
        if use_pooling:
            layers.append(nn.MaxPool2d(2, 2))
        x = nn.Sequential(*layers)
        return x

    
    def deconv_block(self, inputDim, outputDim):
        x = nn.Sequential(
            nn.ConvTranspose2d(inputDim, outputDim, kernel_size=2, stride=2),
            nn.BatchNorm2d(outputDim),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputDim, outputDim, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputDim),
            nn.ReLU(inplace=True)
        )
        return x
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Using concatenation for skip connections
        #print(enc4.size(), enc3.size())
        dec1 = self.dec1(torch.cat([enc4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec1, enc2], dim=1))
        dec3 = self.dec3(torch.cat([dec2, enc1], dim=1))
        
        final = self.final(dec3)
        return final
