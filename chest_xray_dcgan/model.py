
import random

import torch
import torch.nn as nn


# Hyperparameters
batch_size = 128
epochs = 10
lr = 0.0002
nchannels = 1
# latent vector length
zsize = 128
# number of generator and discriminator features
ngf = 64
ndf = 64
# Beta1 hyperparameter for Adam optimizer
beta1 = 0.5

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input shape: nchannels x 64 x 64
            nn.Conv2d(
                in_channels=nchannels,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True),
            # input shape: ndf x 32 x 32
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ndf*2),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True),
            # input shape: ndf*2 x 16 x 16
            nn.Conv2d(
                in_channels=ndf*2,
                out_channels=ndf*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ndf*4),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True),
            # input shape: ndf*4 x 8 x 8
            nn.Conv2d(
                in_channels=ndf*4,
                out_channels=ndf*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ndf*8),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True),
            # input shape: ndf*8 x 4 x 4
            nn.Conv2d(
                in_channels=ndf*8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.main(x)
    
    
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input shape: zsize x 1 x 1
            nn.ConvTranspose2d(
                in_channels=zsize,
                out_channels=ngf*8,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False),
            nn.BatchNorm2d(
                num_features=ngf*8),
            nn.ReLU(True),
            # input shape: ngf*16 x 2 x 2
            nn.ConvTranspose2d(
                in_channels=ngf*8,
                out_channels=ngf*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ngf*4),
            nn.ReLU(True),
            # input shape: ngf*16 x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf*4,
                out_channels=ngf*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ngf*2),
            nn.ReLU(True),
            # input shape: ndf*8 x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf*2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                num_features=ngf),
            nn.ReLU(True),
            # input size: ngf x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nchannels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)