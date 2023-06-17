
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

# seeding for reproducibility
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)

# Hyperparameters
batch_size = 128
epochs = 10
lr = 0.0002
nchannels = 1
# latent vector length
zsize = 100
# number of generator and discriminator features
ngf = 64
ndf = 64
# Beta1 hyperparameter for Adam optimizer
beta1 = 0.5

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
            in_channels=nchannels,
            out_channels=ndf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False),
            nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True)
            )

    
    def forward(self, x):
        return self.main(x)