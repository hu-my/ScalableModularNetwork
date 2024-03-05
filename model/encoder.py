import torch
import torch.nn as nn
import torch.nn.functional as F

class OneConvEncoder(nn.Module):
    def __init__(self):
        super(OneConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU()
        )
        self.out_channels = 6

    def forward(self, x):
        return self.encoder(x) # (b, 6, 10, 10)

class TwoConvEncoder(nn.Module):
    def __init__(self):
        super(TwoConvEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # (b, 32, 24, 24)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
        ) # (b, 16, 8, 8)
        self.out_channels = 64

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out