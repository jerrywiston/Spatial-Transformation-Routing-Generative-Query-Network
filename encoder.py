import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderNetwork(nn.Module):
    def __init__(self, ch=64, csize=128, down_size=4):
        super(EncoderNetwork, self).__init__()
        if down_size == 2:
            self.net = nn.Sequential(
                # (ch,64,64)
                Conv2d(3, ch, 3, stride=1),
                nn.LeakyReLU(),
                # (ch,64,64)
                Conv2d(ch, ch, 3, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch),
                # (ch/2,32,32)
                Conv2d(ch, ch*2, 3, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch*2),
                # (ch/2,32,32)
                Conv2d(ch*2, csize, 3, stride=1),
                # (tsize,32,32)
            )
        else:
            self.net = nn.Sequential(
                # (ch,64,64)
                Conv2d(3, ch, 5, stride=2),
                nn.LeakyReLU(),
                # (ch,32,32)
                Conv2d(ch, ch, 5, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch),
                # (ch/2,32,32)
                Conv2d(ch, ch*2, 3, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch*2),
                # (ch/2,16,16)
                Conv2d(ch*2, csize, 3, stride=1),
                # (tsize,16,16)
            )
        
    def forward(self, x):
        out = self.net(x)
        return out
