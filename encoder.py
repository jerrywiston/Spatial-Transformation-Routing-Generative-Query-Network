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

class EncoderNetwork2(nn.Module):
    def __init__(self, ch=64, csize=128, down_size=4):
        super(EncoderNetwork2, self).__init__()
        self.conv1 = Conv2d(3, ch, 3, stride=2)
        self.conv2 = Conv2d(ch, ch, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = Conv2d(ch, ch, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(ch)
        # Local Path
        self.conv4 = Conv2d(ch, ch*2, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(ch*2)
        self.conv5 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn5 = nn.BatchNorm2d(ch*2)
        self.conv6 = Conv2d(ch*2, csize, 1, stride=1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), inplace=True)
        out = self.bn2(F.leaky_relu(self.conv2(out), inplace=True)) + out
        out = self.bn3(F.leaky_relu(self.conv3(out), inplace=True))
        # Local Path
        out = self.bn4(F.leaky_relu(self.conv4(out), inplace=True))
        out = self.bn5(F.leaky_relu(self.conv5(out), inplace=True)) + out
        out = self.conv6(out)
        return out