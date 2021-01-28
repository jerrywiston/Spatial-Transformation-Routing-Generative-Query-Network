import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderNetwork(nn.Module):
    def __init__(self, ch=64, csize=128, gsize=128):
        super(EncoderNetwork, self).__init__()
        self.conv1 = Conv2d(3, ch, 3, stride=2)
        self.conv2 = Conv2d(ch, ch, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = Conv2d(ch, ch, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(ch)
        # Global Path
        self.conv4_g = Conv2d(ch+7, ch*2, 3, stride=1)
        self.bn4_g = nn.BatchNorm2d(ch*2)
        self.conv5_g = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn5_g = nn.BatchNorm2d(ch*2)
        self.conv6_g = Conv2d(ch*2, gsize, 1, stride=1)
        # Local Path
        self.conv4_l = Conv2d(ch, ch*2, 3, stride=1)
        self.bn4_l = nn.BatchNorm2d(ch*2)
        self.conv5_l = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn5_l = nn.BatchNorm2d(ch*2)
        self.conv6_l = Conv2d(ch*2, csize, 1, stride=1)

    def forward(self, x, v):
        out = F.leaky_relu(self.conv1(x), inplace=True)
        out = self.bn2(F.leaky_relu(self.conv2(out), inplace=True)) + out
        out = self.bn3(F.leaky_relu(self.conv3(out), inplace=True))
        # Global Path
        pose_code = v.view(v.shape[0], -1, 1, 1)
        pose_code = pose_code.repeat(1, 1, 16, 16)
        out_g = torch.cat((out, pose_code), dim=1)
        out_g = self.bn4_g(F.leaky_relu(self.conv4_g(out_g), inplace=True))
        out_g = self.bn5_g(F.leaky_relu(self.conv5_g(out_g), inplace=True)) + out_g
        out_g = self.conv6_g(out_g)
        out_g = torch.mean(out_g, (2,3), keepdim=True)
        # Local Path
        out_l = self.bn4_l(F.leaky_relu(self.conv4_l(out), inplace=True))
        out_l = self.bn5_l(F.leaky_relu(self.conv5_l(out_l), inplace=True)) + out_l
        out_l = self.conv6_l(out_l)
        return out_l, out_g