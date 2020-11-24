import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super(Discriminator, self).__init__()
        self.ch = ch
        # Stem (64)
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        # ResBLock 1 (32)
        self.conv2 = nn.utils.spectral_norm(Conv2d(ch, ch, 3, stride=1))
        self.conv3 = nn.utils.spectral_norm(Conv2d(ch, ch, 3, stride=1))
        # Pool + Feature Dim
        self.pool3 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        self.conv4 = Conv2d(ch, ch*2, 1, stride=1)
        # ResBlock 2 (16)
        self.conv5 = nn.utils.spectral_norm(Conv2d(ch*2, ch*2, 3, stride=1))
        self.conv6 = nn.utils.spectral_norm(Conv2d(ch*2, ch*2, 3, stride=1))
        # Pool + Feature Dim
        self.pool6 = BlurPool2d(filt_size=3, channels=ch*2, stride=2)
        self.conv7 = Conv2d(ch*2, ch*4, 1, stride=1)
        # ResBlock 3 (8)
        self.conv8 = nn.utils.spectral_norm(Conv2d(ch*4, ch*4, 3, stride=1))
        self.conv9 = nn.utils.spectral_norm(Conv2d(ch*4, ch*4, 3, stride=1))
        # Pool + Feature Dim
        self.pool9 = BlurPool2d(filt_size=3, channels=ch*4, stride=2)
        self.conv10 = Conv2d(ch*4, 1, 1, stride=1)
    
    def forward(self, x):
        # Stem
        out = F.leaky_relu(self.conv1(x))
        #out = self.pool1(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(out, inplace=True))
        out = self.conv3(F.leaky_relu(hidden, inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool3(out)
        out = self.conv4(out)
        feature_layer = out
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(out, inplace=True))
        out = self.conv6(F.leaky_relu(hidden, inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool6(out)
        out = self.conv7(out)
        # ResBlock 3
        hidden = self.conv8(F.leaky_relu(out, inplace=True))
        out = self.conv9(F.leaky_relu(hidden, inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool9(out)
        out = self.conv10(out)
        return out, feature_layer