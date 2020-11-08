import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Encoder, self).__init__()
        self.ch = ch
        # Stem (64)
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(ch)
        # ResBLock 1 (32)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = Conv2d(ch, ch, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = Conv2d(ch, ch, 3, stride=1)
        # Pool + Feature Dim
        self.pool3 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        self.conv4 = Conv2d(ch, ch*2, 1, stride=1)
        # ResBlock 2 (16)
        self.conv5 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn5 = nn.BatchNorm2d(ch*2)
        self.conv6 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn6 = nn.BatchNorm2d(ch*2)
        # Pool + Feature Dim
        self.pool6 = BlurPool2d(filt_size=3, channels=ch*2, stride=2)
        self.conv7 = Conv2d(ch*2, ch*4, 1, stride=1)
        # ResBlock 3 (8)
        self.conv8 = Conv2d(ch*4, ch*4, 3, stride=1)
        self.bn8 = nn.BatchNorm2d(ch*4)
        self.conv9 = Conv2d(ch*4, ch*4, 3, stride=1)
        self.bn9 = nn.BatchNorm2d(ch*4)
        # Pool + Feature Dim
        self.pool9 = BlurPool2d(filt_size=3, channels=ch*4, stride=2)
        self.conv10 = Conv2d(ch*4, ch*4, 1, stride=1)
        # Mean and Variance
        self.fc_mu = nn.Linear(4*4*ch*4, z_dim)
        self.fc_logvar = nn.Linear(4*4*ch*4, z_dim)
    
    def forward(self, x):
        # Stem
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        #out = self.pool1(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(self.bn2(out), inplace=True))
        out = self.conv3(F.leaky_relu(self.bn3(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool3(out)
        out = self.conv4(out)
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(self.bn5(out), inplace=True))
        out = self.conv6(F.leaky_relu(self.bn6(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool6(out)
        out = self.conv7(out)
        # ResBlock 3
        hidden = self.conv8(F.leaky_relu(self.bn8(out), inplace=True))
        out = self.conv9(F.leaky_relu(self.bn9(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool9(out)
        out = self.conv10(out)
        # (1024, 4, 4)
        self.mu = self.fc_mu(out.view(-1,4*4*self.ch*4))
        self.logvar = self.fc_logvar(out.view(-1,4*4*self.ch*4))
        return self.mu, self.logvar

def sample_z(mu, logvar):
    eps = torch.randn(mu.size()).to(device)
    return mu + torch.exp(logvar / 2) * eps

class Decoder(nn.Module):
    def __init__(self, z_dim, r_dim, ch=128):
        super(Decoder, self).__init__()
        self.conv1 = Conv2d(r_dim + z_dim + 2, ch*2, 3)
        self.bn1 = nn.BatchNorm2d(ch*2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # ResBlock 1
        self.bn2 = nn.BatchNorm2d(ch*2)
        self.conv2 = Conv2d(ch*2, ch*2, 3)
        self.bn3 = nn.BatchNorm2d(ch*2)
        self.conv3 = Conv2d(ch*2, ch*2, 3)
        # Up + Feature Dim
        self.conv4 = Conv2d(ch*2, ch, 1)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        # ResBlock 2
        self.bn5 = nn.BatchNorm2d(ch)
        self.conv5 = Conv2d(ch, ch, 3)
        self.bn6 = nn.BatchNorm2d(ch)
        self.conv6 = Conv2d(ch, ch, 3)
        # Output
        self.bn7 = nn.BatchNorm2d(ch)
        self.conv7 = Conv2d(ch, 3, 5)
        
    def forward(self, z, r, view_size=(16,16)):
        # View Space Embedding
        x = torch.linspace(-1, 1, view_size[1]).to(device)
        y = torch.linspace(-1, 1, view_size[0]).to(device)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = torch.unsqueeze(torch.unsqueeze(x_grid, 0), 0).repeat(z.size(0),1,1,1)
        y_grid = torch.unsqueeze(torch.unsqueeze(y_grid, 0), 0).repeat(z.size(0),1,1,1)
        z_grid = torch.unsqueeze(torch.unsqueeze(z,-1), -1).repeat(1,1,view_size[0],view_size[1])
        z_code = torch.cat((x_grid, y_grid, z_grid, r), dim=1)
        # Up
        out = F.relu(self.bn1(self.conv1(z_code)))
        out = self.up2(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(self.bn2(out), inplace=True))
        out = self.conv3(F.leaky_relu(self.bn3(hidden), inplace=True)) + out
        # Up + Feature Dim
        out = self.conv4(out)
        out = self.up5(out)
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(self.bn5(out), inplace=True))
        out = self.conv6(F.leaky_relu(self.bn6(hidden), inplace=True)) + out
        # Output
        out = self.conv7(self.bn7(out))
        self.x_samp = torch.sigmoid(out)
        return self.x_samp

class GeneratorNetwork(nn.Module):
    def __init__(self, z_dim, r_dim):
        super(GeneratorNetwork, self).__init__()
        self.z_dim = z_dim
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim, r_dim)
    
    def forward(self, x, r):
        mu, logvar = self.enc(x)
        kl = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
        z = sample_z(mu, logvar)
        x_rec = self.dec(z,r)
        return x_rec, kl

    def sample(self, x_shape, r):
        z_rand = torch.randn(r.size(0), self.z_dim, device=device)
        x_samp = self.dec(z_rand,r)
        return x_samp



