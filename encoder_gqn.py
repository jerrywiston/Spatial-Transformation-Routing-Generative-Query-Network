import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tower(nn.Module):
    def __init__(self, vsize=7, csize=256):
        super(Tower, self).__init__()
        self.vsize = vsize
        self.csize = csize
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+vsize, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+vsize, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), self.vsize, 1, 1).repeat(1, 1, r.shape[2], r.shape[3])
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        return r

class TowerBN(nn.Module):
    def __init__(self, vsize=7, csize=256):
        super(TowerBN, self).__init__()
        self.vsize = vsize
        self.csize = csize
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256+vsize, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256+vsize, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = self.bn2(F.relu(self.conv2(skip_in)))

        r = self.bn3(F.relu(self.conv3(skip_in)))
        r = self.bn4(F.relu(self.conv4(r))) + skip_out

        # Broadcast
        v = v.view(v.size(0), vsize, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = self.bn5(F.relu(self.conv5(skip_in)))

        r = self.bn6(F.relu(self.conv6(skip_in)))
        r = self.bn7(F.relu(self.conv7(r))) + skip_out
        r = F.relu(self.conv8(r))

        return r