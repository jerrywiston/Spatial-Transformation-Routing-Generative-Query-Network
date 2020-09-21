import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        h_conv0 = self.conv0(x)
        if self.bn:
            h_conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h_conv2 = self.bn2(self.conv2(h_conv1))
        else:
            h_conv1 = F.relu(self.conv1(x), inplace=True)
            h_conv2 = self.conv2(h_conv1)
        out = F.relu(h_conv0 + h_conv2, inplace=True)
        return out

class EncoderRes(nn.Module):
    def __init__(self, ch=64, tsize=128):
        super(EncoderRes, self).__init__()
        self.res1 = ResBlock(3, ch, bn=True)
        self.pool1 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        self.res2 = ResBlock(ch, ch*2, bn=True)
        self.pool2 = BlurPool2d(filt_size=3, channels=ch*2, stride=2)
        self.res3 = ResBlock(ch*2, tsize, bn=True)

    def forward(self, x):
        h_res1 = self.res1(x)
        h_pool1 = self.pool1(h_res1)
        h_res2 = self.res2(h_pool1)
        h_pool2 = self.pool2(h_res2)
        h_res3 = self.res3(h_pool2)
        out = torch.tanh(h_res3)
        return out

class DecodeRes(nn.Module):
    def __init__(self, ch=64, tsize=128):
        super(DecoderRes, self).__init__()
        self.res1 = ResBlock(tsize,ch*2, bn=True)
        # (8,8,512) -> (16,16,256)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res2 = ResBlock(ch*2,ch, bn=True)
        # (16,16,256) -> (32,32,256)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res3 = ResBlock(ch,3, bn=True)

    def forward(self, h):
        h = torch.softmax(h, 1)
        h_res1 = self.res1(h)
        h_up2 = self.up2(h_res1)
        h_res2 = self.res2(h_up2)
        h_up3 = self.up3(h_res2)
        h_res3 = self.res3(h_up3)
        out = torch.sigmoid(h_res3)
        return out

class Encoder(nn.Module):
    def __init__(self, ch=64, tsize=128):
        super(Encoder, self).__init__()
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
            Conv2d(ch*2, tsize, 3, stride=1),
            # (tsize,16,16)
        )
        
    def forward(self, x):
        out = self.net(x)
        out = torch.tanh(out)
        return out

class WorldTransform(nn.Module):
    def __init__(self, n_src_cells, n_tgt_cells, vsize, ch=16):
        super(WorldTransform, self).__init__()
        self.ch = ch
        self.n_src_cells = n_src_cells
        self.n_tgt_cells = n_tgt_cells
        self.fc1 = nn.Linear(vsize, 512)
        self.fc2 = nn.Linear(512, ch*n_tgt_cells)
        self.fc3 = nn.Linear(ch, int(n_src_cells/2))
        self.fc4_att = nn.Linear(int(n_src_cells/2), n_src_cells)
        self.fc4_act = nn.Linear(int(n_src_cells/2), 1)
        self.dropout = nn.Dropout(0.4, inplace=True)
      
    def transform(self, v):
        h1 = F.relu(self.fc1(v))
        h2 = F.relu(self.fc2(h1).view(-1, self.ch))
        h3 = F.relu(self.fc3(h2))
        h4_att = self.fc4_att(h3).view(-1, self.n_tgt_cells, self.n_src_cells)
        h4_act = self.fc4_act(h3).view(-1, self.n_tgt_cells, 1)
        attention = torch.softmax(h4_att, 2)
        activation = torch.sigmoid(h4_act)
        return attention, activation

    def forward(self, src_cell, v, drop=False):
        attention, activation = self.transform(v)
        route = attention * activation   # (-1, n_tgt_cells, n_src_cells)
        tgt_cell = torch.bmm(src_cell, route.permute(0,2,1))
        if drop:
            tgt_cell = self.dropout(tgt_cell)
        return tgt_cell # (-1, ch, n_tgt_cells)
        
class ProjTransform(nn.Module):
    def __init__(self, n_src_cells, n_tgt_cells):
        super(ProjTransform, self).__init__()
        self.n_src_cells = n_src_cells
        self.n_tgt_cells = n_tgt_cells
        self.proj = nn.Linear(n_src_cells, n_tgt_cells, bias=False)

    def forward(self, src_cell):
        tsize = src_cell.shape[1]
        tgt_cell = self.proj(src_cell.view(-1, self.n_src_cells)).view(-1, tsize, self.n_tgt_cells)
        return tgt_cell

class Renderer(nn.Module):
    def __init__(self, n_cam_cells, canvas_shape, tsize=64, n_steps=6):
        super(Renderer, self).__init__()
        self.n_cam_cells = n_cam_cells
        self.canvas_shape = canvas_shape
        self.tsize = tsize
        self.n_steps = n_steps
        proj_layers = []
        for i in range(n_steps):
            proj_layers.append(ProjTransform(n_cam_cells, canvas_shape[0]*canvas_shape[1]))
        self.proj_layers = nn.ModuleList(proj_layers)
        self.mask_conv = Conv2d(tsize*2, 1, 3, stride=1)
        
    def forward(self, cam_cell):
        canvas = torch.zeros(cam_cell.shape[0], self.tsize, self.canvas_shape[0], self.canvas_shape[1]).to(device)
        for i in range(self.n_steps):
            draw = self.proj_layers[i](cam_cell).view(-1, self.tsize, self.canvas_shape[0], self.canvas_shape[1])
            mask = torch.sigmoid(self.mask_conv(torch.cat((draw, canvas), 1)))
            canvas = draw*mask + canvas*(1-mask)
        #canvas = torch.softmax(canvas, 1)
        return canvas

class Renderer2(nn.Module):
    def __init__(self, n_cam_cells, canvas_shape, tsize=64, n_steps=6):
        super(Renderer2, self).__init__()
        self.n_cam_cells = n_cam_cells
        self.canvas_shape = canvas_shape
        self.tsize = tsize
        self.n_steps = n_steps
        self.cell_masks  = torch.nn.Parameter(torch.randn(1,n_steps,n_cam_cells), requires_grad=True)
        #self.proj = torch.nn.Parameter(torch.randn(1, n_cam_cells, 16*16)*2-1, requires_grad=True)
        self.proj = ProjTransform(n_cam_cells, 16*16)
        self.mask_conv = Conv2d(tsize*2, 1, 3, stride=1)
        
    def forward(self, cam_cell):
        canvas = torch.zeros(cam_cell.shape[0], self.tsize, self.canvas_shape[0], self.canvas_shape[1]).to(device)
        cell_masks_split = torch.sigmoid(self.cell_masks)
        cell_masks_split = torch.split(cell_masks_split, 1, dim=1)
        for i in range(self.n_steps):
            cmask = cell_masks_split[i]
            cam_cell_mask = cmask.repeat(cam_cell.shape[0],self.tsize,1) * cam_cell
            #proj_dist = torch.softmax(self.proj, 1)
            #draw = torch.bmm(cam_cell_mask, proj_dist.repeat(cam_cell.shape[0],1,1)).view(-1,self.tsize,16,16)
            draw = self.proj(cam_cell_mask).view(-1,self.tsize,16,16)           
            mask = torch.sigmoid(self.mask_conv(torch.cat((draw, canvas), 1)))
            canvas = draw*mask + canvas*(1-mask)
        return canvas

class Decoder(nn.Module):
    def __init__(self, ch=64, tsize=64):
        super(Decoder, self).__init__()
        self.ch = ch
        self.tsize = tsize
        self.net = nn.Sequential(
            # (ch,16,16)
            Conv2d(tsize, ch*2, 3, stride=1),
            nn.LeakyReLU(),
            # (ch,16,16)
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d(ch*2, ch, 3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(ch),
            # (ch/2,32,32)
            Conv2d(ch, ch, 3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(ch),
            # (ch/2,32,32)
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d(ch, 3, 3, stride=1),
            # (3,64,64)
            nn.Sigmoid(),
        )

    def forward(self, h):
        h = torch.softmax(h, 1)
        x_re = self.net(h)
        return x_re