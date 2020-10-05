import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STRN(nn.Module):
    def __init__(self, n_src_cells, n_tgt_cells, vsize, ch=16, n_occ_layers=6, tsize=128):
        super(STRN, self).__init__()
        self.ch = ch
        self.n_src_cells = n_src_cells
        self.n_tgt_cells = n_tgt_cells
        self.n_occ_layers = n_occ_layers
        self.tsize = tsize
        self.fc1 = nn.Linear(vsize, 512)
        self.fc2 = nn.Linear(512, ch*n_tgt_cells)
        self.fc3 = nn.Linear(ch, int(n_src_cells/2))
        self.fc4_att = nn.Linear(int(n_src_cells/2), n_src_cells)
        self.fc4_act = nn.Linear(int(n_src_cells/2), 1)
        self.fc4_occ = nn.Linear(int(n_src_cells/2), n_occ_layers)
        self.mask_conv = Conv2d(tsize*2, 1, 3, stride=1)
      
    def transform(self, v, inv=False):
        h1 = F.relu(self.fc1(v))
        h2 = F.relu(self.fc2(h1).view(-1, self.ch))
        h3 = F.relu(self.fc3(h2))
        h4_att = self.fc4_att(h3).view(-1, self.n_tgt_cells, self.n_src_cells)
        h4_act = self.fc4_act(h3).view(-1, self.n_tgt_cells, 1)
        h4_occ = self.fc4_occ(h3).view(-1, self.n_tgt_cells, self.n_occ_layers)
        if not inv:
            attention = torch.softmax(h4_att, 2)
        else:
            attention = torch.softmax(h4_att, 1)
        activation = torch.sigmoid(h4_act)
        occluding = torch.softmax(h4_occ, 2)
        return attention, activation, occluding

    def forward(self, view_cell, v):
        attention, activation, occluding = self.transform(v)
        route = attention * activation   # (-1, n_tgt_cells, n_src_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))
        return wrd_cell # (-1, ch, n_tgt_cells)
    
    def query(self, wrd_cell, v, canvas_shape=(16,16)):
        attention, activation, occluding = self.transform(v, True)
        route = attention * activation   # (-1, n_tgt_cells, n_src_cells)
        occ_mask = torch.split(occluding, 1, dim=2)
        
        # Initialize Canvas
        query_view_cell = torch.zeros(wrd_cell.shape[0], self.tsize, canvas_shape[0], canvas_shape[1]).to(device)
        # Start Rendering
        for i in range(self.n_occ_layers):
            mask_wrd_cell = wrd_cell * occ_mask[i].permute(0,2,1).repeat(1,self.tsize,1)
            draw = torch.bmm(wrd_cell, route).reshape(-1, self.tsize, canvas_shape[0], canvas_shape[1])
            mask = torch.sigmoid(self.mask_conv(torch.cat((query_view_cell, draw), 1)))
            query_view_cell = draw*mask + query_view_cell*(1-mask)
        
        return query_view_cell