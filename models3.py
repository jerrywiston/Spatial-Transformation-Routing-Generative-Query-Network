import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STRN(nn.Module):
    def __init__(self, n_src_cells, n_tgt_cells, vsize, ch=16, emb_size=32, n_occ_layers=4, tsize=128):
        super(STRN, self).__init__()
        self.ch = ch
        self.emb_size = emb_size
        self.n_src_cells = n_src_cells
        self.n_tgt_cells = n_tgt_cells
        self.n_occ_layers = n_occ_layers
        self.tsize = tsize
        self.fc1 = nn.Linear(vsize, 512)
        self.fc2 = nn.Linear(512, ch*n_tgt_cells)
        self.fc3 = nn.Linear(ch, 128)
        
        self.fc4_act = nn.Linear(128, 1)
        self.fc4_occ = nn.Linear(128, n_occ_layers)
        self.fc4_emb = nn.Linear(128, emb_size)

        self.fc1_view = nn.Linear(2, 128)
        self.fc2_view = nn.Linear(128, emb_size)

    def transform(self, v, inv=False, im_size=(16,16)):
        h1 = F.relu(self.fc1(v))
        h2 = F.relu(self.fc2(h1).view(-1, self.ch))
        h3 = F.relu(self.fc3(h2))
        # Activation
        h4_act = self.fc4_act(h3).view(-1, self.n_tgt_cells, 1)
        activation = torch.sigmoid(h4_act)
        # Occlusion
        h4_occ = self.fc4_occ(h3).view(-1, self.n_tgt_cells, self.n_occ_layers)
        occluding = torch.softmax(h4_occ, 2)
        # Attention
        x = torch.linspace(-1, 1, im_size[0])
        y = torch.linspace(-1, 1, im_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        view_code = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        h1_view = F.relu(self.fc1_view(view_code)) #(256, 128)
        view_emb = torch.unsqueeze(self.fc2_view(h1_view), 0).repeat(v.shape[0], 1, 1) #(-1, view_cell, emb_size)
        wrd_emb = self.fc4_emb(h3).view(-1, self.n_tgt_cells, self.emb_size) #(-1, wrd_cell, emb_size)
        relation = torch.bmm(wrd_emb, view_emb.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        if not inv:
            attention = torch.softmax(relation, 2)
        else:
            attention = torch.softmax(relation, 1)
        return attention, activation, occluding

    def forward(self, view_cell, v):
        attention, activation, occluding = self.transform(v)
        route = attention * activation   # (-1, n_tgt_cells, n_src_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))
        return wrd_cell # (-1, ch, n_tgt_cells)
    
    def query(self, wrd_cell, v, canvas_shape=(16,16), steps=None):
        attention, activation, occluding = self.transform(v, True)
        route = attention * activation   # (-1, n_tgt_cells, n_src_cells)
        occ_mask = torch.split(occluding, 1, dim=2)
        
        # Initialize Canvas
        query_view_cell = torch.zeros(wrd_cell.shape[0], self.tsize, canvas_shape[0], canvas_shape[1]).to(device)
        if steps is None:
            steps = self.n_occ_layers
        # Start Rendering
        for i in range(steps):
            mask_wrd_cell = wrd_cell * occ_mask[i].permute(0,2,1).repeat(1,self.tsize,1)
            draw = torch.bmm(wrd_cell, route).reshape(-1, self.tsize, canvas_shape[0], canvas_shape[1])
            mask, _ = torch.max(draw, dim=1, keepdim=True)
            query_view_cell = draw*mask + query_view_cell*(1-mask)
        
        return query_view_cell