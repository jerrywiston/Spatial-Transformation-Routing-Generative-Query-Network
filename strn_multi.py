import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STRN(nn.Module):
    def __init__(self, n_view_cells, n_wrd_cells, vsize=7, wcode_size=3, emb_size=32, n_occ_layers=4, csize=128):
        super(STRN, self).__init__()
        self.n_view_cells = n_view_cells
        self.n_wrd_cells = n_wrd_cells
        self.wcode_size = wcode_size
        self.emb_size = emb_size
        self.n_occ_layers = n_occ_layers
        self.csize = csize

        # Location Code Extracting for World Cells
        self.w2c = nn.Sequential(
            nn.Linear(vsize, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, wcode_size*n_wrd_cells),
        )
        
        # Camera Space Embedding / Frustum Activation / Occlusion
        self.fc1 = nn.Linear(wcode_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_act = nn.Linear(256, 1)
        self.fc_occ = nn.Linear(256, n_occ_layers)
        self.fc_emb = nn.Linear(256, emb_size)

        # View Space Embedding Network
        self.vse = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_size)
        )

        # Mask Convolution
        #self.conv_mask = nn.Sequential(
        #    Conv2d(self.csize, 32, 1, stride=1)
        #    nn.ReLU(inplace=True),
        #    Conv2d(32, 1, 1, stride=1)
        #)

    def transform(self, v, inv=False, view_size=(16,16)):
        # Camera Space Embedding
        wcode = self.w2c(v).view(-1, self.wcode_size)
        h = F.relu(self.fc1(wcode))
        h = F.relu(self.fc2(h))
        activation = torch.sigmoid(self.fc_act(h).view(-1, self.n_wrd_cells, 1))
        occluding = torch.softmax(self.fc_occ(h).view(-1, self.n_wrd_cells, self.n_occ_layers), 2)
        cs_embedding = self.fc_emb(h).view(-1, self.n_wrd_cells, self.emb_size)
        
        # View Space Embedding
        x = torch.linspace(-1, 1, view_size[0])
        y = torch.linspace(-1, 1, view_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        vs_embedding = self.vse(vcode) #(256, 128)
        vs_embedding = torch.unsqueeze(vs_embedding, 0).repeat(v.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # Cell Distribution
        relation = torch.bmm(cs_embedding, vs_embedding.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        if not inv:
            distribution = torch.softmax(relation, 2)
        else:
            distribution = torch.softmax(relation, 1)
        return distribution, activation, occluding

    def forward(self, view_cell, v, view_size=(16,16)):
        distribution, activation, occluding = self.transform(v, view_size=view_size)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))
        return wrd_cell # (-1, csize, n_wrd_cells)
    
    def query(self, wrd_cell, v, view_size=(16,16), steps=None, occlusion=True):
        distribution, activation, occluding = self.transform(v, inv=True, view_size=view_size)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        occ_mask = torch.split(occluding, 1, dim=2)
        
        if occlusion:
            # Initialize Canvas
            mask_wrd_cell = wrd_cell * occ_mask[0].permute(0,2,1).repeat(1,self.csize,1)
            query_view_cell = torch.bmm(mask_wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
            if steps is None:
                steps = self.n_occ_layers
            # Start Rendering
            for i in range(1,steps):
                mask_wrd_cell = wrd_cell * occ_mask[i].permute(0,2,1).repeat(1,self.csize,1)
                draw = torch.bmm(mask_wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
                mask, _ = torch.max(draw, dim=1, keepdim=True)
                #mask = torch.sigmoid(self.conv_mask(draw))
                query_view_cell = draw*mask + query_view_cell*(1-mask)
        else:
            query_view_cell = torch.bmm(wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
        
        return query_view_cell