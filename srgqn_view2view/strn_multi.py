import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STRN(nn.Module):
    def __init__(self, n_view_cells, vsize=7, emb_size=32, csize=128):
        super(STRN, self).__init__()
        self.n_view_cells = n_view_cells
        self.emb_size = emb_size
        self.csize = csize

        # View Space Embedding Network
        self.emb_net = nn.Sequential(
            nn.Linear(2+7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_size)
        )

    def get_view_code(self, view_size=(16,16)):
        x = torch.linspace(-1, 1, view_size[0])
        y = torch.linspace(-1, 1, view_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        return vcode

    def transform(self, v, vq, view_size=(16,16)):
        # Camera Space Embedding
        vcode_obs = self.get_view_code(), 0).repeat(v.shape[0], 1, 1).reshape(16*16,2)
        vcode_obs = torch.cat([vcode_obs, v.reshape(-1,)], )
        embedding_obs = self.emb_net(vcode_obs) #(256, 128)
        embedding_obs = torch.unsqueeze(query_embedding, 0).repeat(vq.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # View Space Embedding
        vcode_query = self.get_view_code()
        embedding_query = self.emb_net(vcode_query) #(256, 128)
        embedding_query = torch.unsqueeze(query_embedding, 0).repeat(vq.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # Cell Distribution
        relation = torch.bmm(cs_embedding, vs_embedding.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        distribution = torch.softmax(relation, 2)
        return distribution

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