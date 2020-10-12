import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import models2
import generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Model Setting
# ==============================
# Image Size: (3, 64, 64)
# Cell Channel: 128
# View Cells: (16, 16)
# Camera Cells: 512
# World Cells: 2048
# ==============================

# Deterministic
class SRGQN(nn.Module):
    def __init__(self, n_wrd_cells=2048, n_ren_cells=512, tsize=128, ch=64, vsize=7):
        super(SRGQN, self).__init__()
        self.n_wrd_cells = n_wrd_cells
        self.n_ren_cells = n_ren_cells
        self.tsize = tsize
        self.vsize = vsize
        self.ch = ch

        self.encoder = models.Encoder(ch, tsize).to(device)
        self.strn = models2.STRN(16*16, n_wrd_cells, vsize=vsize, ch=64).to(device)
        self.generator = models.Decoder(ch, tsize).to(device)

    def step_observation_encode(self, x, v):
        view_cell = self.encoder(x).reshape(-1, self.tsize, 16*16)
        wrd_cell = self.strn(view_cell, v)
        return wrd_cell
    
    def step_scene_fusion(self, wrd_cell, n_obs): # mode=sum/mean
        scene_cell = torch.sum(wrd_cell.view(-1, n_obs, self.tsize, self.n_wrd_cells), 1, keepdim=False)
        scene_cell = torch.sigmoid(scene_cell)
        return scene_cell
    
    def step_query_view(self, scene_cell, vq, steps=None):
        view_cell_query = self.strn.query(scene_cell, vq, steps)
        x_query = self.generator(view_cell_query)
        return x_query
    
    def visualize_routing(self, view_cell, v, vq, steps=None):
        wrd_cell = self.strn(view_cell.reshape(-1, self.tsize, 16*16), v)
        scene_cell = torch.sigmoid(wrd_cell)
        view_cell_query = self.strn.query(scene_cell, vq, steps=steps)
        return view_cell_query

    def forward(self, x, v, vq, n_obs=3, steps=None):
        # Observation Encode
        wrd_cell = self.step_observation_encode(x, v)
        # Scene Fusion
        scene_cell = self.step_scene_fusion(wrd_cell, n_obs)
        # Query Image
        x_query = self.step_query_view(scene_cell, xq, vq, steps=steps)
        return x_query

