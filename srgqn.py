import torch
import torch.nn as nn
import torch.nn.functional as F
import models
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
        self.view2wrd = models.WorldTransform(16*16, n_wrd_cells, vsize=vsize, ch=64).to(device)
        self.wrd2ren = models.WorldTransform(n_wrd_cells, n_ren_cells, vsize=vsize, ch=64).to(device)
        self.renderer = models.Renderer(n_ren_cells, (16,16), tsize, n_steps=6).to(device)
        self.generator = generator.GeneratorNetwork(x_dim=3, r_dim=tsize, L=6).to(device)

    def step_observation_encode(self, x, v):
        view_cell = self.encoder(x).reshape(-1, self.tsize, 16*16)
        wrd_cell = self.view2wrd(view_cell, v, drop=False)
        return wrd_cell
    
    def step_scene_fusion(self, wrd_cell, n_obs, mode="sum"): # mode=sum/mean
        if mode == "sum":
            scene_cell = torch.sum(wrd_cell.view(-1, n_obs, self.tsize, self.n_wrd_cells), 1, keepdim=False)
            scene_cell = torch.sigmoid(scene_cell)
        else:
            scene_cell = torch.mean(wrd_cell.view(-1, n_obs, self.tsize, self.n_wrd_cells), 1, keepdim=False)
            scene_cell = torch.relu(scene_cell)
        return scene_cell
    
    def step_query_view(self, scene_cell, xq, vq):
        ren_cell_query = self.wrd2ren(scene_cell, vq)
        view_cell_query = self.renderer(ren_cell_query)
        x_query, kl = self.generator(xq, view_cell_query)
        return x_query, kl

    def step_query_view_sample(self, scene_cell, vq, steps=None):
        ren_cell_query = self.wrd2ren(scene_cell, vq)
        view_cell_query = self.renderer(ren_cell_query, steps)
        x_query = self.generator.sample((64,64), view_cell_query)
        return x_query

    def visualize_routing(self, view_cell, v, vq):
        view_cell = view_cell.reshape(-1, self.tsize, 16*16)
        wrd_cell = self.view2wrd(view_cell, v, drop=False)
        scene_cell = torch.sigmoid(wrd_cell)
        ren_cell_query = self.wrd2ren(scene_cell, vq)
        view_cell_query = self.renderer(ren_cell_query)
        return view_cell_query

    def forward(self, x, v, xq, vq, n_obs=3):
        # Observation Encode
        wrd_cell = self.step_observation_encode(x, v)
        # Scene Fusion
        scene_cell = self.step_scene_fusion(wrd_cell, n_obs)
        # Query Image
        x_query, kl = self.step_query_view(scene_cell, xq, vq)
        return x_query, kl

    def sample(self, x, v, vq, n_obs=3, steps=None):
        # Observation Encode
        wrd_cell = self.step_observation_encode(x, v)
        # Scene Fusion
        scene_cell = self.step_scene_fusion(wrd_cell, n_obs)
        # Query Image
        x_query = self.step_query_view_sample(scene_cell, vq, steps)
        return x_query
