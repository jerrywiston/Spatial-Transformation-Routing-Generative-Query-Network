import numpy as np
import cv2
import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
 
from srgqn import SRGQN
from dataset import GqnDatasets

############ Util Functions ############
def draw_result(net, dataset, obs_size=3, gen_size=5):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,obs_size+1].to(device)
        x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
        # Draw Observation
        canvas = np.zeros((64*gen_size,64*(obs_size+2),3), dtype=np.uint8)
        x_obs_draw = (image[:gen_size,:obs_size].detach()*255).permute(0,3,1,4,2).cpu().numpy().astype(np.uint8)
        x_obs_draw = cv2.cvtColor(x_obs_draw.reshape(64*gen_size,64*obs_size,3), cv2.COLOR_BGR2RGB)
        canvas[:64*gen_size,:64*obs_size,:] = x_obs_draw
        # Draw Query GT
        x_gt_draw = (image[:gen_size,obs_size+1].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_gt_draw = cv2.cvtColor(x_gt_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*(obs_size):64*(obs_size+1),:] = x_gt_draw
        # Draw Query Gen
        x_query_draw = (x_query[:gen_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_query_draw = cv2.cvtColor(x_query_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*(obs_size+1):,:] = x_query_draw
        # Draw Grid
        cv2.line(canvas, (0,0),(0,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*(obs_size+2)-1,0),(64*(obs_size+2)-1,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*obs_size,0),(64*obs_size,64*gen_size),(255,0,0), 2)
        cv2.line(canvas, (64*(obs_size+1),0),(64*(obs_size+1),64*gen_size),(0,0,255), 2)
        for i in range(1,4):
            canvas[:,64*i:64*i+1,:] = 0
        for i in range(gen_size):
            canvas[64*i:64*i+1,:,:] = 0
            canvas[64*(i+1)-1:64*(i+1),:,:] = 0
        break
    return canvas

############ Parameter Parsing ############
parser = argparse.ArgumentParser(description='Traning parameters of SR-GQN.')
parser.add_argument('--data_path', nargs='?', type=str, default="../GQN-Datasets-pt/rooms_ring_camera", help='Dataset name.')
parser.add_argument('--frac_train', nargs='?', type=float, default=0.01, help='Fraction of data used for training.')
parser.add_argument('--frac_test', nargs='?', type=float, default=0.01, help='Fraction of data used for testing.')
parser.add_argument('--exp_path', nargs='?', type=str, default="rrc" ,help='Experiment name (for the created result folder).')
args = parser.parse_args()

print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))

############ Dataset ############
path = args.data_path
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SRGQN(n_wrd_cells=args.w, n_ren_cells=args.r, tsize=args.c, ch=64, vsize=7).to(device)
net.load_state_dict(torch.load(PATH))
net.eval()

# ------------ Shuffle Datasets ------------
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for it, batch in enumerate(train_loader):
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)

    obs_size = 3
    x_obs = image[:,obs_size].reshape(-1,3,64,64).to(device)
    v_obs = pose[:,obs_size].reshape(-1,7).to(device)
    x_query_gt = image[:,obs_size+1].to(device)
    v_query = pose[:,obs_size+1].to(device)