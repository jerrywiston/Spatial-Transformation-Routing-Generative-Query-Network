import numpy as np
import cv2
import os
import json
import datetime
import argparse
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from srgqn import SRGQN
from dataset import GqnDatasets

import numpy as np
import matplotlib.pyplot as plt

############ Util Functions ############
def draw_result(net, dataset, obs_size=3, gen_size=5, img_size=(64,64), convert_bgr=True):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,img_size[0],img_size[1]).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,obs_size].to(device)
        x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
        # Draw Observation
        canvas = np.zeros((img_size[0]*gen_size,img_size[1]*(obs_size+2),3), dtype=np.uint8)
        x_obs_draw = (image[:gen_size,:obs_size].detach()*255).permute(0,3,1,4,2).cpu().numpy().astype(np.uint8)
        x_obs_draw = cv2.cvtColor(x_obs_draw.reshape(img_size[0]*gen_size,img_size[1]*obs_size,3), cv2.COLOR_BGR2RGB)
        canvas[:img_size[0]*gen_size,:img_size[1]*obs_size,:] = x_obs_draw
        # Draw Query GT
        x_gt_draw = (image[:gen_size,obs_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_gt_draw = x_gt_draw.reshape(img_size[0]*gen_size,img_size[1],3)
        if convert_bgr:
            x_gt_draw = cv2.cvtColor(x_gt_draw, cv2.COLOR_BGR2RGB)
        canvas[:,img_size[1]*(obs_size):img_size[1]*(obs_size+1),:] = x_gt_draw
        # Draw Query Gen
        x_query_draw = (x_query[:gen_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_query_draw = x_query_draw.reshape(img_size[0]*gen_size,img_size[1],3)
        if convert_bgr:
            x_query_draw = cv2.cvtColor(x_query_draw, cv2.COLOR_BGR2RGB)
        canvas[:,img_size[1]*(obs_size+1):,:] = x_query_draw
        # Draw Grid
        cv2.line(canvas, (0,0),(0,img_size[1]*gen_size),(0,0,0), 2)
        cv2.line(canvas, (img_size[0]*(obs_size+2)-1,0),(img_size[1]*(obs_size+2)-1,img_size[1]*gen_size),(0,0,0), 2)
        cv2.line(canvas, (img_size[0]*obs_size,0),(img_size[1]*obs_size,img_size[1]*gen_size),(255,0,0), 2)
        cv2.line(canvas, (img_size[0]*(obs_size+1),0),(img_size[1]*(obs_size+1),img_size[1]*gen_size),(0,0,255), 2)
        for i in range(1,3):
            canvas[:,img_size[1]*i:img_size[1]*i+1,:] = 0
        for i in range(gen_size):
            canvas[img_size[0]*i:img_size[0]*i+1,:,:] = 0
            canvas[img_size[0]*(i+1)-1:img_size[0]*(i+1),:,:] = 0
        break
    return canvas

def eval(net, dataset, obs_size=3, max_batch=1000, img_size=(64,64)):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rmse_record = []
    for it, batch in enumerate(data_loader):
        if it+1 > max_batch:
            break
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,img_size[0],img_size[1]).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,obs_size].to(device)
        x_query_gt = image[:,obs_size].to(device)
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            mse_batch = (x_query_sample*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
        print("\rProgress: "+str(it).zfill(3)+"/"+str(max_batch), end="")
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    return rmse_mean, rmse_std

def get_config(config):
    # Fill the parameters
    args = lambda: None
    # Model Parameters
    args.w = config.getint('model', 'w')
    args.v = (config.getint('model', 'v_h'), config.getint('model', 'v_w'))
    args.c = config.getint('model', 'c')
    args.ch = config.getint('model', 'ch')
    args.down_size = config.getint('model', 'down_size')
    args.draw_layers = config.getint('model', 'draw_layers')
    args.share_core = config.getboolean('model', 'share_core')
    # Experimental Parameters
    args.data_path = config.get('exp', 'data_path')
    args.frac_train = config.getfloat('exp', 'frac_train')
    args.frac_test = config.getfloat('exp', 'frac_test')
    args.max_obs_size = config.get('exp', 'max_obs_size')
    args.total_steps = config.getint('exp', 'total_steps')
    args.total_epochs = config.getint('exp', 'total_epochs')
    args.kl_scale = config.getfloat('exp', 'kl_scale')
    if config.has_option('exp', 'convert_bgr'):
        args.convert_rgb = config.getboolean('exp', 'convert_bgr')
    else:
        args.convert_rgb = True
    return args

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
exp_path = parser.parse_args().path
print(exp_path)
config_file = exp_path + "config.conf"
config = configparser.ConfigParser()
config.read(config_file)
args = get_config(config)
args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)

# Print 
print("Configure File: %s"%(config_file))
print("Number of world cells: %d"%(args.w))
print("Size of view cells: " + str(args.v))
print("Number of concepts: %d"%(args.c))
print("Number of channels: %d"%(args.ch))
print("Downsampling size of view cell: %d"%(args.down_size))
print("Number of draw layers: %d"%(args.draw_layers))
if args.share_core:
    print("Share core: True")
else:
    print("Share core: False")

############ Dataset ############
path = args.data_path
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Test data: ", len(test_dataset))

############ Create Folder ############
result_path = exp_path + "result/"
save_path = exp_path + "save/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=7, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
rmse_mean, rmse_std = eval(net, test_dataset, obs_size=3, img_size=args.img_size)
print(rmse_mean, rmse_std)
