import numpy as np
import cv2
import os
import json
import datetime
import argparse
import configparser
import config_handle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from strgqn import STRGQN
from dataset import GqnDatasets

import numpy as np
import matplotlib.pyplot as plt

############ Util Functions ############
def eval(net, dataset, obs_size=1, max_batch=1000, img_size=(64,64)):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rmse_record = []
    mae_record = []
    ce_record = []
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
            # rmse
            mse_batch = (x_query_sample*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
            # mae
            mae_batch = torch.abs(x_query_sample*255 - x_query_gt*255)
            mae_batch = mae_batch.mean([1,2,3]).cpu().numpy()
            mae_record.append(mae_batch)
            # ce
            ce_batch = nn.BCELoss()(x_query_sample, x_query_gt)
            ce_batch = ce_batch.mean().cpu().numpy().reshape(1,1)
            ce_record.append(ce_batch)
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(max_batch), end="")
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    mae_record = np.concatenate(mae_record, 0)
    mae_mean = mae_record.mean()
    mae_std = mae_record.std()
    ce_record = np.concatenate(ce_record, 0)
    ce_mean = ce_record.mean()
    ce_std = ce_record.std()
    return rmse_mean, rmse_std, mae_mean, mae_std, ce_mean, ce_std

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
exp_path = parser.parse_args().path
print(exp_path)
config_file = exp_path + "config.conf"
config = configparser.ConfigParser()
config.read(config_file)
args = config_handle.get_config_strgqn(config)
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
net = STRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
rmse_mean, rmse_std, mae_mean, mae_std, ce_mean, ce_std = eval(net, test_dataset, obs_size=3, img_size=args.img_size)
print("RMSE:", rmse_mean, "+/-", rmse_std)
print("MAE:", mae_mean, "+/-", mae_std)
print("CE:", ce_mean, "+/-", ce_std)
