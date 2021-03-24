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

from strgqn import SRGQN
import dataset_shapenet
from dataset_shapenet import get_pose_code as pcode
import numpy as np

magic_list_arith = [
    [[36, 2],[34, 0],[35, 2],[37, 5]],
    [[ 2, 2],[38, 2],[39, 2],[ 3, 5]],
    [[ 6, 2],[ 4, 2],[ 5, 2],[ 7, 5]],
    [[10, 2],[ 8, 2],[ 9, 0],[11, 5]],
    [[15, 2],[13, 0],[14, 2],[16, 5]],
    [[19, 2],[17, 2],[18, 2],[20, 5]],
]

magic_list_obj4 = [
    [[ 39, 43],[ 39, 13],[ 39, 23],[ 39, 51]],
    [[133, 19],[133,  5],[133, 33],[133, 26]],
    [[ 92, 18],[ 92, 31],[ 92, 36],[ 92, 18]],
    [[ 59, 51],[ 59, 28],[ 59, 29],[ 59, 21]],
]

magic_list_train = [
    [[225, 21],[225, 51],[225, 45],[225,  9]],
    [[226, 43],[226,  13],[226, 23],[226, 51]],
]

def eval(net, data, magic_list, path, op="arith"):
    print(data.shape)
    for i in range(len(magic_list)):
        sid_o1, sid_o2, sid_o3, sid_q = magic_list[i][0][0], magic_list[i][1][0], magic_list[i][2][0], magic_list[i][3][0]
        vid_o1, vid_o2, vid_o3, vid_q = magic_list[i][0][1], magic_list[i][1][1], magic_list[i][2][1], magic_list[i][3][1]

        x1_np = data[sid_o1,vid_o1]/255.
        x2_np = data[sid_o2,vid_o2]/255.
        x3_np = data[sid_o3,vid_o3]/255.
        x4_np = data[sid_q,vid_q]/255.

        x_obs = np.concatenate([x1_np[np.newaxis,...], x2_np[np.newaxis,...], x3_np[np.newaxis,...]], 0)
        v_obs = np.concatenate([pcode(vid_o1,0,0)[np.newaxis,...], pcode(vid_o2,0,0)[np.newaxis,...], pcode(vid_o3,0,0)[np.newaxis,...]], 0)
        v_query = pcode(vid_q)[np.newaxis,...]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_obs = torch.FloatTensor(x_obs).permute(0,3,1,2).to(device)
        v_obs = torch.FloatTensor(v_obs).to(device)
        v_query = torch.FloatTensor(v_query).to(device)

        if op == "arith":
            scene_cell = net.step_observation_encode(x_obs, v_obs)
            scene_cell = torch.sigmoid(scene_cell)
            scene_cell = scene_cell[0:1] - scene_cell[1:2] + scene_cell[2:3]
        else:
            scene_cell = net.step_observation_encode(x_obs, v_obs)
            scene_cell = scene_cell[0:1] + scene_cell[1:2] + scene_cell[2:3]
            scene_cell = torch.sigmoid(scene_cell)
            
        x_query = net.step_query_view_sample(scene_cell, v_query)
        x_query = x_query.detach().permute(0,2,3,1).cpu().numpy()[0]

        img_draw = np.concatenate([x1_np, x2_np, x3_np, x4_np, x_query], 1)*255
        img_draw = cv2.cvtColor(img_draw.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_draw[:,127:129,:] = 255
        img_draw[:,255:257,:] = 255
        img_draw[:,383:385,:] = 255
        img_draw[:,511:513,:] = 255
        if op == "arith":
            cv2.imwrite(path+"arith_"+str(i)+".jpg", img_draw)
        elif op == "obj4":
            cv2.imwrite(path+"obj4_"+str(i)+".jpg", img_draw)
        else:
            cv2.imwrite(path+"train_"+str(i)+".jpg", img_draw)

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
dataset_arith = shapenet_dataset.read_dataset(path=path, mode="arith")
dataset_obj4 = shapenet_dataset.read_dataset(path=path, mode="obj4")
dataset_train = shapenet_dataset.read_dataset(path=path, mode="train")
print("Data path: %s"%(args.data_path))

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

eval(net, dataset_arith, magic_list_arith, result_path, "arith")
eval(net, dataset_obj4, magic_list_obj4, result_path, "obj4")
eval(net, dataset_train, magic_list_train, result_path, "train")
