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

from dataset import GqnDatasets
import dataset_shapenet
import config_handle
import utils
from strgqn import STRGQN

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', nargs='?', type=str, default="strgqn" ,help='Experiment name.')
parser.add_argument('--config', nargs='?', type=str, default="./config.conf" ,help='Config filename.')
parser.add_argument("--shapenet", help="Use ShapeNet dataset.", action="store_true")
config_file = parser.parse_args().config
config = configparser.ConfigParser()
config.read(config_file)
args = config_handle.get_config_strgqn(config)
args.exp_name = parser.parse_args().exp_name
args.shapenet = parser.parse_args().shapenet

# Print Training Information
print("Configure File: %s"%(config_file))
print("Experiment Name: %s"%(args.exp_name))
print("Number of world cells: %d"%(args.w))
print("Size of view cells: " + str(args.v))
print("Number of concepts: %d"%(args.c))
print("Number of channels: %d"%(args.ch))
print("Downsampling size of view cell: %d"%(args.down_size))
print("Number of draw layers: %d"%(args.draw_layers))
print("Size of view pose: %d"%(args.vsize))
if args.share_core:
    print("Share core: True")
else:
    print("Share core: False")

############ Create Folder ############
now = datetime.datetime.now()
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "experiments/"
model_name = args.exp_name + "_w%d_c%d"%(args.w, args.c)
model_path = exp_path + tinfo + "_" + model_name + "/"

img_path = model_path + "img/"
save_path = model_path + "save/"
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save config file
with open(model_path + 'config.conf', 'w') as cfile:
    config.write(cfile)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = STRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)

if args.shapenet:
    utils.train_shapenet(net, args, model_path)
else:
    utils.train(net, args, model_path)