import numpy as np
import cv2
import os
import json
import datetime
import argparse
import configparser
import numpy as np
import torch

from gqn import GQN
from dataset import GqnDatasets
import config_handle
import utils

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
exp_path = parser.parse_args().path
print(exp_path)
config_file = exp_path + "config.conf"
config = configparser.ConfigParser()
config.read(config_file)
args = config_handle.get_config_gqn(config)

# Print 
print("Configure File: %s"%(config_file))
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
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test, 
                    view_trans=args.view_trans, distort_type=args.distort_type)
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
net = GQN(csize=args.c, ch=args.ch, vsize=args.vsize, draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

obs_size = 3
row_size = 1
gen_size = 400
img_list = utils.draw_query(net, test_dataset, obs_size=obs_size, vsize=args.vsize, row_size=row_size, gen_size=gen_size, img_size=args.img_size)
print("Output image files ...")
for i in range(len(img_list)):
    cv2.imwrite(result_path+"result_"+str(i).zfill(3)+".jpg", img_list[i]) 