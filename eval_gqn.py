import os
import json
import argparse
import configparser

import torch

from dataset import GqnDatasets
from core.gqn import GQN
import config_handle
import utils

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str, help='Experimental path.')
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
net = GQN(csize=args.c, ch=args.ch, vsize=args.vsize, draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

eval_results = utils.eval(net, test_dataset, obs_size=3, max_batch=1000, shuffle=False)
for temp in eval_results:
    print(temp, ":", eval_results[temp][0], "+/-", eval_results[temp][1])

