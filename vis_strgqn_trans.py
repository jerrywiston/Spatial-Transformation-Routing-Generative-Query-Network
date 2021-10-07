import os
import json
import argparse
import configparser
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GqnDatasets
from core.strgqn_trans import STRGQN
#from core.strgqn_plus import STRGQN
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
args = config_handle.get_config_strgqn(config)

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
net.load_state_dict(torch.load(save_path+"model.pth"))
net.eval()

'''
fig = plt.figure()
ax = fig.gca(projection='3d')

z1 = np.random.randn(50)
x1 = np.random.randn(50)
y1 = np.random.randn(50)
z2 = np.random.randn(50)
x2 = np.random.randn(50)
y2 = np.random.randn(50)

ax.scatter(x1, y1, z1, c=z1, cmap='Reds', marker='^', label='My Points 1')
ax.scatter(x2, y2, z2, c=z2, cmap='Blues', marker='o', label='My Points 2')

ax.legend()
plt.show()
'''
obs_size = 3
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for it, batch in enumerate(data_loader):
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)
    img_size = (image.shape[-2], image.shape[-1])
    vsize = pose.shape[-1]
    # Get Data
    x_obs = image[:,:obs_size].reshape(-1,3,img_size[0],img_size[1]).to(device)
    v_obs = pose[:,:obs_size].reshape(-1,vsize).to(device)
    v_query = pose[:,obs_size].to(device)
    x_query_gt = image[:,obs_size].to(device)

    # Rotation Demo
    divide_num = 12
    v_query = np.zeros((divide_num, 7),dtype=np.float32)
    trans = (0., 0.)
    for i in range(divide_num):
        v_query[i,0] = trans[0]
        v_query[i,1] = trans[1]
        v_query[i,3] = np.cos(np.pi*2/divide_num*i)
        v_query[i,4] = np.sin(np.pi*2/divide_num*i)
    v_query_torch = torch.FloatTensor(v_query).to(device)
    relation, activation, wcode = net.strn.transform(v_query_torch)
    wcode = net.strn.wdist
    wcode_np = wcode.detach().cpu().numpy()
    for i in range(divide_num):
        activation_np = activation[i,:,0].detach().cpu().numpy()
        fig = plt.figure()        
        # 2D Plot
        plt.xlim(xmin=-1.2, xmax=1.2)
        plt.ylim(ymin=-1.2, ymax=1.2)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.scatter(wcode_np[:,0], wcode_np[:,1], c=activation_np, cmap=None, marker='o')
        plt.plot(v_query[i,0], v_query[i,1], "rs")
        plt.savefig(result_path + "/act_rotate_" + str(int(360/divide_num*i)).zfill(3) + ".jpg", bbox_inches = 'tight')
        plt.close()
        print(i)

    # Translation Demo
    divide_num = 4
    v_query = np.zeros((divide_num**2, 7),dtype=np.float32)
    rot = 0
    count = 0
    for i in range(divide_num):
        for j in range(divide_num):
            v_query[count,0] = -1 + i*2/(divide_num-1)
            v_query[count,1] = -1 + j*2/(divide_num-1)
            v_query[count,3] = np.cos(np.deg2rad(rot))
            v_query[count,4] = np.sin(np.deg2rad(rot))
            count += 1
    print(v_query)
    v_query_torch = torch.FloatTensor(v_query).to(device)
    relation, activation, wcode = net.strn.transform(v_query_torch)
    wcode = net.strn.wdist
    wcode_np = wcode.detach().cpu().numpy()
    
    for i in range(divide_num):
        for j in range(divide_num):
            activation_np = activation[i*divide_num+j,:,0].detach().cpu().numpy()
            fig = plt.figure()        
            # 2D Plot
            #plt.xlim(xmin=-1.2, xmax=1.2)
            #plt.ylim(ymin=-1.2, ymax=1.2)
            #
            ax = fig.add_subplot(131)
            ax.set_aspect('equal', adjustable='box')
            plt.scatter(wcode_np[:,0], wcode_np[:,1], c=activation_np, cmap=None, marker='.')
            plt.plot(v_query[i*divide_num+j,0], v_query[i*divide_num+j,1], "rs")
            #
            ax = fig.add_subplot(132)
            ax.set_aspect('equal', adjustable='box')
            plt.scatter(wcode_np[:,0], wcode_np[:,2], c=activation_np, cmap=None, marker='.')
            plt.plot(v_query[i*divide_num+j,0], v_query[i*divide_num+j,2], "rs")
            #
            ax = fig.add_subplot(133)
            ax.set_aspect('equal', adjustable='box')
            plt.scatter(wcode_np[:,1], wcode_np[:,2], c=activation_np, cmap=None, marker='.')
            plt.plot(v_query[i*divide_num+j,1], v_query[i*divide_num+j,2], "rs")
            #
            plt.savefig(result_path + "/act_trans_" + str(i) + "_" + str(j) + ".jpg", bbox_inches = 'tight')
            print(i,j)
            plt.close()
    break




