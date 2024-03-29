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

def eval(net, dataset, obs_size=3, max_batch=400, img_size=(64,64)):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    lh_record = []
    kl_record = []
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
            x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
            kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3])).cpu().numpy()
            mse_batch = nn.MSELoss()(x_query_sample, x_query_gt).cpu().numpy()
            lh_query = mse_batch.mean()
            lh_query_var = mse_batch.var()
            lh_record.append(lh_query)
            kl_record.append(kl_query)
        len_size = min(len(dataset), max_batch)
        print("\r[Eval %s/%s] MSE: %f| KL: %f"%(str(it+1).zfill(4), str(len_size).zfill(4), lh_query, kl_query), end="")
    lh_mean = np.array(lh_record).mean()
    lh_mean_var = np.array(lh_query_var).mean()
    kl_mean = np.array(kl_record).mean()
    print("\nMSE =", lh_mean, ", KL =", kl_mean)
    return float(lh_mean), float(kl_mean), lh_record, kl_record

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

def gaussian_heatmap(mean, std, size):
    img = np.zeros(size, dtype=np.float32)
    mean_pix = (mean[0]*size[0], mean[1]*size[1])
    std_pix = std * size[0]
    for i in range(size[1]):
        for j in range(size[0]):
            temp = ((i-mean_pix[0])**2 + (j-mean_pix[1])**2)/std_pix**2
            img[j,i,:] = np.exp(-0.5 * temp) / (2*np.pi*std_pix**2)
    return img

####
def gaussian_heatmap(mean, std, size):
    img = np.zeros(size, dtype=np.float32)
    mean_pix = (mean[0]*size[0], mean[1]*size[1])
    std_pix = std * size[0]
    for i in range(size[1]):
        for j in range(size[0]):
            temp = ((i-mean_pix[0])**2 + (j-mean_pix[1])**2)/std_pix**2
            img[j,i,:] = np.exp(-0.5 * temp) / (2*np.pi*std_pix**2)
    return img

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
image = None
pose = None
for it, batch in enumerate(data_loader):
    if it >= 4:
        break
    #if it == 0:
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)

    obs_size = 3
    x_obs = image[0,0].permute(1,2,0).numpy()
    x_query = image[0,1].permute(1,2,0).numpy()
    v_obs = pose[0,0].reshape(-1,7).to(device)
    v_query = pose[0,1].reshape(-1,7).to(device)

    relation, activation, wcode = net.strn.transform(v_obs)
    wcode = wcode.detach().cpu().numpy().reshape(args.w, -1)
    activation = activation.detach().cpu().numpy().reshape(args.w, -1)
    print(wcode.shape, activation.shape)
    view_size = args.v

    std = 0.05#0.05
    spos = (0.5,0.7)#(0.3,0.7)#(0.5,0.8)
    #spos = np.random.rand(2)
    #spos[0] = spos[0]*0.6+0.2
    #spos[1] = spos[1]*0.3+0.6
    print("spos", spos)
    hp = gaussian_heatmap(spos, std, (args.v[0],args.v[1],args.c))
    view_cell_sim = hp / np.max(hp) * 1.5 
    view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,args.v[0],args.v[1],args.c).permute(0,3,1,2).to(device)
    
    signal = cv2.resize(hp[:,:,0], (128,128), interpolation=cv2.INTER_NEAREST)
    plt.figure()
    plt.imshow(signal)
    '''
    view_cell_torch = torch.zeros((1,args.c,args.v[0],args.v[1])).to(device)
    x = torch.linspace(0, 1, view_size[0])
    y = torch.linspace(0, 1, view_size[1])
    x_grid, y_grid = torch.meshgrid(x, y)
    vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).to(device) #(16*16, 2)
    print(vcode.shape, view_cell_torch.shape)
    view_cell_torch[0,:2,:] = vcode
    view_cell_torch[0,2,:] = 1
    print(view_cell_torch.max(), view_cell_torch.min())
    '''
    wcell = net.strn(view_cell_torch.reshape(-1,128,view_size[0]*view_size[1]), v_obs)
    wcell = wcell.permute(2,1,0).reshape(args.w, -1).detach().cpu().numpy() 

    xdata = wcode[:,0]
    ydata = wcode[:,1]
    zdata = wcode[:,2]
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("X-axis")
    ax.scatter3D(xdata, ydata, zdata, c=wcell[:,0], cmap='jet')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Y-axis")
    ax.scatter3D(xdata, ydata, zdata, c=wcell[:,1], cmap='jet')
    '''

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.set_title("Gaussian Pixel")
    ax.scatter3D(xdata, ydata, zdata, c=wcell[:,0], cmap='jet')#, s=5)
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Activation")
    ax.scatter3D(xdata, ydata, zdata, c=activation, cmap='jet')
    '''
    id = 39#37
    print(wcode[id], activation[id])
    ax.scatter3D(xdata[id], ydata[id], zdata[id], c=[[1,0,0]], s=100, marker="*")
    
plt.show()

