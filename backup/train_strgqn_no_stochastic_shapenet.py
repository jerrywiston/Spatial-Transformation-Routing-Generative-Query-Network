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

############ Util Functions ############
def draw_result(net, dataset, obs_size=3, gen_size=5, img_size=(128,128), convert_bgr=True):
    x_obs, v_obs, x_query_gt, v_query = dataset_shapenet.get_batch(dataset, obs_size, gen_size)
    x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
    # Draw Observation
    canvas = np.zeros((img_size[0]*gen_size,img_size[1]*(obs_size+2),3), dtype=np.uint8)
    x_obs_draw = (x_obs.detach()*255).reshape(gen_size, obs_size,3, img_size[0], img_size[1]).permute(0,3,1,4,2).cpu().numpy().astype(np.uint8)
    x_obs_draw = cv2.cvtColor(x_obs_draw.reshape(img_size[0]*gen_size,img_size[1]*obs_size,3), cv2.COLOR_BGR2RGB)
    canvas[:img_size[0]*gen_size,:img_size[1]*obs_size,:] = x_obs_draw
    # Draw Query GT
    x_gt_draw = (x_query_gt[:gen_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
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

    return canvas

def eval(net, dataset, obs_size=3, max_batch=400, img_size=(64,64)):
    lh_record = []
    for it in range(max_batch):
        x_obs, v_obs, x_query_gt, v_query = shapenet_dataset.get_batch(dataset, obs_size, 1)
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            mse_batch = nn.BCELoss()(x_query_sample, x_query_gt).cpu().numpy()
            lh_query = mse_batch.mean()
            lh_query_var = mse_batch.var()
            lh_record.append(lh_query)
        len_size = min(len(dataset), max_batch)
        print("\r[Eval %s/%s] CE: %f"%(str(it+1).zfill(4), str(len_size).zfill(4), lh_query), end="")
    lh_mean = np.array(lh_record).mean()
    lh_mean_var = np.array(lh_query_var).mean()
    print("\nCE =", lh_mean)
    return float(lh_mean), lh_record

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
parser.add_argument('--exp_name', nargs='?', type=str, default="rrc" ,help='Experiment name.')
parser.add_argument('--config', nargs='?', type=str, default="./config.conf" ,help='Config filename.')
config_file = parser.parse_args().config
config = configparser.ConfigParser()
config.read(config_file)
args = get_config(config)
args.exp_name = parser.parse_args().exp_name
args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)

# Print 
print("Configure File: %s"%(config_file))
print("Experiment Name: %s"%(args.exp_name))
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
train_dataset = shapenet_dataset.read_dataset(path=path, mode="train")
test_dataset = shapenet_dataset.read_dataset(path=path, mode="test")
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))

############ Create Folder ############
now = datetime.datetime.now()
#tinfo = "%d-%d-%d_%d-%d"%(now.year, now.month, now.day, now.hour, now.minute) #second / microsecond
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "experiments/"
model_name = args.exp_name
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
net = SRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=7, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
params = list(net.parameters())
opt = optim.Adam(params, lr=1e-4, betas=(0.5, 0.999))

############ Training ############
max_obs_size = args.max_obs_size
total_steps = args.total_steps
total_epochs = args.total_epochs
train_record = {"loss_query":[]}
eval_record = {"ce_train":[], "ce_test":[]}
best_mse = 999999
print("Start training ...")
print("==============================")
steps = 0
epochs = 0
eval_step = 5000
start_time = str(datetime.datetime.now())
while(True):
    loss_query_list, lh_query_list = [], []

    # ------------ Get data (Random Observation) ------------
    obs_size = np.random.randint(1,max_obs_size)
    x_obs, v_obs, x_query_gt, v_query = shapenet_dataset.get_batch(train_dataset, obs_size, 32)

    # ------------ Forward ------------
    net.zero_grad()
    x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
    loss_query = nn.MSELoss()(x_query, x_query_gt).mean()

    # ------------ Train ------------
    loss_query.backward()
    opt.step()
    steps += 1
        
    # ------------ Print Result ------------
    if steps % 100 == 0:
        loss_query = float(loss_query.detach().cpu().numpy())
        print("[%s/%s] loss_q: %f"%(str(steps), str(total_steps), loss_query))
            
        loss_query_list.append(loss_query)

    # ------------ Output Image ------------
    if steps % eval_step == 0:
        print("------------------------------")
        print("Experiment start time", start_time)
        print("Evaluation Step", steps, ", time:", str(datetime.datetime.now()))
        print("Generate image ...")
        obs_size = 3
        gen_size = 5
        # Train
        fname = img_path+str(int(steps/eval_step)).zfill(4)+"_train.png"
        canvas = draw_result(net, train_dataset, obs_size, gen_size, args.img_size)
        cv2.imwrite(fname, canvas)
        # Test
        fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
        canvas = draw_result(net, test_dataset, obs_size, gen_size, args.img_size)
        cv2.imwrite(fname, canvas)

        # ------------ Training Record ------------
        train_record["loss_query"].append(loss_query_list)
        print("Dump training record ...")
        with open(model_path+'train_record.json', 'w') as file:
            json.dump(train_record, file)

        # ------------ Evaluation Record ------------
        print("Evaluate Training Data ...")
        lh_train, _ = eval(net, train_dataset, obs_size=3, img_size=args.img_size)
        print("Evaluate Testing Data ...")
        lh_test, _ = eval(net, test_dataset, obs_size=3, img_size=args.img_size)
        eval_record["ce_train"].append(lh_train)
        eval_record["ce_test"].append(lh_test)
        print("Dump evaluation record ...")
        with open(model_path+'eval_record.json', 'w') as file:
            json.dump(eval_record, file)

        # ------------ Save Model ------------
        print("Save model ...")
        torch.save(net.state_dict(), save_path + "srgqn_" + str(steps).zfill(4) + ".pth")

        if lh_test < best_mse:
            best_mse = lh_test
            print("Save best model ...")
            torch.save(net.state_dict(), save_path + "srgqn.pth")
        print("Best Test CE:", best_mse)
        print("------------------------------")

    if steps >= total_steps:
        print("Save final model ...")
        torch.save(net.state_dict(), save_path + "srgqn_final.pth")
        break
    
