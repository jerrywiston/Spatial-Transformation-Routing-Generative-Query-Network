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
 
from srgqn_vae import SRGQN
from dataset import GqnDatasets

############ Util Functions ############
def draw_result(net, dataset, obs_size=3, gen_size=5, img_size=(64,64), convert_bgr=True):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
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
            #kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3])).cpu().numpy()
            kl_query = kl_query.cpu().numpy()
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
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))

############ Create Folder ############
now = datetime.datetime.now()
#tinfo = "%d-%d-%d_%d-%d"%(now.year, now.month, now.day, now.hour, now.minute) #second / microsecond
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
net = SRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=7, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
params = list(net.parameters())
opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))
net.load_state_dict(torch.load("experiments/2020-11-21_rfc_no_rot_w2000_c128/save/srgqn.pth"))
############ Training ############
max_obs_size = args.max_obs_size
total_steps = args.total_steps
total_epochs = args.total_epochs
train_record = {"loss_query":[], "lh_query":[], "kl_query":[]}
eval_record = {"mse_train":[], "kl_train":[], "mse_test":[], "kl_test":[]}
best_mse = 999999
print("Start training ...")
print("==============================")
steps = 0
epochs = 0
eval_step = 5000
start_time = str(datetime.datetime.now())
while(True):
    epochs += 1
    print("Experiment start time", start_time)
    print("Start Epoch", epochs, ", time:", str(datetime.datetime.now()))
    # ------------ Shuffle Datasets ------------
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    loss_query_list, lh_query_list, kl_query_list = [], [], []
    # ------------ One Epoch ------------
    for it, batch in enumerate(train_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)

        # ------------ Get data (Random Observation) ------------
        obs_size = np.random.randint(1,max_obs_size)
        obs_idx = np.random.choice(image.shape[1], obs_size)
        query_idx = np.random.randint(0, image.shape[1]-1)
        
        x_obs = image[:,obs_idx].reshape(-1,3,args.img_size[0],args.img_size[1]).to(device)
        v_obs = pose[:,obs_idx].reshape(-1,7).to(device)
        x_query_gt = image[:,query_idx].to(device)
        v_query = pose[:,query_idx].to(device)

        # ------------ Get data (Fixed Observation) ------------
        '''
        obs_size = 3
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        x_query_gt = image[:,obs_size+1].to(device)
        v_query = pose[:,obs_size+1].to(device)
        '''

        # ------------ Forward ------------
        net.zero_grad()
        x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
        lh_query = nn.MSELoss()(x_query, x_query_gt).mean()
        #kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3]))
        loss_query = lh_query + args.kl_scale*kl_query

        # ------------ Train ------------
        loss_query.backward()
        opt.step()
        steps += 1
        
        # ------------ Print Result ------------
        if steps % 100 == 0:
            loss_query = float(loss_query.detach().cpu().numpy())
            lh_query = float(lh_query.detach().cpu().numpy())
            kl_query = float(kl_query.detach().cpu().numpy())

            print("[Ep %s (%s/%s)] loss_q: %f| lh_q: %f| kl_q: %f"%( \
                str(epochs).zfill(4), str(steps), str(total_steps), \
                loss_query, lh_query, kl_query))
            
            loss_query_list.append(loss_query)
            lh_query_list.append(lh_query)
            kl_query_list.append(kl_query)

        # ------------ Output Image ------------
        if steps % eval_step == 0:
            print("------------------------------")
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
            train_record["lh_query"].append(lh_query_list)
            train_record["kl_query"].append(kl_query_list)
            print("Dump training record ...")
            with open(model_path+'train_record.json', 'w') as file:
                json.dump(train_record, file)

            # ------------ Evaluation Record ------------
            print("Evaluate Training Data ...")
            lh_train, kl_train, _, _ = eval(net, train_dataset, obs_size=3, img_size=args.img_size)
            print("Evaluate Testing Data ...")
            lh_test, kl_test, _, _ = eval(net, test_dataset, obs_size=3, img_size=args.img_size)
            eval_record["mse_train"].append(lh_train)
            eval_record["kl_train"].append(kl_train)
            eval_record["mse_test"].append(lh_test)
            eval_record["kl_test"].append(kl_test)
            print("Dump evaluation record ...")
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)

            # ------------ Save Model (One Epoch) ------------
            if steps%100000 == 0:
                print("Save model ...")
                torch.save(net.state_dict(), save_path + "srgqn_" + str(steps).zfill(4) + ".pth")

            if lh_test < best_mse:
                best_mse = lh_test
                print("Save best model ...")
                torch.save(net.state_dict(), save_path + "srgqn.pth")
            print("Best Test MSE:", best_mse)
            print("------------------------------")

    print("==============================")
    if steps >= total_steps:
        print("Save final model ...")
        torch.save(net.state_dict(), save_path + "srgqn_final.pth")
        break
