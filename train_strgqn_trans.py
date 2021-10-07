import numpy as np
import cv2
import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GqnDatasets
import configparser
import config_handle
import utils
from core.strgqn_trans import STRGQN

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', nargs='?', type=str, default="rrc" ,help='Experiment name.')
parser.add_argument('--config', nargs='?', type=str, default="./config.conf" ,help='Config filename.')
config_file = parser.parse_args().config
config = configparser.ConfigParser()
config.read(config_file)
args = config_handle.get_config_strgqn(config)
args.exp_name = parser.parse_args().exp_name
args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)

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

############ Dataset ############
path = args.data_path
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train, 
                            view_trans=args.view_trans, distort_type=args.distort_type)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test, 
                            view_trans=args.view_trans, distort_type=args.distort_type)
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))
print("Distort type:", args.distort_type)

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
params = list(net.parameters())
opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

############ Loss Function ############
if args.loss_type == "MSE":
    criterion = nn.MSELoss()
elif args.loss_type == "MAE":
    criterion = nn.L1Loss()
elif args.loss_type == "CE":
    creterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()

############ Training ############
max_obs_size = args.max_obs_size
total_steps = args.total_steps
train_record = {"loss_query":[], "lh_query":[], "kl_query":[]}
eval_record = []
best_eval = 999999
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
        v_obs = pose[:,obs_idx].reshape(-1,args.vsize).to(device)
        x_query_gt = image[:,query_idx].to(device)
        v_query = pose[:,query_idx].to(device)

        # ------------ Forward ------------
        net.zero_grad()
        if args.stochastic_unit:
            x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
            lh_query = criterion(x_query, x_query_gt).mean()
            kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3]))
            loss_query = lh_query + args.kl_scale*kl_query
            rec = [float(loss_query.detach().cpu().numpy()), float(lh_query.detach().cpu().numpy()), float(kl_query.detach().cpu().numpy())]
        else:
            x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            kl_query = 0
            lh_query = criterion(x_query, x_query_gt).mean()
            rec = [float(loss_query.detach().cpu().numpy()), float(loss_query.detach().cpu().numpy()), 0]

        # ------------ Train ------------
        loss_query.backward()
        opt.step()
        net.strn.sample_wcode(dist = "uniform")
        steps += 1
        
        # ------------ Print Result ------------
        if steps % 100 == 0:
            loss_query = float(loss_query.detach().cpu().numpy())
            lh_query = float(lh_query.detach().cpu().numpy())
            kl_query = float(kl_query.detach().cpu().numpy())

            print("[Ep %s (%s/%s)] loss_q: %f| lh_q: %f| kl_q: %f"%( \
                    str(epochs).zfill(4), str(steps), str(args.total_steps), rec[0], rec[1], rec[2]))

            loss_query_list.append(rec[0])
            lh_query_list.append(rec[1])
            kl_query_list.append(rec[2])

        # ------------ Output Image ------------
        if steps % eval_step == 0:
            print("------------------------------")
            obs_size = 3
            gen_size = 5
            # Train
            print("Generate training image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_train.png"
            canvas = utils.draw_query(net, train_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
            cv2.imwrite(fname, canvas)
            # Test
            print("Generate testing image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
            canvas = utils.draw_query(net, test_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
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
            eval_results_train = utils.eval(net, train_dataset, obs_size=3, max_batch=400, shuffle=False)
            print("Evaluate Testing Data ...")
            eval_results_test = utils.eval(net, test_dataset, obs_size=3, max_batch=400, shuffle=False)
            eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})
            print("Dump evaluation record ...")
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)

            # ------------ Save Model ------------
            if steps%100000 == 0:
                print("Save model ...")
                torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(4) + ".pth")
            
            # Apply RMSE as the metric for model selection.
            if eval_results_test["rmse"][0] < best_eval:
                best_eval = eval_results_test["rmse"][0]
                print("Save best model ...")
                torch.save(net.state_dict(), save_path + "model.pth")
            print("Best Test RMSE:", best_eval)
            print("------------------------------")

    print("==============================")
    if steps >= total_steps:
        print("Save final model ...")
        torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(4) + ".pth")
        break
    