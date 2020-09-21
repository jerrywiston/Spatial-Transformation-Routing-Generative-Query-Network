import numpy as np
import cv2
import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
 
import srgqn_generative as srgqn
from gqn_dataset import GqnDatasets

############ Util Functions ############
def draw_result(net, dataset, obs_size=4, gen_size=6):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,obs_size+1].to(device)
        x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
        # Draw Observation
        canvas = np.zeros((64*gen_size,64*(obs_size+2),3), dtype=np.uint8)
        x_obs_draw = (image[:gen_size,:obs_size].detach()*255).permute(0,3,1,4,2).cpu().numpy().astype(np.uint8)
        x_obs_draw = cv2.cvtColor(x_obs_draw.reshape(64*gen_size,64*4,3), cv2.COLOR_BGR2RGB)
        canvas[:64*gen_size,:64*4,:] = x_obs_draw
        # Draw Query GT
        x_gt_draw = (image[:gen_size,obs_size+1].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_gt_draw = cv2.cvtColor(x_gt_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*4:64*5,:] = x_gt_draw
        # Draw Query Gen
        x_query_draw = (x_query[:gen_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_query_draw = cv2.cvtColor(x_query_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*5:,:] = x_query_draw
        # Draw Grid
        cv2.line(canvas, (0,0),(0,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*6-1,0),(64*6-1,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*4,0),(64*4,64*gen_size),(255,0,0), 2)
        cv2.line(canvas, (64*5,0),(64*5,64*gen_size),(0,0,255), 2)
        for i in range(1,4):
            canvas[:,64*i:64*i+1,:] = 0
        for i in range(gen_size):
            canvas[64*i:64*i+1,:,:] = 0
            canvas[64*(i+1)-1:64*(i+1),:,:] = 0
        break
    return canvas

def eval(net, dataset, obs_size=4):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    lh_record = []
    kl_record = []
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,obs_size+1].to(device)
        x_query_gt = image[:,obs_size+1].to(device)
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
            lh_query = nn.MSELoss()(x_query_sample, x_query_gt).mean()
            kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3]))
            lh_record.append(lh_query.cpu().numpy())
            kl_record.append(kl_query.cpu().numpy())
        print("\r[Eval %s/%s] MSE: %f| KL: %f"%(str(it).zfill(4), str(len(dataset)).zfill(4), lh_query, kl_query), end="")
    lh_mean = np.array(lh_record).mean()
    kl_mean = np.array(kl_record).mean()
    print("\nMSE =", lh_mean,", KL =", kl_mean)
    return float(lh_mean), float(kl_mean), lh_record, kl_record

############ Dataset ############
#path = "GQN-Datasets-pt/rooms_ring_camera"
path = "../Spatial-Concept-Learner/GQN-Datasets/rooms_ring_camera_pt"
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=1.0)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=0.2)
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))

############ Create Folder ############
now = datetime.datetime.now()
#tinfo = "%d-%d-%d_%d-%d"%(now.year, now.month, now.day, now.hour, now.minute) #second / microsecond
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "experiments/"
model_name = "rrc_w1024_r256_t32"
model_path = exp_path + model_name + "_" + tinfo + "/"

img_path = model_path + "img/"
save_path = model_path + "save/"
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = srgqn.SRGQN(n_wrd_cells=1600, n_ren_cells=400, tsize=128, ch=64, vsize=7).to(device)
params = list(net.parameters())
opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

############ Training ############
max_obs_size = 5
total_epoch = 200
train_record = {"loss_query":[], "lh_query":[], "kl_query":[], "loss_rec":[], "lh_rec":[], "kl_rec":[]}
eval_record = {"mse_train":[], "kl_train":[], "mse_test":[], "kl_test":[]}
print("Start training ...")
print("==============================")
for epoch in range(total_epoch):
    print("Start Epoch", str(datetime.datetime.now()))
    # ------------ Shuffle Datasets ------------
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    loss_query_list, lh_query_list, kl_query_list = [], [], []
    loss_rec_list, lh_rec_list, kl_rec_list = [], [], []
    # ------------ One Epoch ------------
    for it, batch in enumerate(train_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        obs_size = np.random.randint(1,max_obs_size)
        obs_idx = np.random.choice(image.shape[1], obs_size)
        query_idx = np.random.randint(0, image.shape[1]-1)

        # ------------ Get data ------------
        #x_obs = image[:,obs_idx].reshape(-1,3,64,64).to(device)
        #v_obs = pose[:,obs_idx].reshape(-1,7).to(device)
        #x_query_gt = image[:,query_idx].to(device)
        #v_query = pose[:,query_idx].to(device)
        obs_size = 3
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        x_query_gt = image[:,obs_size+1].to(device)
        v_query = pose[:,obs_size+1].to(device)
        
        # ------------ Forward ------------
        net.zero_grad()
        x_rec, kl_rec = 0, 0#net.reconstruct(x_obs)
        lh_rec = 0#nn.MSELoss()(x_rec, x_obs).mean()
        kl_rec = 0#torch.mean(torch.sum(kl_rec, dim=[1,2,3]))
        loss_rec = 0#lh_rec + 0.01*kl_rec
        
        x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
        lh_query = nn.MSELoss()(x_query, x_query_gt).mean()
        kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3]))
        loss_query = lh_query + 0.01*kl_query

        # ------------ Train ------------
        loss = loss_query + 0.01*loss_rec 
        loss.backward()
        opt.step()
        
        # ------------ Print Result ------------
        if it % 100 == 0:
            loss_query = float(loss_query.detach().cpu().numpy())
            lh_query = float(lh_query.detach().cpu().numpy())
            kl_query = float(kl_query.detach().cpu().numpy())
            loss_rec = 0#float(loss_rec.detach().cpu().numpy())
            lh_rec = 0#float(lh_rec.detach().cpu().numpy())
            kl_rec = 0#float(kl_rec.detach().cpu().numpy())

            print("[Ep %s/%s (%s/%s)] loss_q: %f| loss_r: %f| lh_q: %f| lh_r: %f| kl_q: %f| kl_r: %f"%( \
                str(epoch).zfill(4), str(total_epoch).zfill(4), str(it).zfill(4), str(len(train_dataset)).zfill(4), \
                loss_query, loss_rec, lh_query, lh_rec, kl_query, kl_rec))
            
            loss_query_list.append(loss_query)
            lh_query_list.append(lh_query)
            kl_query_list.append(kl_query)
            loss_rec_list.append(loss_rec)
            lh_rec_list.append(lh_rec)
            kl_rec_list.append(kl_rec)
    
    # ------------ Save Model (One Epoch) ------------
    print("Save model ...")
    torch.save(net.state_dict(), save_path + "srgqn_ep" + str(epoch).zfill(4) + ".pth")

    # ------------ Output Image ------------
    print("Generate image ...")
    obs_size = 4
    gen_size = 6
    # Train
    fname = img_path+str(epoch).zfill(4)+"_train.png"
    canvas = draw_result(net, train_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)
    # Test
    fname = img_path+str(epoch).zfill(4)+"_test.png"
    canvas = draw_result(net, test_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)

    # ------------ Training Record ------------
    train_record["loss_query"].append(loss_query_list)
    train_record["lh_query"].append(lh_query_list)
    train_record["kl_query"].append(kl_query_list)
    train_record["loss_rec"].append(loss_rec_list)
    train_record["lh_rec"].append(lh_rec_list)
    train_record["kl_rec"].append(kl_rec_list)
    print("Dump training record ...")
    with open(model_path+'train_record.json', 'w') as file:
        json.dump(train_record, file)

    # ------------ Evaluation Record ------------
    print("Evaluate Training Data ...")
    lh_train, kl_train, _, _ = eval(net, train_dataset, obs_size=4)
    print("Evaluate Testing Data ...")
    lh_test, kl_test, _, _ = eval(net, test_dataset, obs_size=4)
    eval_record["mse_train"].append(lh_train)
    eval_record["kl_train"].append(kl_train)
    eval_record["mse_test"].append(lh_test)
    eval_record["kl_test"].append(kl_test)
    print("Dump evaluation record ...")
    with open(model_path+'eval_record.json', 'w') as file:
        json.dump(eval_record, file)

    print("==============================")
    