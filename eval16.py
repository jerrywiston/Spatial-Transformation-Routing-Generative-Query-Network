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

#from srgqn import SRGQN 
from srgqn2 import SRGQN
from dataset import GqnDatasets

############ Util Functions ############
def draw_result(net, dataset, obs_size=3, gen_size=5, steps=None):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,64,64).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,7).to(device)
        v_query = pose[:,-1].to(device)
        x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size, steps=steps)
        # Draw Observation
        canvas = np.zeros((64*gen_size,64*(obs_size+2),3), dtype=np.uint8)
        x_obs_draw = (image[:gen_size,:obs_size].detach()*255).permute(0,3,1,4,2).cpu().numpy().astype(np.uint8)
        x_obs_draw = cv2.cvtColor(x_obs_draw.reshape(64*gen_size,64*obs_size,3), cv2.COLOR_BGR2RGB)
        canvas[:64*gen_size,:64*obs_size,:] = x_obs_draw
        # Draw Query GT
        x_gt_draw = (image[:gen_size,-1].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_gt_draw = cv2.cvtColor(x_gt_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*(obs_size):64*(obs_size+1),:] = x_gt_draw
        # Draw Query Gen
        x_query_draw = (x_query[:gen_size].detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        x_query_draw = cv2.cvtColor(x_query_draw.reshape(64*gen_size,64,3), cv2.COLOR_BGR2RGB)
        canvas[:,64*(obs_size+1):,:] = x_query_draw
        # Draw Grid
        cv2.line(canvas, (0,0),(0,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*(obs_size+2)-1,0),(64*(obs_size+2)-1,64*gen_size),(0,0,0), 2)
        cv2.line(canvas, (64*obs_size,0),(64*obs_size,64*gen_size),(255,0,0), 2)
        cv2.line(canvas, (64*(obs_size+1),0),(64*(obs_size+1),64*gen_size),(0,0,255), 2)
        for i in range(1,obs_size):
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
            kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3])).cpu().numpy()
            mse_batch = nn.MSELoss()(x_query_sample, x_query_gt).cpu().numpy()
            lh_query = mse_batch.mean()
            lh_query_var = mse_batch.var()
            lh_record.append(lh_query)
            kl_record.append(kl_query)
        print("\r[Eval %s/%s] MSE: %f| KL: %f"%(str(it).zfill(4), str(len(dataset)).zfill(4), lh_query, kl_query), end="")
    lh_mean = np.array(lh_record).mean()
    lh_mean_var = np.array(lh_query_var).mean()
    kl_mean = np.array(kl_record).mean()
    print("\nMSE =", lh_mean, ", KL =", kl_mean)
    return float(lh_mean), float(kl_mean), lh_record, kl_record

############ Parameter Parsing ############
parser = argparse.ArgumentParser(description='Traning parameters of SR-GQN.')
parser.add_argument('--data_path', nargs='?', type=str, default="../GQN-Datasets-pt/rooms_ring_camera", help='Dataset name.')
parser.add_argument('--frac_train', nargs='?', type=float, default=0.01, help='Fraction of data used for training.')
parser.add_argument('--frac_test', nargs='?', type=float, default=0.01, help='Fraction of data used for testing.')
parser.add_argument('--model_path', nargs='?', type=str, default="rrc" ,help='Model path.')
parser.add_argument('--w', nargs='?', type=int, default=2000 ,help='Number of world cells.')
parser.add_argument('--r', nargs='?', type=int, default=500 ,help='Number of render cells.')
parser.add_argument('--c', nargs='?', type=int, default=128 ,help='Number of concepts.')
args = parser.parse_args()

print("Data path: %s"%(args.data_path))
print("Experiments path: %s"%(args.model_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Number of world cells: %d"%(args.w))
print("Number of render cells: %d"%(args.r))
print("Number of concepts: %d"%(args.c))

############ Dataset ############
path = args.data_path
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Train data: ", len(train_dataset))
print("Test data: ", len(test_dataset))

result_path = args.model_path + "result/"
save_path = args.model_path + "save/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SRGQN(n_wrd_cells=args.w, n_ren_cells=args.r, tsize=args.c, ch=64, vsize=7).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

####
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

gen_size = 10
print("Generate image ...")
for i in range(1,10):
    obs_size = i
    # Train
    fname = result_path + "train_" + "obs" + str(i) + ".png"
    canvas = draw_result(net, train_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)
    # Test
    fname = result_path + "test_" + "obs" + str(i) + ".png"
    canvas = draw_result(net, test_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)

obs_size = 3
for i in range(4):
    fname = result_path + "draw_" + str(i) + ".png"
    canvas = draw_result(net, train_dataset, obs_size, gen_size, steps=i)
    cv2.imwrite(fname, canvas)

##############################
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for it, batch in enumerate(data_loader):
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)
    obs_size = 3
    x_obs = image[0,0].permute(1,2,0).numpy()
    x_query = image[0,1].permute(1,2,0).numpy()
    v_obs = pose[0,0].reshape(-1,7).to(device)
    v_query = pose[0,1].reshape(-1,7).to(device)
    #print(v_obs, v_query)

    view_cell_sim = np.zeros([16,16,128])
    #view_cell_sim = np.zeros([32,32,128])
    spos = (10,9)
    #spos = (20,18)
    mag = 8#15.0
    kernal = np.array([[1,2,1],[2,8,2],[1,2,1]])
    #kernal = np.array([[0,0,0],[0,10,0],[0,0,0]])
    #kernal = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    #kernal = np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])

    for i in range(3):
        view_cell_sim[spos[0]-1:spos[0]+2,spos[1]-1:spos[1]+2,i] = kernal
    view_cell_sim /= mag
    view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,16,16,128).permute(0,3,1,2).to(device)
    #view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,32,32,128).permute(0,3,1,2).to(device)
    rlist = []
    for j in range(1,6):
        routing = net.visualize_routing(view_cell_torch, v_obs, v_query, None, im_size=(16, 16))
        routing = routing.permute(0,2,3,1).detach().cpu().reshape(16,16,128).numpy()
        #routing = routing.permute(0,2,3,1).detach().cpu().reshape(32,32,128).numpy()
        rlist.append(routing)

    img_size = (128,128)
    signal_obs = cv2.resize(view_cell_sim[:,:,0:3], img_size, interpolation=cv2.INTER_NEAREST)
    signal_query = cv2.resize(rlist[3][:,:,0:3], img_size, interpolation=cv2.INTER_NEAREST)
    print(np.max(signal_query))
    signal_query = np.minimum(5*signal_query, 1.0)
    x_obs = cv2.cvtColor(x_obs, cv2.COLOR_BGR2RGB)
    x_obs = cv2.resize(x_obs, img_size, interpolation=cv2.INTER_NEAREST)
    x_query = cv2.cvtColor(x_query, cv2.COLOR_BGR2RGB)
    x_query = cv2.resize(x_query, img_size, interpolation=cv2.INTER_NEAREST)
    x_obs_mask = x_obs * (signal_obs*0.7+0.3)
    x_query_mask = x_query * (signal_query*0.7+0.3)
    #print(np.min(signal_query))

    img_canvas = np.zeros([img_size[0]*2, img_size[0]*3, 3])
    img_canvas[0:img_size[1],0:img_size[0],:] = x_obs
    img_canvas[0:img_size[1],img_size[0]:img_size[0]*2,:] = x_obs_mask
    img_canvas[0:img_size[1],img_size[0]*2:img_size[0]*3,:] = signal_obs
    img_canvas[img_size[1]:img_size[1]*2,0:img_size[0],:] = x_query
    img_canvas[img_size[1]:img_size[1]*2,img_size[0]:img_size[0]*2,:] = x_query_mask
    img_canvas[img_size[1]:img_size[1]*2,img_size[0]*2:img_size[0]*3,:] = signal_query
    fname = result_path + "route_" + str(it).zfill(2) + ".png"
    cv2.imwrite(fname, img_canvas*255)

    '''
    cv2.imshow("signal_obs", signal_obs)
    cv2.imshow("signal_query", signal_query)
    cv2.imshow("x_obs", x_obs)
    cv2.imshow("x_query", x_query)
    cv2.imshow("x_obs_mask", x_obs_mask)
    cv2.imshow("x_query_mask", x_query_mask)
    cv2.imshow("canvas", img_canvas)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    '''

