import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def draw_query(net, dataset, obs_size=3, vsize=7, row_size=32, gen_size=10, img_size=(64,64), border=[1,4,3], shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    draw_counter = 0
    row_counter = 0
    img_list = []
    img_row = []
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,img_size[0],img_size[1]).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,vsize).to(device)
        v_query = pose[:,-1].to(device)
        x_query_gt = image[:,-1].to(device)
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            x_query_sample = x_query_sample.detach().permute(0,2,3,1).cpu().numpy()
        
        for j in range(image.shape[0]):
            x_np = []
            bscale = int(img_size[1]/64)
            for i in range(obs_size):
                x_np.append(image[:,i].detach().permute(0,2,3,1).cpu().numpy()[j])
                if i < obs_size-1:
                    x_np.append(np.ones([img_size[0],border[0]*bscale,3]))
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_gt.detach().permute(0,2,3,1).cpu().numpy()[j])
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_sample[j])
            x_np = np.concatenate(x_np, 1)
            img_row.append(x_np)
            row_counter += 1

            if row_counter % row_size == 0:
                img_row = np.concatenate(img_row, 0) * 255
                img_row = cv2.cvtColor(img_row.astype(np.uint8), cv2.COLOR_BGR2RGB)
                img_list.append(img_row)
                draw_counter += 1
                fill_size = len(str(gen_size))
                print("\rProgress: "+str(draw_counter).zfill(fill_size)+"/"+str(gen_size), end="")
                img_row = []
            elif row_counter % row_size < row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))

            if draw_counter >= gen_size:
                print()
                return img_list