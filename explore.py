import numpy as np
import cv2
import argparse
import parse_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from srgqn import SRGQN
from dataset import GqnDatasets

def draw_camera(img_map, v, color=(1,0,0), fov=50, map_size=240, fill_size=8, line_size=400):
    img_map = cv2.flip(img_map, 0)
    pos = (fill_size+int(map_size/2*(1+v[0])), fill_size+int(map_size/2*(1+v[1])))
    ang = np.arctan2(v[4], v[3])
    line1 = (int(pos[0] + line_size*np.cos(ang)), int(pos[1] + line_size*np.sin(ang)))
    line2 = (int(pos[0] + line_size*np.cos(ang+np.deg2rad(fov/2))), int(pos[1] + line_size*np.sin(ang+np.deg2rad(fov/2))))
    line3 = (int(pos[0] + line_size*np.cos(ang-np.deg2rad(fov/2))), int(pos[1] + line_size*np.sin(ang-np.deg2rad(fov/2))))
    #
    cv2.circle(img_map, pos, 5, color, 3)
    cv2.line(img_map, pos, line1, color, 1)
    cv2.line(img_map, pos, line2, color, 2)
    cv2.line(img_map, pos, line3, color, 2)
    #
    img_map = cv2.flip(img_map, 0)
    return img_map

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
exp_path = parser.parse_args().path
save_path = exp_path + "save/"
args = parse_config.load_eval_config(exp_path)
print(exp_path)

# Print 
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
#train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Test data: ", len(test_dataset))

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=7, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(save_path+"srgqn.pth"))
net.eval()

####
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
img_size = (256,256)
fov = 50
map_size = 240
fill_size = 8
line_size = 400
obs_size = 6

for it, batch in enumerate(data_loader):
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)
    x_obs = image[0,:obs_size]
    v_obs = pose[0,:obs_size].reshape(-1,7)
    net.construct_scene_representation(x_obs.to(device), v_obs.to(device))
    print(v_obs)
    
    # Map
    img_map = 0.0*np.ones((map_size+2*fill_size, map_size+2*fill_size,3))
    center_pos = (int(map_size/2+fill_size), int(map_size/2+fill_size))
    cv2.circle(img_map, center_pos, int(map_size/2), (0,1,0), 1)

    x_obs_canvas = 0.2*np.ones([img_size[0]*2, img_size[1], 3], dtype=np.float32)
    for i in range(x_obs.shape[0]):
        osize = (int(img_size[0]/2), int(img_size[1]/2))
        c = int(255*(1.0/x_obs.shape[0]*i+0.0)) * np.array([1,1], dtype=np.uint8)
        color =  cv2.applyColorMap(c, cv2.COLORMAP_VIRIDIS)[0,0] / 255.0
        
        x_view = cv2.cvtColor(x_obs[i].permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
        x_view = cv2.resize(x_view, osize, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(x_view, (0,0), osize, color, 8)
        x_obs_canvas[(i%4)*osize[0]:(i%4+1)*osize[0], int(i/4)*osize[1]:int(i/4+1)*osize[1]] = x_view
        #cv2.imshow("x_obs_"+str(i), x_view)
        # Draw Map        
        img_map = draw_camera(img_map, v_obs[i].numpy(), color=color)

    #cv2.imshow("x_obs", x_obs_canvas)
    query_ang = np.rad2deg(np.arctan2(v_obs[0,1], v_obs[0,0]))
    pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
    ang = np.deg2rad(180+query_ang)
    while(True):
        # Query
        print("\r", pos, np.rad2deg(ang), end="")
        v_query = np.array([pos[0], pos[1], 0, np.cos(ang), np.sin(ang), 1, 0])

        # Network Forward
        v_query_torch = torch.FloatTensor(v_query).unsqueeze(0)
        x_query = net.scene_render(v_query_torch.to(device))
        x_query = x_query[0].detach().cpu()
        
        x_view = cv2.cvtColor(x_query.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
        x_view = cv2.resize(x_view, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(x_view, (0,0), img_size, (0,0,1), 12)
        #cv2.imshow("x_query", x_view)

        img_map_curr = draw_camera(img_map.copy(), v_query, color=(0,0,1))
        cv2.rectangle(img_map_curr, (0,0), img_size, (0.3,0.3,0.3), 12)
        #cv2.imshow("map", img_map_curr)
        view_canvas = cv2.vconcat([x_view.astype(np.float32), img_map_curr.astype(np.float32)])
        view_canvas = cv2.hconcat([x_obs_canvas, view_canvas])
        cv2.imshow("View", view_canvas)

        k = cv2.waitKey(0)
        if k == ord('q'):
            query_ang -= 2
            query_ang = query_ang % 360
            pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
            ang = np.deg2rad(180+query_ang)
        if k == ord('e'):
            query_ang += 2
            query_ang = query_ang % 360
            pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
            ang = np.deg2rad(180+query_ang)
        '''
        if k == ord('a'):
            pos[0] -= 0.1
        if k == ord('d'):
            pos[0] += 0.1
        if k == ord('s'):
            pos[1] += 0.1
        if k == ord('w'):
            pos[1] -= 0.1
        '''
        if k == 32:
            break
        if k == 27:
            exit()
    print()

