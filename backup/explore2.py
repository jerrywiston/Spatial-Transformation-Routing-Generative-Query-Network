###########################################
#      Keyboard Control Instruction       #
#-----------------------------------------#
# R: Switch <Auto Demo>/<Human Control>.  #
# W/A/S/D: Move.                          #
# Q/E: Turn perspective.                  #
# Z/C: Around the ring.                   #
# 1~8: Move to observation pose.          #
# SPACE: Next data.                       #
# ENTER: Re-render the canvas.            # 
###########################################

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
np.set_printoptions(precision=3)

def draw_camera(img_map, v, color=(1,0,0), fov=50, map_size=240, fill_size=8, line_size=400, center_line=True):
    global view_inverse
    img_map = cv2.flip(img_map, 0)
    pos = (fill_size+int(map_size/2*(1+v[0])), fill_size+int(map_size/2*(1+v[1])))
    ang = np.arctan2(v[4], v[3])
    if view_inverse:
        ang += np.pi
    pts1 = (int(pos[0] + line_size*np.cos(ang)), int(pos[1] + line_size*np.sin(ang)))
    pts2 = (int(pos[0] + line_size*np.cos(ang+np.deg2rad(fov/2))), int(pos[1] + line_size*np.sin(ang+np.deg2rad(fov/2))))
    pts3 = (int(pos[0] + line_size*np.cos(ang-np.deg2rad(fov/2))), int(pos[1] + line_size*np.sin(ang-np.deg2rad(fov/2))))
    #
    cv2.circle(img_map, pos, 5, color, 3)
    if center_line:
        cv2.line(img_map, pos, pts1, color, 1)
    else:
        cv2.line(img_map, pts2, pts3, color, 2)
    cv2.line(img_map, pos, pts2, color, 2)
    cv2.line(img_map, pos, pts3, color, 2)
    #
    img_map = cv2.flip(img_map, 0)
    return img_map

def gaussian_heatmap(mean, std, size):
    img = np.zeros(size, dtype=np.float32)
    mean_pix = (mean[0]*size[0], mean[1]*size[1])
    std_pix = std * size[0]
    for i in range(size[1]):
        for j in range(size[0]):
            temp = ((i-mean_pix[0])**2 + (j-mean_pix[1])**2)/std_pix**2
            img[j,i,:] = np.exp(-0.5 * temp) / (2*np.pi*std_pix**2)
    return img

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
parser.add_argument("-i", "--view_inverse", help="view inverse", action="store_true")
parser.add_argument("-k", "--keyboard", help="human control", action="store_true")
parser.add_argument("-a", "--auto_demo", help="Auto demo", action="store_true")
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

############ Parameters ############
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
img_size = (256,256)
fov = 50
map_size = 240
fill_size = 8
obs_size = 8
demo_loop = 2
human_control = parser.parse_args().keyboard #True#False
view_inverse = parser.parse_args().view_inverse #True#False
obs_increase = parser.parse_args().auto_demo
render = True
signal_pos = None
feat_size = 16

############ Events ############
def onMouse(event, x, y, flags, param):
    global obs_act, img_size, render, signal_pos, feat_size
    obs_size = (int(img_size[0]/2), int(img_size[1]/2))
    if event == cv2.EVENT_LBUTTONDOWN:
        idxy = (int(x/obs_size[1]), int(y/obs_size[0]))
        id = 4*idxy[0] + idxy[1]
        #print(id)
        if id < 8:
            render = True
            if obs_act[id] == 0:
                obs_act[id] = 1
            else:
                obs_act[id] = 0
            #print(obs_act)
    if event == cv2.EVENT_RBUTTONDOWN:
        idxy = (int(x/img_size[1]*2), int(y/img_size[0]*2))
        id = 4*idxy[0] + idxy[1]
        x_local = x % obs_size[1]
        y_local = y % obs_size[0]
        if obs_size[1]*0.05 < x_local < obs_size[1]*0.95 and \
            obs_size[0]*0.05 < y_local < obs_size[0]*0.95 and id < 8:
            x_local_norm = x_local / obs_size[1]
            y_local_norm = y_local / obs_size[0]
            print(x_local_norm, y_local_norm, id)
            signal_pos = {"global":(x,y), "local":(x_local_norm, y_local_norm), "id":id}
            render = True
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            #print("up")
            if feat_size < 64:
                feat_size *= 2
                render = True
        else:
            #print("down")
            if feat_size > 8:
                feat_size = int(feat_size/2)
                render = True
            
cv2.namedWindow('View')
cv2.setMouseCallback('View', onMouse)

############ Main ############
def demo(x_obs, v_obs):
    global human_control, render, obs_act, demo_loop, signal_pos, feat_size, ent_queue
    render = True
    x_obs_torch = x_obs.to(device)
    v_obs_torch = v_obs.to(device)
    net.construct_scene_representation(x_obs_torch, v_obs_torch)
    obs_act = np.array([0]*obs_size)
    obs_act[0:1] = 1

    # Map
    img_map = 0.0*np.ones((map_size+2*fill_size, map_size+2*fill_size,3))
    center_pos = (int(map_size/2+fill_size), int(map_size/2+fill_size))
    cv2.circle(img_map, center_pos, int(map_size/2), (0,1,0), 1)
    
    # Initialize Pose
    query_ang = np.rad2deg(np.arctan2(v_obs[0,1], v_obs[0,0]))
    if human_control:
        ang = np.arctan2(v_obs[0,4], v_obs[0,3])
        pos = [float(v_obs[0,0].numpy()), float(v_obs[0,1].numpy())]
    else:
        pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
        if view_inverse:
            ang = np.deg2rad(query_ang)
        else:
            ang = np.deg2rad(180+query_ang)
    
    step = 0
    while(True):
        # Query Pose
        v_query = np.array([pos[0], pos[1], 0, np.cos(ang), np.sin(ang), 1, 0])
        print("\rStep:", str(step).zfill(3), "/", 180*demo_loop+1 ,", Camera Pose:", pos, np.rad2deg(ang), end="")

        # Render
        if render:
            render = False
            # Network Forward
            v_query_torch = torch.FloatTensor(v_query).unsqueeze(0)
            x_query = net.scene_render(v_query_torch.to(device), obs_act)
            x_query = x_query[0].detach().cpu()
            x_query_view = cv2.cvtColor(x_query.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
            x_query_view = cv2.resize(x_query_view, img_size, interpolation=cv2.INTER_NEAREST)
            
            # Draw Signal
            if signal_pos is not None:
                std = 0.04
                hp = gaussian_heatmap(signal_pos["local"], std, (feat_size,feat_size,args.c))
                view_cell_sim = hp / np.max(hp) * 50
                view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,feat_size,feat_size,args.c).permute(0,3,1,2).to(device)
                routing = net.visualize_routing(view_cell_torch, v_obs_torch[signal_pos["id"]].unsqueeze(0), v_query_torch.to(device), view_size=(feat_size, feat_size))
                routing = routing.permute(0,2,3,1).detach().cpu().reshape(feat_size,feat_size,args.c).numpy()[:,:,0:3]
                signal_query = cv2.resize(routing, img_size, interpolation=cv2.INTER_NEAREST)
                x_query_view = x_query_view * (signal_query*0.75+0.25)

            # Draw Query Image
            ren_text = "Render"
            if signal_pos is None:
                ren_text = "Render"
            else:
                ren_text = "Query Signal " + str(feat_size) + "x" + str(feat_size)
            cv2.putText(x_query_view, ren_text, (10,24), cv2.FONT_HERSHEY_TRIPLEX , 0.6, (0,0,1), 1, cv2.LINE_AA)
            cv2.rectangle(x_query_view, (0,0), img_size, (0,0,1), 12)

            # Draw Observation Images
            x_obs_canvas = 0.2*np.ones([img_size[0]*2, img_size[1], 3], dtype=np.float32)
            for i in range(x_obs.shape[0]):
                osize = (int(img_size[0]/2), int(img_size[1]/2))
                c = int(255*(0.8/x_obs.shape[0]*i+0.1)) * np.array([1,1], dtype=np.uint8)
                color =  cv2.applyColorMap(c, cv2.COLORMAP_VIRIDIS)[0,0] / 255.0
                    
                x_obs_view = cv2.cvtColor(x_obs[i].permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
                x_obs_view = cv2.resize(x_obs_view, osize, interpolation=cv2.INTER_NEAREST)
                if obs_act[i] == 0:
                    x_obs_view *= 0.2
                cv2.rectangle(x_obs_view, (0,0), osize, color, 8)
                text = "Obs" + str(i+1)
                cv2.putText(x_obs_view, text, (5,20), cv2.FONT_HERSHEY_TRIPLEX , 0.6, color, 1, cv2.LINE_AA)
                x_obs_canvas[(i%4)*osize[0]:(i%4+1)*osize[0], int(i/4)*osize[1]:int(i/4+1)*osize[1]] = x_obs_view

            # Draw Observation Cameras
            img_map_cam = img_map.copy()
            for i in range(obs_act.shape[0]):
                c = int(255*(0.8/obs_act.shape[0]*i+0.1)) * np.array([1,1], dtype=np.uint8)
                color =  cv2.applyColorMap(c, cv2.COLORMAP_VIRIDIS)[0,0] / 255.0
                if obs_act[i] == 1: 
                    img_map_cam = draw_camera(img_map_cam, v_obs[i].numpy(), color=color, line_size=30, center_line=False)
            
            # Draw Query Camera
            img_map_cam = draw_camera(img_map_cam.copy(), v_query, color=(0,0,1))
            cv2.rectangle(img_map_cam, (0,0), img_size, (0.4,0.4,0.4), 12)
            cv2.putText(img_map_cam, "Cam", (10,25), cv2.FONT_HERSHEY_TRIPLEX , 0.6, (0.4,0.4,0.4), 1, cv2.LINE_AA)
            view_canvas = cv2.vconcat([x_query_view.astype(np.float32), img_map_cam.astype(np.float32)])
            view_canvas = cv2.hconcat([x_obs_canvas, view_canvas])
            if signal_pos is not None:
                cv2.circle(view_canvas, signal_pos['global'], 5, (0,1,1), 3)
            cv2.imshow("View", view_canvas)
        
        ########################################################
        # View Control
        if human_control:
            k = cv2.waitKey(10)
            # Ring Control
            if k == ord('z'):
                render = True
                query_ang -= 2
                query_ang = query_ang % 360
                pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
                if view_inverse:
                    ang = np.deg2rad(query_ang)
                else:
                    ang = np.deg2rad(180+query_ang)
            if k == ord('c'):
                render = True
                query_ang += 2
                query_ang = query_ang % 360
                pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
                if view_inverse:
                    ang = np.deg2rad(query_ang)
                else:
                    ang = np.deg2rad(180+query_ang)
            # Move Control
            if k == ord('w'):
                render = True
                pos[0] -= np.cos(ang) * 0.05
                pos[1] -= np.sin(ang) * 0.05
            if k == ord('s'):
                render = True
                pos[0] += np.cos(ang) * 0.05
                pos[1] += np.sin(ang) * 0.05
            if k == ord('q'):
                render = True
                ang += np.deg2rad(4)
            if k == ord('e'):
                render = True
                ang -= np.deg2rad(4)
            if k == ord('a'):
                render = True
                pos[0] += np.sin(ang) * 0.05
                pos[1] -= np.cos(ang) * 0.05
            if k == ord('d'):
                render = True
                pos[0] -= np.sin(ang) * 0.05
                pos[1] += np.cos(ang) * 0.05
            # Switch to Observation Camera
            if ord('1') <= k <= ord('8'):
                render = True
                cid = int(k - 49) 
                ang = np.arctan2(v_obs[cid,4], v_obs[cid,3])
                pos = [float(v_obs[cid,0].numpy()), float(v_obs[cid,1].numpy())]
            # Re-render
            if k == 13:
                render = True
            # Swith Human Control / Ring Demo
            if k == ord('r'):
                human_control = False
        else:
            render = True
            step += 1
            query_ang += 2
            pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
            if view_inverse:
                ang = np.deg2rad(query_ang)
            else:
                ang = np.deg2rad(180+query_ang)
            if step > 180*demo_loop+1:
                break
            if obs_increase:
                progress = int(step / (180*demo_loop) * 8)
                #print(progress)
                for i in range(len(obs_act)):
                    if i <= progress:
                        obs_act[i] = 1
            k = cv2.waitKey(1)
            # Swith Human Control / Ring Demo
            if k == ord('r'):
                human_control = True
        # Next / Break
        if k == 32:
            break
        if k == 27:
            exit()
        if k == ord('f'):
            signal_pos = None
            render = True
    print()

obs_act = np.array([0]*obs_size)
obs_act[0:3] = 1
#print("[ Press any button to start ]")
#cv2.waitKey(0)
for it, batch in enumerate(data_loader):
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)    
    for bit in range(image.shape[0]):
        print("[ Data", it+1, "| Batch", bit+1, "]")
        x_obs = image[bit,:obs_size]
        v_obs = pose[bit,:obs_size].reshape(-1,7)
        demo(x_obs, v_obs)
    break