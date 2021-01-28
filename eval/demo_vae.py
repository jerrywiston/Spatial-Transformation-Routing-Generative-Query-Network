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
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#from srgqn import SRGQN
from srgqn_vae import SRGQN
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

def draw_signal_line(img_map, v, x, fov=50, color=(1,1,1), map_size=240, fill_size=8, line_size=400):#(0,0.8,0.8)
    global view_inverse
    img_map = cv2.flip(img_map, 0)
    pos = (fill_size+int(map_size/2*(1+v[0])), fill_size+int(map_size/2*(1+v[1])))
    ang = np.arctan2(v[4], v[3])
    if view_inverse:
        ang += np.pi
    ang_res = np.pi/2 - np.arctan2(0.5*np.tan(np.deg2rad(90-fov/2)), 0.5-x)
    pix_len = np.pi/2 - np.arctan2(0.5*np.tan(np.deg2rad(90-fov/2)), 1/32)
    pts1 = (int(pos[0] + line_size*np.cos(ang+ang_res+pix_len)), int(pos[1] + line_size*np.sin(ang+ang_res+pix_len)))
    pts2 = (int(pos[0] + line_size*np.cos(ang+ang_res-pix_len)), int(pos[1] + line_size*np.sin(ang+ang_res-pix_len)))
    #
    cv2.circle(img_map, pos, 3, color, 2)
    cv2.line(img_map, pos, pts1, color, 1)
    cv2.line(img_map, pos, pts2, color, 1)
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

def local2global(id, local_pos):
    global img_size
    return (int(img_size[1]/2*(int(id/4)+local_pos[0])),int(img_size[0]/2*(id%4+local_pos[1])))


############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str ,help='Experiment name.')
parser.add_argument("-i", "--view_inverse", help="view inverse", action="store_true")
parser.add_argument("-k", "--keyboard", help="human control", action="store_true")
parser.add_argument("-demo1", "--obs_demo", help="Observation demo", action="store_true")
parser.add_argument("-demo2", "--signal_demo", help="Signal demo", action="store_true")
parser.add_argument("-s", "--save_video", help="Save video", action="store_true")
parser.add_argument("-n", "--noise", help="Apply noise", action="store_true")
parser.add_argument('--obs', nargs='?', type=int, default=4, help='Initial number of observations.')
parser.add_argument('--round', nargs='?', type=int, default=2, help='Number of round per scene.')
parser.add_argument('--signal', nargs='?', type=int, default=40, help='Amplitude of signal.')
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
demo_loop = parser.parse_args().round
human_control = parser.parse_args().keyboard #True#False
view_inverse = parser.parse_args().view_inverse #True#False
obs_demo = parser.parse_args().obs_demo
signal_demo = parser.parse_args().signal_demo
obs_init = parser.parse_args().obs
save_video = parser.parse_args().save_video
noise = parser.parse_args().noise
render = True
signal_pos = {"global":(int(img_size[1]/2*0.5),int(img_size[0]/2*0.7)), "local":(0.5, 0.7), "id":0}#None
signal_amp = parser.parse_args().signal
feat_size = 16
result_path = exp_path + "result/"
if save_video:
    if not os.path.exists(result_path):
        os.makedirs(result_path)

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
            #print(x_local_norm, y_local_norm, id)
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
def demo(x_obs, v_obs, write_path=None):
    global human_control, render, obs_act, demo_loop, signal_pos, feat_size, obs_init, save_video, result_path, obs_demo, signal_demo, signal_amp
    #
    writer = None
    render = True
    x_obs_torch = x_obs.to(device)
    v_obs_torch = v_obs.to(device)
    net.construct_scene_representation(x_obs_torch, v_obs_torch)
    obs_act = np.array([0]*obs_size)
    obs_act[0:min(obs_init,8)] = 1

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
        print("\rStep:", str(step+1).zfill(3), "/", 180*demo_loop ,", Camera Pose:", pos, np.rad2deg(ang), end="")

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
                pos_temp = [pos[0] - np.cos(ang)*0.05, pos[1] - np.sin(ang)*0.05]
                if -1 <= pos_temp[0] <= 1 and -1 <= pos_temp[1] <= 1:
                    render = True
                    pos = pos_temp
            if k == ord('s'):
                pos_temp = [pos[0] + np.cos(ang)*0.05, pos[1] + np.sin(ang)*0.05]
                if -1 <= pos_temp[0] <= 1 and -1 <= pos_temp[1] <= 1:
                    render = True
                    pos = pos_temp
            if k == ord('q'):
                render = True
                ang += np.deg2rad(4)
            if k == ord('e'):
                render = True
                ang -= np.deg2rad(4)
            if k == ord('a'):
                pos_temp = [pos[0] + np.sin(ang)*0.05, pos[1] - np.cos(ang)*0.05]
                if -1 <= pos_temp[0] <= 1 and -1 <= pos_temp[1] <= 1:
                    render = True
                    pos = pos_temp
            if k == ord('d'):
                pos_temp = [pos[0] - np.sin(ang)*0.05, pos[1] + np.cos(ang)*0.05]
                if -1 <= pos_temp[0] <= 1 and -1 <= pos_temp[1] <= 1:
                    render = True
                    pos = pos_temp
            # Switch to Observation Camera
            if ord('1') <= k <= ord('8'):
                render = True
                cid = int(k - 49) 
                ang = np.arctan2(v_obs[cid,4], v_obs[cid,3])
                pos = [float(v_obs[cid,0].numpy()), float(v_obs[cid,1].numpy())]
            # Swith Human Control / Ring Demo
            if k == ord('r'):
                human_control = False
            
        else:
            render = True
            query_ang += 2
            pos = [np.cos(np.deg2rad(query_ang)), np.sin(np.deg2rad(query_ang))]
            if view_inverse:
                ang = np.deg2rad(query_ang)
            else:
                ang = np.deg2rad(180+query_ang)
            if step >= 180*demo_loop:
                break
            if obs_demo:
                progress = int(step / (180*demo_loop) * 8)
                #print(progress)
                for i in range(len(obs_act)):
                    if i <= progress:
                        obs_act[i] = 1
                    else:
                        obs_act[i] = 0
                signal_pos = {"global":local2global(progress, (0.5,0.7)), "local":(0.5, 0.7), "id":progress}
            if signal_demo:
                progress = int(step / (180*demo_loop) * 4)
                progress2 = int(step / (180*demo_loop) * 12)
                if progress2%3 == 0:
                    feat_size = 16
                elif progress2%3 == 1:
                    feat_size = 32
                elif progress2%3 == 2:
                    feat_size = 64

                if step / (180*demo_loop) * 4 == round(step / (180*demo_loop) * 4):
                    spos = np.random.rand(2)
                    spos[0] = spos[0]*0.8+0.1
                    spos[1] = spos[1]*0.25+0.65
                    id = int(progress)
                    signal_pos = {"global":local2global(id, spos), "local":spos, "id":id}
            step += 1
            k = cv2.waitKey(1)
            # Swith Human Control / Ring Demo
            if k == ord('r'):
                human_control = True

        # Re-render
        if k == 13:
            render = True
        # Next / Break
        if k == 32:
            break
        if k == 27:
            exit()
        # Remove signal
        if k == ord('f'):
            signal_pos = None
            render = True 

        ##################################################
        # Render
        if render:
            render = False
            v_query = np.array([pos[0], pos[1], 0, np.cos(ang), np.sin(ang), 1, 0])
            # Network Forward
            v_query_torch = torch.FloatTensor(v_query).unsqueeze(0)
            x_query = net.scene_render(v_query_torch.to(device), obs_act, noise)
            x_query = x_query[0].detach().cpu()
            x_query_view = cv2.cvtColor(x_query.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
            x_query_view = cv2.resize(x_query_view, img_size, interpolation=cv2.INTER_NEAREST)
            
            # Draw Signal
            signal_query = np.zeros_like(x_query_view)
            if signal_pos is not None:
                std = 0.03#0.04
                hp = gaussian_heatmap(signal_pos["local"], std, (feat_size,feat_size,args.c))
                view_cell_sim = hp / np.max(hp) * signal_amp
                view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,feat_size,feat_size,args.c).permute(0,3,1,2).to(device)
                routing = net.visualize_routing(view_cell_torch, v_obs_torch[signal_pos["id"]].unsqueeze(0), v_query_torch.to(device), view_size=(feat_size, feat_size))
                routing = torch.clamp(routing, min=0, max=1)
                routing = routing.permute(0,2,3,1).detach().cpu().reshape(feat_size,feat_size,args.c).numpy()[:,:,0:3]
                signal_query = cv2.resize(routing, img_size, interpolation=cv2.INTER_NEAREST)
            
            # Signal Masked Image
            color = (0.8,0.8,0.8)
            x_signal_view = x_query_view * (signal_query*0.7+0.3)    
            cv2.rectangle(x_signal_view, (0,0), img_size, color, 12)
            cv2.putText(x_signal_view, "Signal Masked Image", (10,24), cv2.FONT_HERSHEY_TRIPLEX , 0.6, color, 1, cv2.LINE_AA)
            
            # Signal Image
            color = (0.6,0.6,0.6)
            ren_text = "Query Signal " + str(feat_size) + "x" + str(feat_size)
            cv2.rectangle(signal_query, (0,0), img_size, color, 12)
            cv2.putText(signal_query, ren_text, (10,24), cv2.FONT_HERSHEY_TRIPLEX , 0.6, color, 1, cv2.LINE_AA)
            x_signal_canvas = cv2.vconcat([x_signal_view,signal_query])

            # Draw Query Image 
            cv2.putText(x_query_view, "Render", (10,24), cv2.FONT_HERSHEY_TRIPLEX , 0.6, (0,0,1), 1, cv2.LINE_AA)
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

            img_map_cam = img_map.copy()
            # Draw Observation Cameras
            for i in range(obs_act.shape[0]):
                c = int(255*(0.8/obs_act.shape[0]*i+0.1)) * np.array([1,1], dtype=np.uint8)
                color =  cv2.applyColorMap(c, cv2.COLORMAP_VIRIDIS)[0,0] / 255.0
                if obs_act[i] == 1: 
                    img_map_cam = draw_camera(img_map_cam, v_obs[i].numpy(), color=color, line_size=30, center_line=False)
            
            # Signal Line
            if signal_pos is not None:
                img_map_cam = draw_signal_line(img_map_cam, v_obs[signal_pos['id']], signal_pos['local'][0])

            # Draw Query Camera
            img_map_cam = draw_camera(img_map_cam.copy(), v_query, color=(0,0,1))
            cv2.rectangle(img_map_cam, (0,0), img_size, (0.4,0.4,0.4), 12)
            cv2.putText(img_map_cam, "Cam", (10,25), cv2.FONT_HERSHEY_TRIPLEX , 0.6, (0.4,0.4,0.4), 1, cv2.LINE_AA)

            # Combine Blocks
            view_canvas = cv2.vconcat([x_query_view.astype(np.float32), img_map_cam.astype(np.float32)])
            view_canvas = cv2.hconcat([x_obs_canvas, view_canvas, x_signal_canvas])
            if signal_pos is not None:
                cv2.circle(view_canvas, signal_pos['global'], 5, (1,1,1), 3)
                cv2.circle(view_canvas, signal_pos['global'], 8, (0,0,0), 2)
                cv2.circle(view_canvas, signal_pos['global'], 3, (0,0,0), 2)
                #cv2.circle(view_canvas, signal_pos['global'], 5, (0,1,1), 3)
            cv2.imshow("View", view_canvas)
            if save_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    size = (view_canvas.shape[1], view_canvas.shape[0])
                    writer = cv2.VideoWriter(path,fourcc, 20.0, size)
                frame = (view_canvas*255).astype(np.uint8)
                writer.write(frame)
    print()
    if save_video:
        writer.release()

obs_act = np.array([0]*obs_size)
obs_act[0:3] = 1
if signal_demo:
    demo_loop = 4
for it, batch in enumerate(data_loader):
    print("[ Data", it+1, "]")
    print("Press BACKSPACE to skip the data or others to run ...")
    k = cv2.waitKey(0)
    if k == 8:
        print("Skip data", it+1)
        continue
    if k == 27:
        exit()
    # Get Data
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)    
    # Demo
    for bit in range(image.shape[0]):
        print("[ Data", it+1, "| Batch", bit+1, "/", image.shape[0], "]")
        x_obs = image[bit,:obs_size]
        v_obs = pose[bit,:obs_size].reshape(-1,7)
        path = result_path+"data_"+str(it+1).zfill(2)+"_batch_"+str(bit+1).zfill(2)+".avi"
        if obs_demo == True:
            path = result_path+"demo1_data_"+str(it+1).zfill(2)+"_batch_"+str(bit+1).zfill(2)+".avi"
        if signal_demo == True:
            path = result_path+"demo2_data_"+str(it+1).zfill(2)+"_batch_"+str(bit+1).zfill(2)+".avi"
        demo(x_obs, v_obs, write_path=path)
