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
train_dataset = GqnDatasets(root_dir=path, train=True, fraction=args.frac_train)
test_dataset = GqnDatasets(root_dir=path, train=False, fraction=args.frac_test)
print("Data path: %s"%(args.data_path))
print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
print("Train data: ", len(train_dataset))
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

####
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

gen_size = 10
print("Generate image ...")
for i in range(1,6):
    obs_size = i
    # Train
    fname = result_path + "train_" + "obs" + str(i) + ".png"
    canvas = draw_result(net, train_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)
    # Test
    fname = result_path + "test_" + "obs" + str(i) + ".png"
    canvas = draw_result(net, test_dataset, obs_size, gen_size)
    cv2.imwrite(fname, canvas)

##############################
# Routing Visualization
##############################
def gaussian_heatmap(mean, std, size):
    img = np.zeros(size, dtype=np.float32)
    mean_pix = (mean[0]*size[0], mean[1]*size[1])
    std_pix = std * size[0]
    for i in range(size[1]):
        for j in range(size[0]):
            temp = ((i-mean_pix[0])**2 + (j-mean_pix[1])**2)/std_pix**2
            img[j,i,:] = np.exp(-0.5 * temp) / (2*np.pi*std_pix**2)
    return img

def drawlines(img,lines,pts):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img.shape
    img = cv2.cvtColor(img[:,:,0],cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img,0)
    for r,pt in zip(lines,pts):
        color = tuple([0,0,255])
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
    img = cv2.flip(img,0)
    return img

def crossmat(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]], dtype=vec.dtype)

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
    
    feat_size = 16
    std = 0.04#0.05
    spos = (0.5,0.7)#(0.3,0.7)#(0.5,0.8)
    hp = gaussian_heatmap(spos, std, (feat_size,feat_size,args.c))
    view_cell_sim = hp / np.max(hp) * 1.5 #np.log(hp/(1-hp)+1e-10)
    print(view_cell_sim.max(), view_cell_sim.min())
    view_cell_torch = torch.FloatTensor(view_cell_sim).reshape(1,feat_size,feat_size,args.c).permute(0,3,1,2).to(device)
    rlist = []
    for j in range(1,6):
        routing = net.visualize_routing(view_cell_torch, v_obs, v_query, None, view_size=(feat_size, feat_size))
        routing = routing.permute(0,2,3,1).detach().cpu().reshape(feat_size,feat_size,args.c).numpy()
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
    
    #cv2.imwrite(fname, img_canvas*255)
    '''
    cv2.imshow("signal_obs", signal_obs)
    cv2.imshow("signal_query", signal_query)
    cv2.imshow("x_obs", x_obs)
    cv2.imshow("x_query", x_query)
    cv2.imshow("x_obs_mask", x_obs_mask)
    cv2.imshow("x_query_mask", x_query_mask)
    '''

    #############################
    # Epipolar Line
    #############################
    fov = 50
    image = batch[0].squeeze(0)
    pose = batch[1].squeeze(0)
    obs_size = 3
    x_obs = image[0,0].permute(1,2,0).numpy()
    x_query = image[0,1].permute(1,2,0).numpy()
    v_obs = pose[0,0].reshape(-1,7)
    v_query = pose[0,1].reshape(-1,7)
    print(v_obs[0], v_query[0])
    
    map_size = 240
    fill_size = 8
    line_size = 400
    img_map = np.zeros((map_size+2*fill_size, map_size+2*fill_size,3))
    # Observation
    obs_pos = (fill_size+int(map_size/2*(1+v_obs[0,0])), fill_size+int(map_size/2*(1+v_obs[0,1])))
    obs_ang = np.arctan2(v_obs[0,4], v_obs[0,3])
    obs_line1 = (int(obs_pos[0] + line_size*np.cos(obs_ang)), int(obs_pos[1] + line_size*np.sin(obs_ang)))
    obs_line2 = (int(obs_pos[0] + line_size*np.cos(obs_ang+np.deg2rad(fov/2))), int(obs_pos[1] + line_size*np.sin(obs_ang+np.deg2rad(fov/2))))
    obs_line3 = (int(obs_pos[0] + line_size*np.cos(obs_ang-np.deg2rad(fov/2))), int(obs_pos[1] + line_size*np.sin(obs_ang-np.deg2rad(fov/2))))
    # Query
    query_pos = (fill_size+int(map_size/2*(1+v_query[0,0])), fill_size+int(map_size/2*(1+v_query[0,1])))
    query_ang = np.arctan2(v_query[0,4], v_query[0,3])
    query_line1 = (int(query_pos[0] + line_size*np.cos(query_ang)), int(query_pos[1] + line_size*np.sin(query_ang)))
    query_line2 = (int(query_pos[0] + line_size*np.cos(query_ang+np.deg2rad(fov/2))), int(query_pos[1] + line_size*np.sin(query_ang+np.deg2rad(fov/2))))
    query_line3 = (int(query_pos[0] + line_size*np.cos(query_ang-np.deg2rad(fov/2))), int(query_pos[1] + line_size*np.sin(query_ang-np.deg2rad(fov/2))))
    print("ang", np.rad2deg(obs_ang), np.rad2deg(query_ang))
    ##
    center_pos = (int(map_size/2+fill_size), int(map_size/2+fill_size))
    cv2.circle(img_map, center_pos, int(map_size/2), (0,1,0), 1)
    #
    cv2.circle(img_map, obs_pos, 5, (1,0,0), 3)
    cv2.line(img_map, obs_pos, obs_line1, (1,0,0), 1)
    cv2.line(img_map, obs_pos, obs_line2, (1,0,0), 2)
    cv2.line(img_map, obs_pos, obs_line3, (1,0,0), 2)
    #
    cv2.circle(img_map, query_pos, 5, (0,0,1), 3)
    cv2.line(img_map, query_pos, query_line1, (0,0,1), 1)
    cv2.line(img_map, query_pos, query_line2, (0,0,1), 2)
    cv2.line(img_map, query_pos, query_line3, (0,0,1), 2)
    img_map = cv2.flip(img_map, 0)
    cv2.imshow("map", img_map)

    fname = result_path + "route_map_" + str(it).zfill(2) + ".png"
    cv2.imwrite(fname, img_map*255)

    # Draw Epipolar Line
    f = feat_size / np.tan(np.deg2rad(fov/2)) /2
    cx, cy = (feat_size-1)/2, (feat_size-1)/2
    cam_int = np.array([[f,0,cx],
                        [0,f,cy],
                        [0,0,1]])
    print("Camera Int:")
    print(cam_int)
    #
    pih = -np.pi/2
    M1 = np.array([ [np.cos(-obs_ang+pih), 0, np.sin(-obs_ang+pih), v_obs[0,0]],
                    [ 0, 1, 0, v_obs[0,2]],
                    [-np.sin(-obs_ang+pih), 0, np.cos(-obs_ang+pih), v_obs[0,1]],
                    [0,0,0,1]])
    M2 = np.array([ [np.cos(-query_ang+pih), 0, np.sin(-query_ang+pih), v_query[0,0]],
                    [ 0, 1, 0, v_query[0,2]],                
                    [-np.sin(-query_ang+pih), 0, np.cos(-query_ang+pih), v_query[0,1]],
                    [0,0,0,1]])

    #Rel = np.linalg.inv(M1) @ M2
    Rel = np.linalg.inv(M2) @ M1
    #Rel = M1 @ np.linalg.inv(M2) 
    #Rel = M2 @ np.linalg.inv(M1) 
    print("Relative POse:")
    print(Rel)
    print("Relative Angle:")
    print(np.rad2deg(np.arctan2(Rel[1,0],Rel[0,0])))
    
    R = Rel[:3,:3]
    t = Rel[:3,3]
    print(R,t)
    E = (crossmat(t).dot(R)).T
    F = (np.linalg.inv(cam_int).T).dot(E).dot(np.linalg.inv(cam_int))
    print("F:")
    pt = np.array([(spos[0])*feat_size, (spos[1])*feat_size, 1])
    print(F)
    FX = F @ pt
    FX = (FX/np.sqrt(FX[0]**2+FX[1]**2)).reshape(-1,3)
    print(FX)
    ###
    ptt = np.array([0.5*feat_size,1*feat_size,1])
    ptt = np.linalg.inv(cam_int).dot(ptt)
    print(ptt)
    print(np.tan(np.deg2rad(25)))
    
    pts = np.array([(spos[0])*feat_size, (1-spos[1])*feat_size])
    line = cv2.computeCorrespondEpilines(pts.reshape(-1,1,2), 2, F)
    line = line.reshape(-1,3)
    print("Line:", line)
    print(np.rad2deg(np.arctan2(line[0,1],line[0,0])))

    epi_img2 = 4*rlist[3][:,:,0:3]
    #epi_img2 = np.zeros((feat_size, feat_size, 3), dtype=np.float32)
    epi_img2 = drawlines(epi_img2,line,pts)
    epi_img2 = cv2.resize(epi_img2, img_size, interpolation=cv2.INTER_NEAREST)
    
    epi_img1 = signal_obs.copy()
    #epi_img1 = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
    cv2.circle(epi_img1, (int(img_size[0]*spos[0]),int(img_size[1]*spos[1])), int(1.5*std*img_size[0]), (0,0,1) , 6)
    epi_img = cv2.vconcat([epi_img1, epi_img2])

    img_canvas = img_canvas.astype(np.float32)
    img_canvas = cv2.hconcat([img_canvas, epi_img])
    fname = result_path + "route_" + str(it).zfill(2) + ".png"
    cv2.imwrite(fname, img_canvas*255)
    cv2.imshow("canvas", img_canvas)

    k = cv2.waitKey(0)
    if k == ord('q'):
        break
