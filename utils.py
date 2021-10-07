import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dataset
import dataset_shapenet
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ Draw Results ############
def draw_query(net, dataset, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,1]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    draw_counter = 0
    row_counter = 0
    img_list = []
    img_row = []
    for it, batch in enumerate(data_loader):
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        img_size = (image.shape[-2], image.shape[-1])
        vsize = pose.shape[-1]
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
            elif row_counter % row_size <= row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))

            if draw_counter >= gen_size:
                print()
                return img_list

def draw_query_shapenet(net, dataset, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,3]):
    img_list = []
    for it in range(gen_size):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = dataset_shapenet.get_batch(dataset, obs_size, row_size)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            x_query_sample = x_query_sample.detach().permute(0,2,3,1).cpu().numpy()
        
        for j in range(x_query_sample.shape[0]):
            x_np = []
            bscale = int(img_size[1]/64)
            for i in range(obs_size):
                x_np.append(x_obs[obs_size*j+i].detach().permute(1,2,0).cpu().numpy())
                if i < obs_size-1:
                    x_np.append(np.ones([img_size[0],border[0]*bscale,3]))
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_gt.detach().permute(0,2,3,1).cpu().numpy()[j])
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_sample[j])
            x_np = np.concatenate(x_np, 1)
            img_row.append(x_np)

            if j < row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))
        
        img_row = np.concatenate(img_row, 0) * 255
        img_row = cv2.cvtColor(img_row.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_list.append(img_row)
        fill_size = len(str(gen_size))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(gen_size), end="")
    print()
    return img_list

############ Evaluation ############
def eval(net, dataset, obs_size=3, max_batch=1000, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    rmse_record = []
    mae_record = []
    ce_record = []
    for it, batch in enumerate(data_loader):
        if it+1 > max_batch:
            break
        image = batch[0].squeeze(0)
        pose = batch[1].squeeze(0)
        img_size = (image.shape[-2], image.shape[-1])
        vsize = pose.shape[-1]
        # Get Data
        x_obs = image[:,:obs_size].reshape(-1,3,img_size[0],img_size[1]).to(device)
        v_obs = pose[:,:obs_size].reshape(-1,vsize).to(device)
        v_query = pose[:,obs_size].to(device)
        x_query_gt = image[:,obs_size].to(device)
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            # rmse
            mse_batch = (x_query_sample*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
            # mae
            mae_batch = torch.abs(x_query_sample*255 - x_query_gt*255)
            mae_batch = mae_batch.mean([1,2,3]).cpu().numpy()
            mae_record.append(mae_batch)
            # ce
            ce_batch = nn.BCELoss()(x_query_sample, x_query_gt)
            ce_batch = ce_batch.mean().cpu().numpy().reshape(1,1)
            ce_record.append(ce_batch)
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(max_batch), end="")
    
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    mae_record = np.concatenate(mae_record, 0)
    mae_mean = mae_record.mean()
    mae_std = mae_record.std()
    ce_record = np.concatenate(ce_record, 0)
    ce_mean = ce_record.mean()
    ce_std = ce_record.std()
    return {"rmse":[float(rmse_mean), float(rmse_std)],
            "mae" :[float(mae_mean), float(mae_std)],
            "ce"  :[float(ce_mean), float(ce_std)]}

def eval_shapenet(net, dataset, obs_size=3, max_batch=1000, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    rmse_record = []
    mae_record = []
    ce_record = []
    for it in range(max_batch):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = dataset_shapenet.get_batch(dataset, obs_size, 32)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            # rmse
            mse_batch = (x_query_sample*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
            # mae
            mae_batch = torch.abs(x_query_sample*255 - x_query_gt*255)
            mae_batch = mae_batch.mean([1,2,3]).cpu().numpy()
            mae_record.append(mae_batch)
            # ce
            ce_batch = nn.BCELoss()(x_query_sample, x_query_gt)
            ce_batch = ce_batch.mean().cpu().numpy().reshape(1,1)
            ce_record.append(ce_batch)
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(max_batch), end="")
    
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    mae_record = np.concatenate(mae_record, 0)
    mae_mean = mae_record.mean()
    mae_std = mae_record.std()
    ce_record = np.concatenate(ce_record, 0)
    ce_mean = ce_record.mean()
    ce_std = ce_record.std()
    return {"rmse":[float(rmse_mean), float(rmse_std)],
            "mae" :[float(mae_mean), float(mae_std)],
            "ce"  :[float(ce_mean), float(ce_std)]}

############ Training ############
def train(net, args, model_path):
    params = list(net.parameters())
    opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

    # ------------ Read Datasets ------------
    train_dataset = dataset.GqnDatasets(root_dir=args.data_path, train=True, fraction=args.frac_train, 
                                view_trans=args.view_trans, distort_type=args.distort_type)
    test_dataset = dataset.GqnDatasets(root_dir=args.data_path, train=False, fraction=args.frac_test, 
                                view_trans=args.view_trans, distort_type=args.distort_type)
    print("Data path: %s"%(args.data_path))
    print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
    print("Train data: ", len(train_dataset))
    print("Test data: ", len(test_dataset))
    print("Distort type:", args.distort_type)

    # ------------ Loss Function ------------
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "MAE":
        criterion = nn.L1Loss()
    elif args.loss_type == "CE":
        creterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # ------------ Prepare Variable------------
    img_path = model_path + "img/"
    save_path = model_path + "save/"
    train_record = {"loss_query":[], "lh_query":[], "kl_query":[]}
    eval_record = []
    best_eval = 999999
    steps = 0
    epochs = 0
    eval_step = 5000
    zfill_size = len(str(args.total_steps))

    # ------------ Start Training Steps ------------
    print("Start training ...")
    print("==============================")
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
            obs_size = np.random.randint(1,args.max_obs_size)
            obs_idx = np.random.choice(image.shape[1], obs_size)
            query_idx = np.random.randint(0, image.shape[1]-1)
            
            x_obs = image[:,obs_idx].reshape(-1,3,image.shape[-2],image.shape[-1]).to(device)
            v_obs = pose[:,obs_idx].reshape(-1,pose.shape[-1]).to(device)
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
            steps += 1
            
            # ------------ Print Result ------------
            if steps % 100 == 0:
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
                canvas = draw_query(net, train_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
                cv2.imwrite(fname, canvas)
                # Test
                print("Generate testing image ...")
                fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
                canvas = draw_query(net, test_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
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
                eval_results_train = eval(net, train_dataset, obs_size=3, max_batch=400, shuffle=False)
                print("Evaluate Testing Data ...")
                eval_results_test = eval(net, test_dataset, obs_size=3, max_batch=400, shuffle=False)
                eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})
                print("Dump evaluation record ...")
                with open(model_path+'eval_record.json', 'w') as file:
                    json.dump(eval_record, file)

                # ------------ Save Model ------------
                if steps%100000 == 0:
                    print("Save model ...")
                    torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
                
                # Apply RMSE as the metric for model selection.
                if eval_results_test["rmse"][0] < best_eval:
                    best_eval = eval_results_test["rmse"][0]
                    print("Save best model ...")
                    torch.save(net.state_dict(), save_path + "model.pth")
                print("Best Test RMSE:", best_eval)
                print("------------------------------")

        print("==============================")
        if steps >= args.total_steps:
            print("Save final model ...")
            torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
            break

def train_shapenet(net, args, model_path):
    params = list(net.parameters())
    opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

    # ------------ Read Datasets ------------
    train_dataset = dataset_shapenet.read_dataset(path=args.data_path, mode="train")
    test_dataset = dataset_shapenet.read_dataset(path=args.data_path, mode="test")
    print("Data path: %s"%(args.data_path))
    print("Data fraction: %f / %f"%(args.frac_train, args.frac_test))
    print("Train data: ", len(train_dataset))
    print("Test data: ", len(test_dataset))
    print("Distort type:", args.distort_type)
        
    # ------------ Loss Function ------------
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "MAE":
        criterion = nn.L1Loss()
    elif args.loss_type == "CE":
        creterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    
    # ------------ Prepare Variable ------------
    img_path = model_path + "img/"
    save_path = model_path + "save/"
    train_record = {"loss_query":[], "lh_query":[], "kl_query":[]}
    eval_record = []
    best_eval = 999999
    steps = 0
    epochs = 0
    eval_step = 5000
    zfill_size = len(str(args.total_steps))

    while(True):    
        # ------------ Get data (Random Observation) ------------
        obs_size = np.random.randint(1,args.max_obs_size)
        x_obs, v_obs, x_query_gt, v_query = dataset_shapenet.get_batch(train_dataset, obs_size, 32)

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
            loss_query = criterion(x_query, x_query_gt).mean()
            rec = [float(loss_query.detach().cpu().numpy()), float(loss_query.detach().cpu().numpy()), 0]
            
        # ------------ Train ------------
        loss_query.backward()
        opt.step()
        steps += 1

        # ------------ Print Result ------------
        if steps % 100 == 0:
            print("[Ep %s/%s] loss_q: %f| lh_q: %f| kl_q: %f"%(str(steps), str(args.total_steps), rec[0], rec[1], rec[2]))

        # ------------ Output Image ------------
        if steps % eval_step == 0:
            print("------------------------------")
            obs_size = 3
            gen_size = 5
            # Train
            print("Generate training image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_train.png"
            canvas = draw_query_shapenet(net, train_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
            cv2.imwrite(fname, canvas)
            # Test
            print("Generate testing image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
            canvas = draw_query_shapenet(net, test_dataset, obs_size=3, row_size=5, gen_size=1, shuffle=True)[0]
            cv2.imwrite(fname, canvas)

            # ------------ Training Record ------------
            train_record["loss_query"].append(rec[0])
            train_record["lh_query"].append(rec[1])
            train_record["kl_query"].append(rec[2])
            print("Dump training record ...")
            with open(model_path+'train_record.json', 'w') as file:
                json.dump(train_record, file)

            # ------------ Evaluation Record ------------
            print("Evaluate Training Data ...")
            eval_results_train = eval_shapenet(net, train_dataset, obs_size=3, max_batch=400, shuffle=False)
            print("Evaluate Testing Data ...")
            eval_results_test = eval_shapenet(net, test_dataset, obs_size=3, max_batch=400, shuffle=False)
            eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})
            print("Dump evaluation record ...")
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)

            # ------------ Save Model ------------
            if steps%100000 == 0:
                print("Save model ...")
                torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
                
            # Apply RMSE as the metric for model selection.
            if eval_results_test["rmse"][0] < best_eval:
                best_eval = eval_results_test["rmse"][0]
                print("Save best model ...")
                torch.save(net.state_dict(), save_path + "model.pth")
            print("Best Test RMSE:", best_eval)
            print("------------------------------")

        if steps >= args.total_steps:
            print("Save final model ...")
            torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
            break