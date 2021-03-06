import numpy as np
import cv2

# train, test, obj4, arith, multi
def read_dataset(path="shapenet_data.npz", mode="train"):
    data = np.load(path)[mode]
    return data

def get_pose_code(id, theta_bias, phi_bias):
    theta = float(int(id/18+1)*20 + theta_bias)
    phi = float((id%18)*20 + phi_bias)
    #print(theta, phi)
    x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    y = np.cos(np.deg2rad(theta))
    z = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    pose_code = [x, y, z, np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]
    return np.array(pose_code).reshape(1,-1)

def get_batch(data, obs_size, batch_size=32, to_torch=True):
    batch_id = np.random.choice(data.shape[0], batch_size)
    img_obs = []
    pose_obs = []
    img_query = []
    pose_query = []

    for i, bid in enumerate(batch_id):
        obs_id = np.random.choice(data[0].shape[0], obs_size+1)
        pose_list = []
        phi_bias = np.random.uniform(-90,90)
        for j, oid in enumerate(obs_id+1):
            pose = get_pose_code(oid, 0, phi_bias)
            pose_list.append(pose)
        pose_np = np.concatenate(pose_list, 0)
        img_np = data[bid, obs_id]/255.0
        
        img_obs.append(img_np[:obs_size])
        pose_obs.append(pose_np[:obs_size])
        img_query.append(img_np[obs_size:])
        pose_query.append(pose_np[obs_size:])
    
    img_obs = np.concatenate(img_obs, 0)
    pose_obs = np.concatenate(pose_obs, 0)
    img_query = np.concatenate(img_query, 0)
    pose_query = np.concatenate(pose_query, 0)

    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

    #print(img_obs.shape)
    #print(pose_obs.shape)
    #print(img_query.shape)
    #print(pose_query.shape)

    return img_obs, pose_obs, img_query, pose_query

if __name__ == "__main__":
    dataset = read_dataset("shapenet_data.npz", "train")
    print("Get Batch")
    img_obs, pose_obs, img_query, pose_query = get_batch(dataset, 4, 3, False)
    import cv2
    for i in range(img_obs.shape[0]):
        cv2.imshow("test", img_obs[i])
        cv2.waitKey(0)
    for i in range(img_query.shape[0]):
        cv2.imshow("test", img_query[i])
        cv2.waitKey(0)
