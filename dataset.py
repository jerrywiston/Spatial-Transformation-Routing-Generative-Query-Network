import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import distort

def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class GqnDatasets(Dataset):
    """
    Shepart Metzler mental rotation task
    dataset. Based on the dataset provided
    in the GQN paper. Either 5-parts or
    7-parts.
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param fraction: fraction of dataset to use
    :param target_transform: transform on viewpoints
    """
    def __init__(self, root_dir, train=True, transform=None, fraction=1.0, view_trans=True, distort_type=None):
        super(GqnDatasets, self).__init__()
        assert fraction > 0.0 and fraction <= 1.0
        prefix = "train" if train else "test"
        self.root_dir = os.path.join(root_dir, prefix)
        self.records = sorted([p for p in os.listdir(self.root_dir) if "pt" in p])
        self.records = self.records[:int(len(self.records)*fraction)]
        self.transform = transform
        self.distort_type = distort_type
        self.view_trans = view_trans

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.records[idx])
        with gzip.open(scene_path, "r") as f:
            data = torch.load(f)
            images, viewpoints = list(zip(*data))
        #print(len(images), images[1].shape)
        images = np.stack(images)
        #print(images.shape)
        viewpoints = np.stack(viewpoints)

        # uint8 -> float32
        images = images.transpose(0, 1, 4, 2, 3)
        images = torch.FloatTensor(images)/255

        if self.transform:
            images = self.transform(images)
        
        if self.distort_type is not None:
            if distort_type == "barrel_low":
                grid = distort_barrel_low(images.shape[3], images.shape[4])
            elif distort_type == "barrel_high":
                grid = distort_barrel_high(images.shape[3], images.shape[4])
            elif distort_type == "stretch":
                grid = distort.stretch(images.shape[3], images.shape[4])
            shape_rec = images.shape
            images = images.reshape(shape_rec[0]*shape_rec[1], shape_rec[2], shape_rec[3], shape_rec[4])
            grid = torch.FloatTensor(grid).repeat(images.shape[0],1,1,1)
            images = F.grid_sample(images, grid=grid)
            images = images.reshape(shape_rec)

        viewpoints = torch.FloatTensor(viewpoints)
        if self.view_trans:
            viewpoints = transform_viewpoint(viewpoints)

        return images, viewpoints

def _test(path="GQN-Datasets/rooms_ring_camera"):
    from torch.utils.data import DataLoader
    train_dataset = GqnDatasets(root_dir=path, train=True, fraction=1.0)
    #for idx, (image, pose) in enumerate(train_dataset):
    #    print(idx, image.shape, pose.shape)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for it, batch in enumerate(train_loader):
        print(it, len(batch), batch[0].squeeze(0).shape, batch[1].squeeze(0).shape)
        #print(batch[1])
        exit()

if __name__ == "__main__":
    _test()