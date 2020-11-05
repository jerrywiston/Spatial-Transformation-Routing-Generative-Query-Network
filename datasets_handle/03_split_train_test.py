import os
path = "/data/Datasets/EGQN-Datasets/rooms-random-objects/"
for i in range(2,1944):
    id = str(i).zfill(4)
    fn = id + "-of-1943.tfrecord"
    train_test = "train/"
    if i>1600:
        train_test="test/"
    os.system("mv "+path+fn+" "+path+train_test+fn)

