import numpy as np
import cv2

# train, test, obj4, arith, multi
filename = "shapenet_data.npz"
data = np.load(filename)

# Read data
train_data = data['train']
data_id = 4
for i in range(train_data[data_id].shape[0]):
    print(int(i/18+1)*20, (i%18)*20)
    cv2.imshow("test", train_data[data_id,i])
    cv2.waitKey(0)