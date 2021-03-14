import numpy as np
import cv2
data = np.load("shapenet_data.npz")["obj4"]

print(data.shape)
sid = 0
vid = 0
while(True):
    print("Scene:", sid, "| View:", vid)
    img = cv2.resize(data[sid,vid], (256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("test1",img)
    k = cv2.waitKey(0)
    if k == ord('e'):
        sid += 1
        vid = 0
    elif k == ord('a'):
        if vid > 0:
            vid -= 1
        else:
            vid = data.shape[1]-1
    elif k == ord('d'):
        if vid < data.shape[1]-1:
            vid += 1
        else:
            vid = 0
    elif k == ord('q'):
        exit()
