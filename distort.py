import numpy as np
#k1=-0.50, k2=-0.15, scale=1.15 #Pincushion
#k1=0.7, k2=0.7, scale=0.66 #Barral
#k1=2.5, k2=2.5, scale=0.35 #Barral 2
def distort_barrel(img_h, img_w, k1=2.5, k2=2.5, scale=0.35):
    img_cx = img_w/2.0
    img_cy = img_h/2.0
    grid = np.zeros((img_h, img_w, 2), dtype=np.float32)
    for j in range(img_h): #i->x / j->y
        for i in range(img_w):
            rsq = ((i-img_cx)/img_w)**2 + ((j-img_cy)/img_h)**2
            grid[j,i,0] = scale*(((i-img_cx)/img_w) * (1 + k1*rsq + k2*rsq**2)) + 0.5
            grid[j,i,1] = scale*(((j-img_cy)/img_h) * (1 + k1*rsq + k2*rsq**2)) + 0.5
    grid = (grid*2-1)[np.newaxis,...]
    return grid

def distort_sin(img_h, img_w, k1=10, k2=32, scale=0.94):
    img_cx = img_w/2.0
    img_cy = img_h/2.0
    grid = np.zeros((img_h, img_w, 2), dtype=np.float32)
    for j in range(img_h): #i->x / j->y
        for i in range(img_w):
            rsq = (i-img_cx)/img_w
            grid[j,i,0] = i/img_w
            grid[j,i,1] = scale * (((np.sin(k1*rsq) / k2) + j/img_h) - 0.5) + 0.5
    grid = (grid*2-1)[np.newaxis,...]
    return grid

def stretch(img_h, img_w, scale=0.5):
    img_cx = img_w/2.0
    img_cy = img_h/2.0
    grid = np.zeros((img_h, img_w, 2), dtype=np.float32)
    for j in range(img_h): #i->x / j->y
        for i in range(img_w):
            grid[j,i,0] = scale*((i-img_cx)/img_w) + 0.5
            grid[j,i,1] = j/img_h
    grid = (grid*2-1)[np.newaxis,...]
    return grid
    