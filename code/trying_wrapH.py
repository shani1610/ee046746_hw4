from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2lab, lab2rgb
import math
from tqdm import tqdm

def warpH(im1, H, out_size):
    lab_image = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB) #LAB
    x_range = np.arange(0, lab_image.shape[1])
    y_range = np.arange(0, lab_image.shape[0])
    warp_im1 = np.zeros((out_size[0], out_size[1],3),dtype="uint8")
    f = {}
    for i, channel in enumerate(["L","A","B"]):
        z = lab_image[:,:,i]
        f[channel] = interp2d(x_range, y_range, z, copy="False")
    H_inverse = np.linalg.inv(H)
    rgb_zero = np.array([0,0,0],dtype="uint8").reshape(1,1,3)
    lab_zero = cv2.cvtColor(rgb_zero, cv2.COLOR_BGR2LAB)
    for i in tqdm(range(warp_im1.shape[1])): #x
        for j in range(warp_im1.shape[0]): #y
            p2 = np.array([i,j,1]).reshape(-1,1) #indexs of wrap_im1
            p1 = H_inverse@p2
            p1 = p1/p1[2,0] #normalized the third index
            if p1[0]<0 or p1[0]>=im1.shape[1] or p1[1]<0 or p1[1]>=im1.shape[0]:
                warp_im1[j,i,:] = lab_zero
                continue
            for t, channel in enumerate(["L","A","B"]):
                warp_im1[j, i, t] = int(round(f[channel](p1[0,0], p1[1,0])[0]))
    warp_im1 = cv2.cvtColor(warp_im1.astype("uint8"), cv2.COLOR_LAB2BGR)
    return warp_im1

# test
# full_path_img1 = "./data/incline_R.png"
# im1 = cv2.imread(full_path_img1)
# image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
# H = [[1,0,0],[0,1,0],[0,0,1]]
# H = np.array(H)
# out_size = [3,3]
# wrap_im1=warpH(im1, H, out_size)