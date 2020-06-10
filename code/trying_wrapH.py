from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2lab, lab2rgb
import math

full_path_img1 = "./data/incline_R.png"
im1 = cv2.imread(full_path_img1)
image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)


def mulH(im, H):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_idxs = np.array([i, j, 1])
            print(im_idxs)
            print(H)
            [new_i, new_j, new_k] = H.dot(im_idxs)
            print(new_i)
            if isinstance(new_i, int)==False or isinstance(new_j, int)==False:
                # inverse wraping interpolation
                x_range = np.arange(math.floor(new_i), math.ceil(new_i), 0.25)
                y_range = np.arange(math.floor(new_j), math.ceil(new_j), 0.25)
                xx, yy = np.meshgrid(x_range, y_range)
                z = im[:,:,:]
                f = interp2d(x_range, y_range, z, kind='cubic')
                trans_im[new_i,new_j] = f(x_range, y_range)
    return trans_im

H = [[1,0,0],[0,1,0],[0,0,1]]
H = np.array(H)
trans_im = mulH(im1, H)

def warpH(im1, H, out_size):
    # if im1 is colored split it to channels
    b_im = im1[:, :, 0]
    g_im = im1[:, :, 1]
    r_im = im1[:, :, 2]

    # lets work first on the b_im:
    warp_im1 = np.zeros((out_size[0], out_size[1]))
    normalized_im1 = im1 / 255.0
    lab_im1 = rgb2lab(normalized_im1)
    return warp_im1


b, g, r = cv2.split(im1)
plt.imshow(b)
