from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2lab, lab2rgb
import math

from ee046746_hw4.code.trying_getPoint import getPoints, computeH
from ee046746_hw4.code.trying_wrapH import warpH, Translation


def imageStitching(img1, wrap_img2):
    #stitcher = cv2.Stitcher_create(False)
    #panoImg = stitcher.stitch((img1, wrap_img2))
    panoImg = np.maximum(wrap_img2, img1)
    panoImg = np.uint8(panoImg)
    return panoImg

# test -------------------------------------------------------
full_path_img1 = "./data/incline_L.png"
image1 = cv2.imread(full_path_img1)
im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
full_path_img2 = "./data/incline_R.png"
image2 = cv2.imread(full_path_img2)
im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
N=8


#out_size = [im1.shape[0], im1.shape[1]] #new_imH,new_imW
p1, p2 = getPoints(im1, im2, N)
H2to1 = computeH(p1, p2)
H_trans, out_size = Translation(im1,H2to1)
wrap_img1 = warpH(im1, H_trans, out_size)
plt.figure(2)
plt.imshow(wrap_img1)
plt.show()
# panoramaTest = imageStitching(image1, wrap_img2)
# plt.figure(3)
# plt.imshow(panoramaTest)
# plt.show()

