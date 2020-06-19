from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2lab, lab2rgb
import math

from ee046746_hw4.code.trying_getPoint import getPoints, computeH
from ee046746_hw4.code.trying_wrapH import warpH

def getPoints_SIFT(im1, im2):

    # Initiate SIFT detector
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print('matches:')
    print(matches)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
     #good is the matches

    result = []
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good, result, flags=2 )
    plt.imshow(img3), plt.show()

    return matches

# test -----------------------------------------
full_path_img1 = "./data/incline_L.png"
im1 = cv2.imread(full_path_img1)
image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

full_path_img2 = "./data/incline_R.png"
im2 = cv2.imread(full_path_img2)
image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

matches = getPoints_SIFT(image1, image2)

# N=8
#
# p1, p2 = getPoints(image1, image2, N)
# H2to1=computeH(p1, p2)
# print('H matrix')
# print(H2to1)

# h, status = cv2.findHomography(im1, im2)
# print('h matrix')
# print(h)
