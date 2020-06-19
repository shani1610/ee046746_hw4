from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.interpolate import interp2d
from skimage.color import rgb2lab, lab2rgb
import math
from tqdm import tqdm

def warpH(im1, H, out_size):
    lab_image = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB) #LAB
    x_range = np.arange(0, lab_image.shape[1])
    y_range = np.arange(0, lab_image.shape[0])
    warp_im1 = np.zeros((out_size[0], out_size[1],3),dtype="uint8")
    f = {}
    for i, channel in enumerate(["L","A","B"]):
        z = lab_image[:,:,i]
        f[channel] = interp2d(x_range, y_range, z, copy="False")
    H_inverse = np.linalg.inv(H)
    rgb_zero = np.array([0,0,0],dtype="uint8").reshape(1,1,3)
    lab_zero = cv2.cvtColor(rgb_zero, cv2.COLOR_RGB2LAB)
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
    warp_im1 = cv2.cvtColor(warp_im1.astype("uint8"), cv2.COLOR_LAB2RGB)
    return warp_im1


# def FindCorners(im1, H):
#
#     H_inverse = np.linalg.inv(H)
#
#     left_top_original = np.array([0,0,1]).reshape(-1,1)
#     left_bottom_original = np.array([0,im1.shape[0],1]).reshape(-1,1)
#     right_top_original = np.array([im1.shape[1], 0,1]).reshape(-1,1)
#     right_bottom_original = np.array([im1.shape[1], im1.shape[0] ,1]).reshape(-1,1)
#
#     left_top = H_inverse @ left_top_original
#     left_bottom = H_inverse @ left_bottom_original
#     right_top = H_inverse @ right_top_original
#     right_bottom = H_inverse @ right_bottom_original
#
#     #normalized
#     left_top /= left_top[2,:]
#     left_bottom /= left_bottom[2,:]
#     right_top /= right_top[2,:]
#     right_bottom /= right_bottom[2,:]
#
#     return left_top, left_bottom, right_top, right_bottom


def FindCorners(im1, H):
    """ This function finds the limit of the transformed image so we won't get cut image.
        We do it with the transformation of the corners (homograph transforms plane to plane).
        input:
            :param im1: the first image.
            :param H: The homography matrix.
        output:
            :param LT: Left Top corner of the transformed image.
            :param LB: Left Bottom corner of the transformed image.
            :param RT: Right Top corner of the transformed image.
            :param RB: Right Bottom corner of the transformed image.
    """
    # find the limit of the transformed image so we won't get cut image
    # we do it with the transformation of the corners (homograph transforms plane to plane)
    corners = pinv(H) @ np.array([[0, 0, im1.shape[1], im1.shape[1]],
                                  [0, im1.shape[0], 0, im1.shape[0]],
                                  [1, 1, 1, 1]])
    corners /= corners[2, :]
    # first seperate to right and left corners
    leftCorners = corners[:2, np.argsort(corners[0])[:2]]
    rightCorners = corners[:2, np.argsort(corners[0])[2:]]

    # now seperate also to top and bottom corners
    LT = leftCorners[:, leftCorners[1].argmin()]  # Left Top
    LB = leftCorners[:, leftCorners[1].argmax()]  # Left Bottom
    RT = rightCorners[:, rightCorners[1].argmin()]  # Right Top
    RB = rightCorners[:, rightCorners[1].argmax()]  # Right Bottom

    return LT, LB, RT, RB

# def Translation(im1, H):
#
#     left_top, left_bottom, right_top, right_bottom = FindCorners(im1, H)
#
#     axis_y_top = min(right_top[1], left_top[1])
#     axis_y_bottom = max(right_bottom[1], left_bottom[1])
#     axis_x_left = min(left_top[0], left_bottom[0])
#     axis_x_right = max(right_top[0], right_bottom[0])
#
#     out_size = (int(axis_y_bottom - axis_y_top), int(axis_x_right - axis_x_left))
#
#     trans_mat = np.array([[1,0, int(axis_x_left)], [0,1,int(axis_y_top)],[0, 0, 1]])
#
#     H_trans = H @ trans_mat
#
#     return H_trans, out_size


def Translation(im1, H):
    """
    to get the right transformed image which is not cut, we need to know it limits as the other camera sees it.
    this function tells us the image size as the other camera sees it so it won't be cut. also we get the size of
    the  transformed image.
    :param im1: input image which will be transformed
    :param H: homograph from the other image to im1
    :return:
        H_fixed - fixed homograph that transform im1 to non-negetive coordinates
        outSize - the size of the im1 as the other camera sees it
        tx - offset in x
        ty - offset in y
    """
    # we need the corners as the other camera sees it:
    LT, LB, RT, RB = FindCorners(im1, H)
    xLeft = int(min(LT[0], LB[0]))       # the most left index
    xRight = int(max(RT[0], RB[0]))      # the most right index
    yTop = int(min(RT[1], LT[1]))    # the most top index
    yBootom = int(max(RB[1], LB[1]))     # the most bottom index

    outSize = (yBootom - yTop, xRight - xLeft)
    tx = xLeft
    ty = yTop
    translationMatrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    # xLeft and yTop are mapped to 0
    H_fixed = H @ translationMatrix
    return H_fixed, outSize


# test
# full_path_img1 = "./data/incline_R.png"
# im1 = cv2.imread(full_path_img1)
# image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
# H = [[1,0,0],[0,1,0],[0,0,1]]
# H = np.array(H)
# out_size = [3,3]
# wrap_im1=warpH(im1, H, out_size)