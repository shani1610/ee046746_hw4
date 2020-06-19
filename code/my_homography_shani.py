from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

# Add imports if needed:
from scipy.interpolate import interp2d
from skimage import color
from tqdm import tqdm
from numpy.linalg import pinv

"""
Your code here
"""

# end imports

# Add extra functions here:

def FindCorners(im1, H):
    # # This function used is function Translation that is used in HW function wrapH
    # H_inverse = np.linalg.inv(H)
    #
    # left_top_original = np.array([0, 0, 1]).reshape(-1, 1)
    # left_bottom_original = np.array([0, im1.shape[0], 1]).reshape(-1, 1)
    # right_top_original = np.array([im1.shape[1], 0, 1]).reshape(-1, 1)
    # right_bottom_original = np.array([im1.shape[1], im1.shape[0], 1]).reshape(-1, 1)
    #
    # left_top = H_inverse @ left_top_original
    # left_bottom = H_inverse @ left_bottom_original
    # right_top = H_inverse @ right_top_original
    # right_bottom = H_inverse @ right_bottom_original
    #
    # # normalized
    # left_top /= left_top[2, :]
    # left_bottom /= left_bottom[2, :]
    # right_top /= right_top[2, :]
    # right_bottom /= right_bottom[2, :]
    #
    # return left_top, left_bottom, right_top, right_bottom
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

def Translation(im1, H):
        # # This function used is in HW function wrapH
        #
        # left_top, left_bottom, right_top, right_bottom = FindCorners(im1, H)
        #
        # axis_y_top = int(min(right_top[1], left_top[1]))
        # axis_y_bottom = int(max(right_bottom[1], left_bottom[1]))
        # axis_x_left = int(min(left_top[0], left_bottom[0]))
        # axis_x_right = int(max(right_top[0], right_bottom[0]))
        # axis_arr = [axis_y_top, axis_y_bottom, axis_x_left, axis_x_right ]
        # out_size = (abs(axis_y_bottom - axis_y_top), abs((axis_x_right - axis_x_left)))
        # trans_mat = np.array([[1, 0, axis_x_left], [0, 1, axis_y_top], [0, 0, 1]])
        # H_trans = H @ trans_mat #h2to1
        # return H_trans, out_size, axis_arr
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
    arr = [xLeft, xRight, yTop, yBootom]
    return H_fixed, outSize, arr


def getScaled(im2, warp_im1, axis_arr):
        #axis_arr = [axis_y_top, axis_y_bottom, axis_x_left, axis_x_right ]
        #
        # axis_y_top = axis_arr[0]
        # axis_y_bottom = axis_arr[1]
        # axis_x_left = axis_arr[2]
        # axis_x_right = axis_arr[3]
        #
        # shape_y = max(axis_y_bottom, im2.shape[0]) - min(axis_y_top, 0)
        #
        # shape_x = max(axis_x_right, im2.shape[1]) - min(axis_x_left, 0)
        #
        # print('shape y')
        # print(shape_y)
        #
        # print('shape x')
        # print(shape_x)
        #
        # warp_im1_scaled = np.zeros((shape_y, shape_x, 3))
        #
        # im2_scaled = np.zeros(warp_im1_scaled.shape)
        #
        # im2_mask = np.where(im2 > 0)
        # im2_scaled[im2_mask[0] - axis_y_top, im2_mask[1] - axis_x_left, im2_mask[2]] = im2[im2_mask]
        # print('im2_scaled')
        # print(im2_scaled)
        #
        # im1_warp_mask = np.where(warp_im1 > 0)
        # warp_im1_scaled[im1_warp_mask] = warp_im1[im1_warp_mask]
        # print('warp_im1_scaled')
        # print(warp_im1_scaled)
        #
        # return warp_im1_scaled, im2_scaled
    #axis_arr = [axis_y_top, axis_y_bottom, axis_x_left, axis_x_right]
        #arr = [xLeft, xRight, yTop, yBootom]
    xLeft = axis_arr[0]
    xRight = axis_arr[1]
    yTop = axis_arr[2]
    yBootom = axis_arr[3]
    warp_im1_big = np.zeros((max(yBootom, im2.shape[0]) - min(yTop, 0), (max(xRight, im2.shape[1]) - (min(xLeft, 0))), 3))
    im1_warp_maskIdx = np.where(warp_im1 > 0)
    warp_im1_big[im1_warp_maskIdx] = warp_im1[im1_warp_maskIdx]
    im2_big = np.zeros(warp_im1_big.shape)
    im2_maskIdx = np.where(im2 > 0)
    im2_big[im2_maskIdx[0] - yTop, im2_maskIdx[1] - xLeft, im2_maskIdx[2]] = im2[im2_maskIdx]
    return im2_big, warp_im1_big
    # Extra functions end

    # --------------------------------------------------------------------------------

    # HW functions:
def getPoints(im1, im2, N):
        fig = plt.figure(figsize=(9, 13))
        fig.add_subplot(1, 2, 1)
        plt.imshow(im1)
        fig.add_subplot(1, 2, 2)
        plt.imshow(im2)
        x = plt.ginput(N + 1, show_clicks=True, mouse_add=1)
        p1: List[Any] = []
        p2: List[Any] = []
        for i in range(N):
            if i % 2 == 0:  # even = image 1 = left
                p1.append(x[i])
            else:  # odd = image 2 = right
                p2.append(x[i])
        p1 = np.array(p1).T
        p2 = np.array(p2).T
        return p1, p2

def computeH(p1, p2):
        assert (p1.shape[1] == p2.shape[1])  # N
        assert (p1.shape[0] == 2)  # columns x and y
        N = p1.shape[1]
        A = np.zeros((2 * N, 9))
        for i in range(N):
            xi = p1[0][i]
            yi = p1[1][i]
            ui = p2[0][i]
            vi = p2[1][i]
            A[2 * i] = [xi, yi, 1, 0, 0, 0, -xi * ui, -yi * ui, -ui]
            A[2 * i + 1] = [0, 0, 0, xi, yi, 1, -xi * vi, -yi * vi, -vi]
            # A[2*i] = [-ui, -vi, -1, 0, 0, 0, ui*xi, vi*xi, xi]
            # A[2*i+1] = [0, 0, 0, -ui, -vi, -1, ui*yi, vi*yi, yi]
        (U, D, V_t) = np.linalg.svd(A, True)
        V = V_t.T
        H2to1 = V[:, -1]
        H2to1 = np.reshape(H2to1, [3, 3])
        return H2to1

def warpH(im1, H, out_size):
        # lab_image = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)  # LAB
        # x_range = np.arange(0, lab_image.shape[1])
        # y_range = np.arange(0, lab_image.shape[0])
        # warp_im1 = np.zeros((out_size[0], out_size[1], 3), dtype="uint8")
        # f = {}
        # for i, channel in enumerate(["L", "A", "B"]):
        #     z = lab_image[:, :, i]
        #     f[channel] = interp2d(x_range, y_range, z, copy="False", kind='linear')
        # #H_inverse = np.linalg.inv(H)
        # rgb_zero = np.array([0, 0, 0], dtype="uint8").reshape(1, 1, 3)
        # lab_zero = cv2.cvtColor(rgb_zero, cv2.COLOR_RGB2LAB)
        # for i in tqdm(range(warp_im1.shape[1])):  # x
        #     for j in range(warp_im1.shape[0]):  # y
        #         p2 = np.array([i, j, 1]).reshape(-1, 1)  # indexs of wrap_im1
        #         p1 = H @ p2
        #         p1 = p1 / p1[2, 0]  # normalized the third index
        #         if p1[0] < 0 or p1[0] >= im1.shape[1] or p1[1] < 0 or p1[1] >= im1.shape[0]:
        #             warp_im1[j, i, :] = lab_zero
        #             continue
        #         for t, channel in enumerate(["L", "A", "B"]):
        #             warp_im1[j, i, t] = int(round(f[channel](p1[0, 0], p1[1, 0])[0]))
        # warp_im1 = cv2.cvtColor(warp_im1.astype("uint8"), cv2.COLOR_LAB2RGB)
        # return warp_im1
        im1_LAB = color.rgb2lab(im1)  # convert to lab color space
        eps = 1e-17
        warp_im1_LAB = np.zeros(tuple(out_size) + (3,))
        x_out, y_out = np.meshgrid(np.arange(out_size[1], dtype='uint16'), np.arange(out_size[0], dtype='uint16'))
        x_out = x_out.reshape(-1)  # row stack more convenient
        y_out = y_out.reshape(-1)  # row stack more convenient
        # q are the homogeneous indices of the out image
        q = np.concatenate((x_out.reshape(1, -1), y_out.reshape(1, -1), np.ones((1, x_out.size), dtype='uint16')),
                           axis=0)
        # p are the homogeneous indices of the input image
        p = H @ q
        p /= p[2, :] + eps  # don't divide by 0

        x_in = p[0]
        y_in = p[1]

        insideOfLimit = np.where(~((x_in < 0) | (y_in < 0) | (x_in >= im1.shape[1]) | (y_in >= im1.shape[0])))[0]
        # remove the indices that fall out of the im1 shape
        x_out = x_out[insideOfLimit]
        y_out = y_out[insideOfLimit]
        x_in = x_in[insideOfLimit]
        y_in = y_in[insideOfLimit]

        # first, care about integer indices which are not need interpolation
        integerIdx = (x_in % 1 == 0) & (y_in % 1 == 0)
        warp_im1_LAB[y_out[integerIdx], x_out[integerIdx], :] = im1_LAB[x_in[integerIdx].astype('uint32'),
                                                                y_in[integerIdx].astype('uint32'), :]

        # now for the interpolation part:
        interpIndices = np.where(~integerIdx)[0]
        for channel in range(im1.shape[2]):
            f = interp2d(np.arange(im1.shape[1]), np.arange(im1.shape[0]), im1_LAB[:, :, channel], kind='linear',
                         fill_value=0)
            warp_im1_LAB[y_out[interpIndices], x_out[interpIndices], channel] = \
                np.array([float(f(XX, YY)) for XX, YY in zip(x_in[interpIndices], y_in[interpIndices])])

        warp_im1 = color.lab2rgb(warp_im1_LAB)  # range of values [0, 1]
        warp_im1 = (warp_im1 * 255).astype('uint8')  # range of values [0, 255]
        return warp_im1

def imageStitching(img1, wrap_img2):
        # panoImg = np.maximum(img1, wrap_img2)
        #
        # panoImg = np.uint8(panoImg)
        panoImg = np.zeros(img1.shape, dtype='uint8')
        im1_mask = img1 > 0
        im2_wrap_mask = wrap_img2 > 0
        panoImg[im1_mask] = img1[im1_mask]
        panoImg[im2_wrap_mask] = wrap_img2[im2_wrap_mask]


        return panoImg

    #
    # def ransacH(matches, locs1, locs2, nIter, tol):
    #     """
    #     Your code here
    #     """
    #     return bestH
    #
    # def getPoints_SIFT(im1, im2):
    #     """
    #     Your code here
    #     """
    #     return p1, p2

if __name__ == '__main__':
        print('my_homography')
        downSampleRate = 4
        image1 = cv2.imread('data/incline_L.png')
        image2 = cv2.imread('data/incline_R.png')
        im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        im1 = im1[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems
        im2 = im2[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems
        # part 2.1
        #N = 8
        #p1, p2 = getPoints(im1, im2, N)
        # part 2.2
        #H2to1 = computeH(p1, p2)
        H2to1 = np.array([[1.69424090e-03,  1.92939042e-05,  9.98620205e-01],
                   [-2.63154587e-04,  2.48851693e-03, -5.23534795e-02],
                   [-1.14541269e-06,  1.20477243e-07,  2.76877412e-03]])

        # part 2.3
        H_trans, out_size, axis_arr = Translation(im1, H2to1)  # not in HW
        warp_im1 = warpH(im1, H_trans, out_size)
        plt.figure(2)
        plt.imshow(warp_im1)
        plt.show()
        im2_scaled, warp_im1_scaled = getScaled(im2, warp_im1, axis_arr)
        plt.imshow(warp_im1_scaled)
        plt.imshow(im2_scaled)
        panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
        plt.figure(3)
        plt.imshow(panoramaTest)
        plt.show()
