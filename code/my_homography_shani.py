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
    left_top_original = np.array([0, 0, 1]).reshape(-1, 1)
    left_bottom_original = np.array([0, im1.shape[0], 1]).reshape(-1, 1)
    right_top_original = np.array([im1.shape[1], 0, 1]).reshape(-1, 1)
    right_bottom_original = np.array([im1.shape[1], im1.shape[0], 1]).reshape(-1, 1)
    #
    left_top =pinv(H) @ left_top_original
    left_bottom = pinv(H) @ left_bottom_original
    right_top =pinv(H) @ right_top_original
    right_bottom = pinv(H) @ right_bottom_original
    #
    # # normalized
    left_top /= left_top[2, :]
    left_bottom /= left_bottom[2, :]
    right_top /= right_top[2, :]
    right_bottom /= right_bottom[2, :]

    print(left_top)
    print(left_bottom)
    print(right_top)
    print(right_bottom)
    axis_top_button = [left_top, left_bottom, right_top, right_bottom]
    #
    return axis_top_button


def Translation(im1, H):
    # # This function used is in HW function wrapH
    #

    axis_top_button = FindCorners(im1, H)
    left_top = axis_top_button[0]
    left_bottom = axis_top_button[1]
    right_top = axis_top_button[2]
    right_bottom = axis_top_button[3]

    axis_y_top = int(min(right_top[1], left_top[1]))
    axis_y_bottom = int(max(right_bottom[1], left_bottom[1]))
    axis_x_left = int(min(left_top[0], left_bottom[0]))
    axis_x_right = int(max(right_top[0], right_bottom[0]))

    print(axis_y_top)
    print(axis_y_bottom)
    print(axis_x_left)
    print(axis_x_right)

    axis_arr = [axis_y_top, axis_y_bottom, axis_x_left, axis_x_right]
    out_size = (abs(axis_y_bottom - axis_y_top), abs((axis_x_right - axis_x_left)))
    trans_mat = np.array([[1, 0, axis_x_left], [0, 1, axis_y_top], [0, 0, 1]])
    H_trans = H @ trans_mat  # h2to1
    return H_trans, out_size, axis_arr


def getScaled(im2, warp_im1, axis_arr, warp_is_left):
    #
    axis_y_top = axis_arr[0]
    axis_y_bottom = axis_arr[1]
    axis_x_left = axis_arr[2]
    axis_x_right = axis_arr[3]
    print('axis_y_top')
    print(axis_y_top)
    print('axis_y_bottom')
    print(axis_y_bottom)
    print('axis_x_left')
    print(axis_x_left)
    print('axis_x_right')
    print(axis_x_right)

    shape_y = max(axis_y_bottom, im2.shape[0]) - min(axis_y_top, 0)
    shape_x = max(axis_x_right, im2.shape[1]) - min(axis_x_left, 0)
    print('shape_y')
    print(shape_y)
    print('shape_x')
    print(shape_x)

    warp_im1_scaled = np.zeros((shape_y, shape_x, 3))
    im2_scaled = np.zeros(warp_im1_scaled.shape)
    im2_mask = np.where(im2 > 0)
    if warp_is_left:
        im2_scaled[im2_mask[0] - min(axis_y_top, 0), im2_mask[1] - axis_x_left, im2_mask[2]] = im2[im2_mask]
    else:
        im2_scaled[im2_mask[0] - min(axis_y_top, 0), im2_mask[1], im2_mask[2]] = im2[im2_mask]
    plt.figure(2)
    plt.imshow(im2_scaled)
    plt.show()
    im1_warp_mask = np.where(warp_im1 > 0)
    if warp_is_left:
        warp_im1_scaled[im1_warp_mask] = warp_im1[im1_warp_mask]
    else:
        warp_im1_scaled[im1_warp_mask[0], im1_warp_mask[1] + axis_x_left, im1_warp_mask[2]] = warp_im1[im1_warp_mask]
    plt.figure(3)
    plt.imshow(warp_im1_scaled)
    plt.show()
    return warp_im1_scaled, im2_scaled

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
    lab_image = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)  # LAB
    warp_im1 = np.zeros((out_size[0], out_size[1], 3), dtype="uint8")
    x_range = np.arange(0, lab_image.shape[1])
    y_range = np.arange(0, lab_image.shape[0])
    zero_val = cv2.cvtColor(np.array([0, 0, 0], dtype="uint8").reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
    f = {}

    for idx, ch in enumerate(["L", "A", "B"]):
        z_range = lab_image[:, :, idx]
        f[ch] = interp2d(x_range, y_range, z_range, copy="False", kind='linear')
    # H_inverse = np.linalg.inv(H)

    for x in tqdm(range(warp_im1.shape[1])):  # x
        for y in range(warp_im1.shape[0]):  # y
            p2 = np.array([x, y, 1]).reshape(-1, 1)  # indexs of wrap_im1
            p1 = H @ p2
            p1 = p1 / p1[2, 0]  # normalized the third index
            if p1[0] > 0 and p1[1] > 0 and p1[0] < im1.shape[1] and p1[1] < im1.shape[0]:
                for idx, ch in enumerate(["L", "A", "B"]):
                    warp_im1[y, x, idx] = int(round(f[ch](p1[0, 0], p1[1, 0])[0]))
                continue
            warp_im1[y, x, :] = zero_val

    warp_im1 = cv2.cvtColor(warp_im1.astype("uint8"), cv2.COLOR_LAB2RGB)
    return warp_im1


def imageStitching(img1, wrap_img2):
    panoImg = np.maximum(img1, wrap_img2)
    #
    panoImg = np.uint8(panoImg)
    return panoImg

    #
    # def ransacH(matches, locs1, locs2, nIter, tol):
    #     """
    #     Your code here
    #     """
    #     return bestH
    #


def getPoints_SIFT1(im1, im2):

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print(matches)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    #good_matches = sorted(good_matches, key=lambda x: x.distance)
    #good_matches = good_matches.sort(key=lambda match: match.distance)
    #target_image = np.zeros((im1.shape[1], im1.shape[0]))
    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good_matches, target_image, flags=2)
    #plt.imshow(img3), plt.show()

    p1 = []
    p2 = []
    p1 = np.float32([kp1[m[0].queryIdx].pt for m in good_matches[:min(10, len(good_matches))]]).reshape(-1, 2)
    p2 = np.float32([kp2[m[0].trainIdx].pt for m in good_matches[:min(10, len(good_matches))]]).reshape(-1, 2)

    p1=p1.T
    p2=p2.T

    return p1, p2


def getPoints_SIFT(im1, im2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    number_of_matches = min(len(matches), 10)

    p_cor = []
    p_cor = np.stack([kp1[match.queryIdx].pt + kp2[match.trainIdx].pt for match in
                     matches[:number_of_matches]]).T
    p1 = p_cor[:2]
    p2 = p_cor[2:]
    return p1, p2

def prepareToMerge(xLeft, xRight, yTop, yBootom, warp_im1, im2):

    warp_im1_big = np.zeros((max(yBootom, im2.shape[0]) - min(yTop, 0), max(xRight, im2.shape[1]) - min(xLeft, 0), 3),
                            dtype='uint8')
    im1_warp_maskIdx = np.where(warp_im1 > 0)
    warp_im1_big[im1_warp_maskIdx[0] + max(yTop, 0), im1_warp_maskIdx[1], im1_warp_maskIdx[2]] = warp_im1[im1_warp_maskIdx]
    im2_big = np.zeros(warp_im1_big.shape, dtype='uint8')
    im2_maskIdx = np.where(im2 > 0)
    im2_big[im2_maskIdx[0] + max(-yTop, 0), im2_maskIdx[1] + max(xLeft, 0), im2_maskIdx[2]] = im2[im2_maskIdx]
    return warp_im1_big, im2_big


def panoramaTwoImg(im1, im2, warp_is_left):
    p2, p1 = getPoints_SIFT(im1, im2)
    #H2to1 = computeH(p1, p2)
    nIter = 1000
    tol = 5
    bestH = ransacH(p1, p2, nIter, tol)
    H_trans, out_size, axis_arr = Translation(im1, bestH)
    print('axis arr')
    print(axis_arr)
    warp_im1 = warpH(im1, H_trans, out_size)
    plt.figure(4)
    plt.imshow(warp_im1)
    plt.show()
    warp_im1_scaled, im2_scaled = getScaled(im2, warp_im1, axis_arr, warp_is_left)
    panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
    plt.figure(5)
    plt.imshow(panoramaTest)
    plt.show()
    return panoramaTest, H2to1

def panoramaTwoImgTmp(im1, im2, warp_is_left, H2to1):
    p1, p2 = getPoints_SIFT(im1, im2)
    #H2to1 = computeH(p1, p2)
    nIter = 1000
    tol = 5
    bestH = ransacH(p1, p2, nIter, tol)
    # H3to2 = np.array([[-5.11736316e-03,  3.64884082e-04,  7.42714231e-01],
    #                   [-1.72835740e-03, -4.79612570e-03, 6.69555992e-01],
    #                   [-1.81873113e-06,  3.28148995e-08, -4.24677187e-03]])
    # H = H3to2@H2to1
    H_trans, out_size, axis_arr = Translation(im1, bestH)
    print('axis arr')
    print(axis_arr)
    warp_im1 = warpH(im1, H_trans, out_size)
    plt.figure(4)
    plt.imshow(warp_im1)
    plt.show()
    warp_im1_scaled, im2_scaled = getScaled(im2, warp_im1, axis_arr, warp_is_left)
    panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
    plt.figure(5)
    plt.imshow(panoramaTest)
    plt.show()
    return panoramaTest

def ransacH_Someone(p1, p2, nIter, tol=5):
    bestH = np.zeros(3)
    max_score = 0
    for i in range(nIter):
        model = np.random.choice(p1.shape[0], 4)  # choose 4 random matches

        H = computeH(p1[model].T, p2[model].T)  # compute H for model
        p1_match = H @ np.vstack((p2.T, np.ones(1, p2.shape[0]))) # transform to homogeneous
        p1_match /= p1_match[2, :]  # normalization
        p1_match = p1_match[:2, :].T  # convert back to (x,y) points
        dist = np.linalg.norm(p1-p1_match, axis=1)
        score = np.sum(dist <= tol)
        if max_score < score:
            max_score = score
            bestH = H
    return bestH


def ransacH(p1, p2, nIter, tol):
    """
    Your code here  # TODO: DOC!
    """
    matches_Number = p1.shape[1]
    minSizeToCalcH = 4  # need 4 couples to calc H linearly
    usedGroupIndices = []
    bestScore = 0
    bestInliersIndices = np.array([])
    for _ in range(nIter):
        groupIndices = np.random.choice(matches_Number, minSizeToCalcH)
        while groupIndices in usedGroupIndices:
            # if we already used those matches, take others
            groupIndices = np.random.choice(matches_Number, minSizeToCalcH)
        usedGroupIndices.append(groupIndices)
        p1_sample = p1[:, groupIndices]  # sample randomly
        p2_sample = p2[:, groupIndices]  # sample randomly
        currH = computeH(p1_sample, p2_sample)  # calc H

        # count inliers:
        p2_with_ones = np.concatenate((p2, np.ones((1, matches_Number))), axis=0)
        p1_est = currH @ p2_with_ones
        p1_est /= p1_est[2, :]
        p1_est = p1_est[:2, :]
        inliesIndices = np.where(((p1_est - p1) ** 2).sum(1) < tol)[0]

        # calc score and if it's better update parameters
        currScore = inliesIndices.size / matches_Number
        if currScore > bestScore:
            bestScore = currScore
            # bestH = currH
            bestInliersIndices = inliesIndices

    bestH = computeH(p1[:, bestInliersIndices], p2[:, bestInliersIndices])
    return bestH

def beachTest():
    # images beach
    beach1 = cv2.imread('data/beach1.jpg')
    beach2 = cv2.imread('data/beach2.jpg')
    beach3 = cv2.imread('data/beach3.jpg')
    beach4 = cv2.imread('data/beach4.jpg')
    beach5 = cv2.imread('data/beach5.jpg')
    im_beach1 = cv2.cvtColor(beach1, cv2.COLOR_BGR2RGB)
    im_beach2 = cv2.cvtColor(beach2, cv2.COLOR_BGR2RGB)
    im_beach3 = cv2.cvtColor(beach3, cv2.COLOR_BGR2RGB)
    im_beach4 = cv2.cvtColor(beach4, cv2.COLOR_BGR2RGB)
    im_beach5 = cv2.cvtColor(beach5, cv2.COLOR_BGR2RGB)

    # 1+2
    im_beach1 = cv2.resize(im_beach1, (im_beach1.shape[0] // downSampleRate,
                                       im_beach1.shape[1] // downSampleRate))
    im_beach2 = cv2.resize(im_beach2, (im_beach2.shape[0] // downSampleRate,
                                       im_beach2.shape[1] // downSampleRate))
    panorama12 = panoramaTwoImg(im_beach1, im_beach2)
    cv2.imwrite('./my_data/beach_panorama12_SIFT.jpg', panorama12)

    # 1+2+3
    panorama12 = cv2.imread('./my_data/beach_panorama12_SIFT.jpg')
    panorama12 = cv2.cvtColor(panorama12, cv2.COLOR_BGR2RGB)
    im_beach3 = cv2.resize(im_beach3, (im_beach3.shape[0] // downSampleRate,
                                       im_beach3.shape[1] // downSampleRate))
    panorama123 = panoramaTwoImg(panorama12, im_beach3)
    cv2.imwrite('./my_data/beach_panorama123_SIFT.jpg', panorama123)

    # 4+5
    im_beach4 = cv2.resize(im_beach4, (im_beach4.shape[0] // downSampleRate,
                                       im_beach4.shape[1] // downSampleRate))
    im_beach5 = cv2.resize(im_beach5, (im_beach5.shape[0] // downSampleRate,
                                       im_beach5.shape[1] // downSampleRate))
    panorama45 = panoramaTwoImg(im_beach4, im_beach5)
    cv2.imwrite('./my_data/beach_panorama45_SIFT.jpg', panorama45)

    # 1+2+3+4+5
    panorama123 = cv2.imread('./my_data/beach_panorama123_SIFT.jpg')
    panorama123 = cv2.cvtColor(panorama123, cv2.COLOR_BGR2RGB)
    panorama45 = cv2.imread('./my_data/beach_panorama45_SIFT.jpg')
    panorama45 = cv2.cvtColor(panorama45, cv2.COLOR_BGR2RGB)
    panorama_final_beach = panoramaTwoImg(panorama123, panorama45)
    cv2.imwrite('./my_data/beach_panorama_final_beach_SIFT.jpg', panorama_final_beach)
    return beachPanoramaTest

def sintraTest():
    downSampleRate = 4

    sintra1 = cv2.imread('data/sintra1.JPG')
    sintra2 = cv2.imread('data/sintra2.JPG')
    sintra3 = cv2.imread('data/sintra3.JPG')
    sintra4 = cv2.imread('data/sintra4.JPG')
    sintra5 = cv2.imread('data/sintra5.JPG')

    im_sintra1 = cv2.cvtColor(sintra1, cv2.COLOR_BGR2RGB)
    im_sintra2 = cv2.cvtColor(sintra2, cv2.COLOR_BGR2RGB)
    im_sintra3 = cv2.cvtColor(sintra3, cv2.COLOR_BGR2RGB)
    im_sintra4 = cv2.cvtColor(sintra4, cv2.COLOR_BGR2RGB)
    im_sintra5 = cv2.cvtColor(sintra5, cv2.COLOR_BGR2RGB)

    im_sintra1 = cv2.resize(im_sintra1, (im_sintra1.shape[0] // downSampleRate,
                                         im_sintra1.shape[1] // downSampleRate))
    im_sintra2 = cv2.resize(im_sintra2, (im_sintra2.shape[0] // downSampleRate,
                                        im_sintra2.shape[1] // downSampleRate))
    im_sintra3 = cv2.resize(im_sintra3, (im_sintra3.shape[0] // downSampleRate,
                                         im_sintra3.shape[1] // downSampleRate))
    im_sintra4 = cv2.resize(im_sintra4, (im_sintra4.shape[0] // downSampleRate,
                                         im_sintra4.shape[1] // downSampleRate))
    im_sintra5 = cv2.resize(im_sintra5, (im_sintra5.shape[0] // downSampleRate,
                                         im_sintra5.shape[1] // downSampleRate))

    # 2+3
    panorama23, H3to2 = panoramaTwoImg(im_sintra2, im_sintra3,  warp_is_left=False)
    cv2.imwrite('./my_data/sintra_panorama23_SIFT.jpg', panorama23)

    # 3+4
    panorama34, H3to4 = panoramaTwoImg(im_sintra4, im_sintra3,  warp_is_left=True)
    cv2.imwrite('./my_data/sintra_panorama34_SIFT.jpg', panorama34)

    # 1+2
    panorama12, H2to1 = panoramaTwoImg(im_sintra1, im_sintra2,  warp_is_left=False)
    cv2.imwrite('./my_data/sintra_panorama12_SIFT.jpg', panorama12)

    # 4+5
    panorama45, H4to5 = panoramaTwoImg(im_sintra5, im_sintra4,  warp_is_left=True)
    cv2.imwrite('./my_data/sintra_panorama45_SIFT.jpg', panorama45)

    # 2+3+4
    panorama23 = cv2.imread('./my_data/sintra_panorama23_SIFT.jpg')
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    panorama34 = cv2.imread('./my_data/sintra_panorama34_SIFT.jpg')
    panorama34 = cv2.cvtColor(panorama34, cv2.COLOR_BGR2RGB)
    panorama234, H23to34 = panoramaTwoImg(panorama34, panorama23, warp_is_left=True)
    cv2.imwrite('./my_data/sintra_panorama234_SIFT.jpg', panorama234)

    # 1+2+3+4
    panorama234 = cv2.imread('./my_data/sintra_panorama234_SIFT.jpg')
    panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
    panorama1234 = panoramaTwoImgTmp(im_sintra1, panorama234, False, H2to1)
    cv2.imwrite('./my_data/sintra_panorama1234_SIFT.jpg', panorama1234)

    #1+2+3+4+5
    panorama1234 = cv2.imread('./my_data/sintra_panorama1234_SIFT.jpg')
    panorama1234 = cv2.cvtColor(panorama1234, cv2.COLOR_BGR2RGB)
    panorama_final_sintra, Hfinal = panoramaTwoImgTmp(im_sintra5, panorama1234, True, H4to5)
    cv2.imwrite('./my_data/sintra_panorama_final_SIFT.jpg', panorama_final_sintra)
    return panorama_final_sintra

if __name__ == '__main__':
        print('my_homography')
        # downSampleRate = 4
        # image1 = cv2.imread('data/incline_L.png')
        # image2 = cv2.imread('data/incline_R.png')
        # im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # #im1 = im1[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems
        # #im2 = im2[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems
        # panoramaTest = panoramaTwoImg(im2, im1, warp_is_left=False)

        print('my sintra')
        panorama_final_sintra = sintraTest()
        plt.figure(6)
        plt.imshow(panorama_final_sintra)
        plt.show()

        # downSampleRate = 4
        # sintra1 = cv2.imread('data/sintra1.JPG')
        # im_sintra1 = cv2.cvtColor(sintra1, cv2.COLOR_BGR2RGB)
        # panorama234 = cv2.imread('./my_data/sintra_panorama234_SIFT.jpg')
        # panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
        # panorama1234, Hsmt1 = panoramaTwoImg(im_sintra1, panorama234, warp_is_left=False)
        # cv2.imwrite('./my_data/sintra_panorama1234_SIFT.jpg', panorama1234)
        # plt.figure(6)
        # plt.imshow(panorama1234)
        # plt.show()