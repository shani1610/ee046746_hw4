import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt

# Add imports if needed:
import pickle
from scipy.interpolate import interp2d
import time
from numpy.linalg import pinv
from skimage import color


# end imports

# Add extra functions here:
def Q2_2(im1, im2):
    """ This function runs Question 2.2 section. It means that it runs computeH function
        output:
            :param im1: first image
            :param im2: second image
            :param H2to1: H homogeneous matrix which transform from im2 to im1
    """
    p1, p2 = getPoints(im1, im2, N=4)
    H2to1 = computeH(p1, p2)

    # create 10 random points to transform:
    numOfPoints = 10
    # limit x range so point will not pass image size
    x = np.random.randint(0, im2.shape[1] // 2, (1, numOfPoints))
    y = np.random.randint(0, im2.shape[0], (1, numOfPoints))
    p2_hom_random = np.concatenate((x, y, np.ones((1, numOfPoints))), axis=0)
    p1_hom_outRandom = H2to1 @ p2_hom_random
    p1_hom_outRandom /= p1_hom_outRandom[[-1], :]

    # show images with matches points:
    fig, ax = plt.subplots(2, 1)
    ax[1].imshow(im1)
    ax[0].imshow(im2)
    for idx, (p_out, p_in) in enumerate(zip(p1_hom_outRandom.T, p2_hom_random.T)):
        inCircle = plt.Circle(p_in[:2], radius=6, color='g', fill=False)
        ax[0].add_artist(inCircle)
        ax[0].text(p_in[0] + 8, p_in[1] + 8, str(idx))
        outCircle = plt.Circle(p_out[:2], radius=6, color='r', fill=False)
        ax[1].add_artist(outCircle)
        ax[1].text(p_out[0] + 8, p_out[1] + 8, str(idx))
    fig.suptitle('Transformation of random pixels')
    ax[1].set_title('Out points after transformation')
    ax[0].set_title('In points - random choosed')
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()
    return im1, im2, H2to1


def getTransformedCorners(im1, H):
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


def fixH(im1, H):
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
    LT, LB, RT, RB = getTransformedCorners(im1, H)
    xLeft = int(min(LT[0], LB[0]))  # the most left index
    xRight = int(max(RT[0], RB[0]))  # the most right index
    yTop = int(min(RT[1], LT[1]))  # the most top index
    yBootom = int(max(RB[1], LB[1]))  # the most bottom index

    outSize = (yBootom - yTop, xRight - xLeft)
    tx = xLeft
    ty = yTop
    translationMatrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    # xLeft and yTop are mapped to 0
    H_fixed = H @ translationMatrix
    return H_fixed, outSize, xLeft, xRight, yTop, yBootom


def Q2_3(im1, im2):
    """ This function runs Question 2.3 section.
        It means that it does image warping to im1 with warpH function.
    """

    p1, p2 = getPoints(im1, im2, N=4)
    H2to1 = computeH(p1, p2)
    print('H2to1')
    print(H2to1)

    # H that works good for us for this example.
    # We chose points and calculate this H with computeH function. we saved it to avoid runtime for debugging
    # H = np.array([[1.69424090e-03, 1.92939042e-05, 9.98620205e-01],
    #               [-2.63154587e-04, 2.48851693e-03, -5.23534795e-02],
    #               [-1.14541269e-06, 1.20477243e-07, 2.76877412e-03]])
    H, outSize, xLeft, xRight, yTop, yBootom = fixH(im1, H2to1)
    t = time.time()
    warp_im1 = warpH(im1, H, outSize)
    print("warping time: ", time.time() - t)
    plt.imshow(warp_im1)
    plt.title('incline_L - as incline_R sees it. linear interpolation')
    _ = plt.axis('off')
    plt.show()
    return H, warp_im1, im2, xLeft, xRight, yTop, yBootom


def Q2_4(im1, im2):
    H, warp_im1, im2, xLeft, xRight, yTop, yBootom = Q2_3(im1, im2)
    # data = [H, warp_im1, im2, xLeft, xRight, yTop, yBootom]
    # with open(r'C:\Users\ברק\Desktop\ראייה ממוחשבת\hw4\code\my_data\H__warp_im1__im2__xLeft__xRight__yTop__yBootom.txt',
    #           'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # with open(r'C:\Users\ברק\Desktop\ראייה ממוחשבת\hw4\code\my_data\H__warp_im1__im2__xLeft__xRight__yTop__yBootom.txt',
    #           'rb') as f:
    #     data = pickle.load(f)
    #
    # H, warp_im1, im2, xLeft, xRight, yTop, yBootom = data

    warp_im1_big = np.zeros((max(yBootom, im2.shape[0]) - min(yTop, 0), max(xRight, im2.shape[1]) - min(xLeft, 0), 3))
    im1_warp_maskIdx = np.where(warp_im1 > 0)
    warp_im1_big[im1_warp_maskIdx] = warp_im1[im1_warp_maskIdx]
    im2_big = np.zeros(warp_im1_big.shape)
    im2_maskIdx = np.where(im2 > 0)
    im2_big[im2_maskIdx[0] - yTop, im2_maskIdx[1] - xLeft, im2_maskIdx[2]] = im2[im2_maskIdx]
    panoImg = imageStitching(im2_big, warp_im1_big)
    plt.imshow(panoImg)


def prepareToMerge(xLeft, xRight, yTop, yBootom, warp_im1, im2):
    """

    :param xLeft:
    :param xRight:
    :param yTop:
    :param yBootom:
    :param warp_im1:
    :param im2:
    :return:
    """
    warp_im1_big = np.zeros((max(yBootom, im2.shape[0]) - min(yTop, 0), max(xRight, im2.shape[1]) - min(xLeft, 0), 3),
                            dtype='uint8')
    im1_warp_maskIdx = np.where(warp_im1 > 0)
    warp_im1_big[im1_warp_maskIdx[0] + max(yTop, 0), im1_warp_maskIdx[1] + max(xLeft, 0), im1_warp_maskIdx[2]] = \
    warp_im1[im1_warp_maskIdx]
    im2_big = np.zeros(warp_im1_big.shape, dtype='uint8')
    im2_maskIdx = np.where(im2 > 0)
    im2_big[im2_maskIdx[0] + max(-yTop, 0), im2_maskIdx[1] + max(-xLeft, 0), im2_maskIdx[2]] = im2[im2_maskIdx]
    return warp_im1_big, im2_big


def Q2_5(im1, im2):
    p1, p2 = getPoints_SIFT(im1, im2)
    H = computeH(p1, p2)
    H, outSize, xLeft, xRight, yTop, yBootom = fixH(im1, H)
    warp_im1 = warpH(im1, H, outSize)

    # stitching preparation:
    warp_im1_big, im2_big = prepareToMerge(xLeft, xRight, yTop, yBootom, warp_im1, im2)

    # stitching:
    panoImg = imageStitching(im2_big, warp_im1_big)
    plt.imshow(panoImg), plt.title('panorama image using SIFT')
    _ = plt.axis('off')
    plt.show()


def mergeImages(im_origin, im_to_add, manualPoints=False):
    """

    :param im_origin:
    :param im_to_add:
    :return:
    """
    if manualPoints:
        p1, p2 = getPoints(im_to_add, im_origin, N=4)
    else:
        p1, p2 = getPoints_SIFT(im_to_add, im_origin)
    H = computeH(p1, p2)

    # downSampleRate = 2
    # im_to_add = cv2.resize(im_to_add, (im_to_add.shape[0] // downSampleRate, im_to_add.shape[1] // downSampleRate))
    # im_origin = cv2.resize(im_origin, (im_origin.shape[0] // downSampleRate, im_origin.shape[1] // downSampleRate))

    H, outSize, xLeft, xRight, yTop, yBootom = fixH(im_to_add, H)

    warp_im_to_add = warpH(im_to_add, H, outSize)

    # stitching preparation:
    warp_im_to_add, im_origin_big = prepareToMerge(xLeft, xRight, yTop, yBootom, warp_im_to_add, im_origin)

    # stitching:
    panoImg = imageStitching(im_origin_big, warp_im_to_add)
    return panoImg


def Q2_7():
    # ---- beach panorama ----
    downSampleRate = 2
    panorama123 = cv2.cvtColor(cv2.imread('data/beach1.jpg'), cv2.COLOR_BGR2RGB)
    # downSample because the memory problems
    panorama123 = cv2.resize(panorama123, (panorama123.shape[0] // downSampleRate,
                                           panorama123.shape[1] // downSampleRate))
    for i in [2, 3]:
        t = time.time()
        currIm = cv2.cvtColor(cv2.imread('data/beach' + str(i) + '.jpg'), cv2.COLOR_BGR2RGB)
        # downSample because the memory problems
        currIm = cv2.resize(currIm, (currIm.shape[0] // downSampleRate, currIm.shape[1] // downSampleRate))
        panorama123 = mergeImages(panorama123, currIm)
        print("i: ", i, "time: ", time.time() - t)

    cv2.imwrite('./my_data/beach_panorama123_SIFT.png', panorama123)

    # now the last images:
    im1 = cv2.cvtColor(cv2.imread('data/beach4.jpg'), cv2.COLOR_BGR2RGB)
    im1 = cv2.resize(im1, (im1.shape[0] // downSampleRate, im1.shape[1] // downSampleRate))
    im2 = cv2.cvtColor(cv2.imread('data/beach5.jpg'), cv2.COLOR_BGR2RGB)
    im2 = cv2.resize(im2, (im2.shape[0] // downSampleRate, im2.shape[1] // downSampleRate))
    panorama45 = mergeImages(im1, im2)

    cv2.imwrite('./my_data/beach_panorama45_SIFT.png', panorama45)

    # now merge the two parts to one:
    panorama = mergeImages(panorama123, panorama45)
    plt.imshow(panorama)
    plt.show()
    return panorama


# Extra functions end

# HW functions:
def getPoints(im1, im2, N):
    """
        This function displays the images im1 and im2 in order to
        select N couples of corresponding points on the two images
        input:
            :param im1:  first image
            :param im2: second image
            :param N: number of couples of correspondig points to select
        output:
            p1 - matrix of size 2xN which consists of pixel indices (x, y).T of points selected in im1.
            p2 - matrix of size 2xN which consists of pixel indices (x, y).T of points selected in im2.
        """
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Choose ' + str(N) + ' couples of corresponding points, couple after couple')
    ax[0].imshow(im1)
    ax[0].set_title('image 1')
    ax[1].imshow(im2)
    ax[1].set_title('image 2')

    # choose N couples manually:
    p = fig.ginput(2 * N, timeout=-1)
    # get p1 and p2. because we choose couple couple, p1 elements lays in even indices of p, and p2 in odd indices.
    p1 = np.stack(p[::2]).T
    p2 = np.stack(p[1::2]).T
    return p1, p2


def computeH(p1, p2):
    # TODO: describe function PDF and here.
    """
    The function set of matching points between two
    images and calculates the transformation between them.
    input:
        :param p1: 2 X N matrix of coordinates (x,y) in image 1.
        :param p2: 2 X N matrix of coordinates (x,y) in image 2.
    output:
        :param H2to1: homogenous matrix that transforms points p2 to p1.
    """
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    N = min(p1.shape[1], p2.shape[1])
    if N < 4:
        # we need 4 couples at least
        return None
    # homogenoues coordinates:
    q = np.concatenate((p2, np.ones((1, N))), axis=0)
    # create the data matrix from p1, p1:
    A = np.zeros((2 * N, 9))
    A[::2, :3] = q.T
    A[1::2, 3:6] = q.T
    A[::2, 6:] = -q.T * p1[0, :].reshape(-1, 1)
    A[1::2, 6:] = -q.T * p1[1, :].reshape(-1, 1)

    # we take A^TA for the EVD (same eigenvectors and squared eigenvalues). thus the matrix is 9x9.
    eigVal, eigVec = np.linalg.eig(A.T @ A)
    idxOfSmallestVec = np.abs(eigVal).argmin()
    # take smallest eigen vector (Ah=0):
    h = eigVec[:, idxOfSmallestVec]
    H2to1 = h.reshape(3, 3)
    return H2to1


def warpH(im1, H, out_size):
    """
           This function gets an input image and a transformation
           matrix H and returns the projected image.
           input:
               :param im1: colored image
               :param H: 3X3 matrix encoding the homography between im1 and im2.
               :param out_size: is the size of the wanted output (new_imH,new_imW).
           output:
               :param warp_im1: transposed warp image im1 include empty background.
       """

    im1_LAB = color.rgb2lab(im1)  # convert to lab color space
    eps = 1e-17
    warp_im1_LAB = np.zeros(tuple(out_size) + (3,))
    x_out, y_out = np.meshgrid(np.arange(out_size[1], dtype='uint16'), np.arange(out_size[0], dtype='uint16'))
    x_out = x_out.reshape(-1)  # row stack more convenient
    y_out = y_out.reshape(-1)  # row stack more convenient
    # q are the homogeneous indices of the out image
    q = np.concatenate((x_out.reshape(1, -1), y_out.reshape(1, -1), np.ones((1, x_out.size), dtype='uint16')), axis=0)
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
    """
    This function gets two images after axis alignment and returns a union of the two.
    input:
        :param img1: fisrt image
        :param wrap_img2: second image after warping.
    output:
        :param panoImg: the output gathered panorama.
    """
    panoImg = np.zeros(img1.shape, dtype='uint8')
    im1_mask = img1 > 0
    im2_wrap_mask = wrap_img2 > 0
    panoImg[im1_mask] = img1[im1_mask]
    panoImg[im2_wrap_mask] = wrap_img2[im2_wrap_mask]
    return panoImg


def getPoints_SIFT(im1, im2):
    """
    Your code here
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # number of matches that will serve us to calculate the homograph
    matchesToCountOn = 10

    pAll = np.stack([kp1[match.queryIdx].pt + kp2[match.trainIdx].pt for match in
                     matches[:min(len(matches), matchesToCountOn)]]).T
    p1 = pAll[:2]
    p2 = pAll[2:]
    return p1, p2


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
        p1_est = currH @ np.concatenate((p2, np.ones(1, matches_Number)), axis=1)
        p1_est /= p1_est[2, :]
        inliesIndices = np.where(((p1_est - p1) ** 2).sum(1) < tol)[0]

        # calc score and if it's better update parameters
        currScore = inliesIndices.size / matches_Number
        if currScore > bestScore:
            bestScore = currScore
            # bestH = currH
            bestInliersIndices = inliesIndices

    bestH = computeH(p1[:, bestInliersIndices], p2[:, bestInliersIndices])
    return bestH


# def blender():

if __name__ == '__main__':
    print('my_homography')
    downSampleRate = 4
    im1 = cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)
    im1 = im1[::downSampleRate, ::downSampleRate, :]    # downSample because the memory problems
    im2 = im2[::downSampleRate, ::downSampleRate, :]    # downSample because the memory problems
    #Q2_2(im1, im2)
    #Q2_3(im1, im2)
    #Q2_4(im1, im2)
    Q2_5(im1, im2)
    # panoImg = mergeImages(im2, im1)
    # beach_panorama_SIFT, beach_panorama_manual, palace_panorama_SIFT, palace_panorama_manual = Q2_7()
    # beach_panorama_SIFT = Q2_7()
    # cv2.imwrite('./my_data/beach_panorama_SIFT.png', beach_panorama_SIFT)
    # plt.imshow(panoImg)
    # plt.show()