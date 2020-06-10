from typing import List, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt

def getPoints(im1, im2, N):
    fig = plt.figure(figsize=(9, 13))
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2)
    x = plt.ginput(N + 1)
    p1: List[Any] = []
    p2: List[Any] = []
    for i in range(N):
        if i % 2 == 0:  # even = image 1 = left
            p1.append(x[i])
        else:  # odd = image 2 = right
            p2.append(x[i])
    p1=np.array(p1)
    p2=np.array(p2)
    return p1, p2


full_path_img1 = "./data/incline_R.png"
im1 = cv2.imread(full_path_img1)
image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

full_path_img2 = "./data/incline_L.png"
im2 = cv2.imread(full_path_img2)
image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

N=4

p1, p2 = getPoints(image1, image2, N)
print(p1,p2)

#

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    N=p1.shape[1]
    A=np.zeros((2*N,9))
    for i in range(N):
        xi=p1[0][i]
        yi=p1[1][i]
        ui=p2[0][i]
        vi=p2[1][i]
        A[2*i] = [xi, yi, 1, 0, 0, 0, -xi*ui, -yi*ui, -ui]
        A[2*i+1] = [0, 0, 0, xi, yi, 1, -xi*vi, -yi*vi, -vi]
    (U, D, V) = np.linalg.svd(A, False)
    H2to1 = V[:, -1]
    return H2to1

H2to1=computeH(p1, p2)
print(H2to1)