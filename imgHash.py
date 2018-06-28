#!/usr/bin/python

import imagehash as imghash
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from PIL import Image
import os
from math import *
import numpy as np
from scipy import signal


# 以图搜图原理
# 缩小图片到8*8
# 获取灰度图像
# 计算hash(phash ahash等)
# 计算hamming距离,越小越相似
# 现成模块 imagehash
def showplt(plts):
    colum = 3
    row = int(len(plts) / colum)
    if (row == 0):
        row = 1
    elif (len(plts) % (colum * row) >= 1):
        row = row + 1
    for i in range(len(plts)):
        if (plts[i] is not None):
            plt.subplot(row, colum, i + 1)
            plt.imshow(plts[i], 'gray')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('C://Users/dellpc/Desktop/test.jpg')
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(image_hsv)
    height, width = H.shape
    newimg = np.zeros((height, width))
    plts = [img, H, newimg]
    for x in range(width):
        for y in range(height):
            H_v = H[x][y]
            S_v = S[x][y]
            V_v = V[x][y]
            if ((H_v >= 35 and H_v <= 77) and (S_v >= 40) and (H_v >= 40)):
                newimg[x][y] = 1
            else:
                newimg[x][y] = 0
    # plt.hist(H.ravel(), bins=256, range=[0, 256])
    # plt.hist(S.ravel(), bins=256, range=[0, 256])
    # plt.hist(V.ravel(), bins=256, range=[0, 256])
    # plt.show()
    showplt(plts)
