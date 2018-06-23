import cv2
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from PIL import Image
import os
from math import *
import numpy as np


def showplt(plts):
    colum = 3
    row = int(len(plts) / colum)
    if (row == 0):
        row = 1
    elif (len(plts) % (colum * row) >= 1):
        row = row + 1
    for i in range(len(plts)):
        plt.subplot(row, colum, i + 1)
        plt.imshow(plts[i], 'gray')
    plt.show()


# cv分别提取三通道
def showRGB(img):
    # cv打开图片是以BGR方式,需要转化为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lay1 = img[:, :, 0]
    lay2 = img[:, :, 1]
    lay3 = img[:, :, 2]
    plts = [img, lay1, lay2, lay3]
    showplt(plts)


# 横向投影
def RowProj(src):
    gate = 0  # 阈值
    binary = np.copy(src)
    (x, y) = binary.shape
    a = [0 for z in range(0, x)]
    for i in range(0, x):
        for j in range(0, y):
            if binary[i, j] == 0:
                a[i] = a[i] + 1
                binary[i, j] = 255  # to be white
    for i in range(0, x):
        for j in range(0, a[i]):
            binary[i, j] = 0
    showplt([src, binary])


# 垂直投影
def ColumProj(src):
    gate = 0  # 阈值
    binary = np.copy(src)
    (x, y) = binary.shape
    a = [0 for z in range(0, y)]
    for i in range(0, y):
        for j in range(0, x):
            if binary[j, i] == 0:
                a[i] = a[i] + 1
                binary[j, i] = 255  # to be white
    for i in range(0, y):
        for j in range(0, a[i]):
            binary[j, i] = 0
    showplt([src, binary])


# 图像二值化
def showBinary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    ret, binary2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, binary3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    ret, binary4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    ret, binary5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # Otsu 滤波
    ret2, binary6 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # (x, y) = thresh1.shape
    # print(x, y)
    # a = [0 for z in range(0, x)]
    # for i in range(0, x):
    #     for j in range(0, y):
    #         if thresh1[i, j] == 0:
    #             a[i] = a[i] + 1
    #             thresh1[i, j] = 255  # to be white
    # for i in range(0, x):
    #     for j in range(0, a[i]):
    #         thresh1[i, j] = 0

    # thresh1 = binary1
    # region = []
    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     # 计算该轮廓的面积
    #     area = cv2.contourArea(cnt)
    #     # 面积小的都筛选掉
    #     if (area < 35):
    #         continue
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     region.append(box)
    # print(region[0])
    # print(region)
    # drawRect(region, binary1)
    # result = drawRect([region[0], np.asarray([[0, 0], [100, 0], [100, 100], [0, 100]])], img)
    # plts = [gray, binary1, result]
    # showplt(plts)


def enhance(img):
    image = Image.open('images/111.jpg')
    image.show()
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    # image_brightened.show()

    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    # image_colored.show()

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.show()

    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.show()


def showsome(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gauss_noiseImage = cv2.addGaussianNoise(img, 0.1)  # 添加10%的高斯噪声
    equalizeHist_Image = cv2.equalizeHist(gray)  # 对图片进行直方图均衡化
    Denoising_img = cv2.blur(gray, (5, 5))  # 对图片进行均值滤波
    plts = [gray, equalizeHist_Image, Denoising_img]
    showplt(plts)
    # plt画直方图
    # plt.hist(gray.ravel(),255)
    # plt.hist(equalizeHist_Image.ravel(),255)
    # plt.show()


def facePick():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread('faces.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 车牌区域检测
def carTag():
    img = cv2.imread('images/car4.jpg')
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image_hue, image_saturation, image_value = cv2.split(image_hsv)
    # 高斯平滑
    gaussian = cv2.GaussianBlur(image_saturation, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值平滑
    median = cv2.medianBlur(gaussian, 5)
    # soble边缘检测
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 1, ksize=5)
    ret, binary = cv2.threshold(sobel, 200, 255, cv2.THRESH_BINARY)
    median2 = cv2.medianBlur(binary, 5)
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(median2, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    region = findPlateNumberRegion(dilation2)
    result = drawRect(region, img)
    plts = [image_saturation, gaussian, median, sobel, binary, median2, dilation, erosion, dilation2, result]
    showplt(plts)


def findPlateNumberRegion(img):
    region = []
    # 查找轮廓
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 2000):
            continue
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print(rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 车牌正常情况下长高比在2.7-5之间
        ratio = float(width) / float(height)
        if (ratio > 5 or ratio < 2):
            continue
        region.append(box)
    return region


def drawRect(region, img):
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    return img


def drawLine(lines, img):
    if (lines is None):
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


# 定义旋转rotate函数
def rotateImg(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated


# 旋转并裁剪图片
def rotateCut(img, degree, box):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew))
    pt0 = box[1]
    pt3 = box[3]
    [[pt0[0]], [pt0[1]]] = np.dot(matRotation, np.array([[pt0[0]], [pt0[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt0[1]):int(pt3[1]), int(pt0[0]):int(pt3[0])]
    return imgOut


# 纸张区域识别
def OCR_area():
    img = cv2.imread('images/444.jpg')
    height, width = img.shape[:2]
    size = (int(width * 0.6), int(height * 0.6))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    canny = cv2.Canny(gaussian, 30, 90)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilation = cv2.dilate(canny, element1, iterations=1)
    # ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    region = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 找到最小的矩形，该矩形可能有方向
        boundrect = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        # 排除最外层框
        if (boundrect[2] == size[0] or rect[1][1] < 1000):
            continue
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(rect)
        region.append(box)

    result = drawRect(region, img)
    plts = [gray, gaussian, canny, dilation, result]
    if (len(rects) == 1):
        box = region[0]
        rect = rects[0]
        angle = rect[2]
        newimg = rotateCut(img, angle, box)
        plts.append(newimg)
    showplt(plts)


def OCR_demo():
    img = cv2.imread('images/222.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    # 膨胀一次，让轮廓突出
    # dilation = cv2.dilate(binary, element1, iterations=1)
    dilation = cv2.erode(gray, element1)
    ret, binary = cv2.threshold(dilation, 100, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    region = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 找到最小的矩形，该矩形可能有方向
        area = cv2.contourArea(cnt)
        if (area < 100):
            continue
        rect = cv2.minAreaRect(cnt)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高度
        height = abs(box[0][1] - box[2][1])
        if (height < 16 or height > 100):
            continue
        region.append(box)
    result = drawRect(region, img)
    plts = [gray, dilation, binary, result]
    showplt(plts)


def 直线检测():
    img = cv2.imread('images/222.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, binary = cv2.threshold(gauss, 80, 255, cv2.THRESH_BINARY_INV)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element1, iterations=1)

    canny = cv2.Canny(dilation, 50, 150)

    # 经验参数
    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny, 10, np.pi / 180, 160, None, minLineLength, maxLineGap)
    lineresult = drawLine(lines, img)
    plts = [gray, gauss, binary, dilation, canny, lineresult]
    showplt(plts)


if __name__ == '__main__':
    img = cv2.imread('images/333.jpg')
    # showRGB(img)
    # showBinary(img)
    # showsome(img)
    # enhance(img)
    # facePick()
    # carTag()
    # OCR_area()
    # OCR_demo()
    直线检测()
