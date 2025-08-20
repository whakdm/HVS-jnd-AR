import numpy as np
import cv2
from func_statistic_value import func_statistic_value
from func_edge import func_edge


def func_disorder_jnd(img0, img, r):
    """计算基于无序性的JND"""
    # 创建高斯核
    kernel_size = 2 * r + 1
    ker = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    ker = ker * ker.T  # 转换为2D核

    # 计算方差图
    vari_map = func_statistic_value(img0, 3)

    # 计算图像均值
    img_mean = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_DEFAULT)

    # 取均值和原图的最小值
    img_min = np.minimum(img_mean, img)

    # 初始化jnd_dis
    jnd_dis = img.copy()
    jnd_dis[vari_map < 10] = img_min[vari_map < 10]

    # 计算超边缘并应用
    superedge = func_edge(img0, 0.7)
    jnd_dis = jnd_dis * superedge

    return jnd_dis