import numpy as np
import cv2


def func_statistic_value(img_in, r):
    """计算图像的方差值"""
    if img_in.dtype != np.float64:
        img_in = img_in.astype(np.float64)

    # 创建均值滤波器
    kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float64) / ((2 * r + 1) ** 2)

    # 计算均值
    mean_mask = cv2.filter2D(img_in, -1, kernel, borderType=cv2.BORDER_DEFAULT)
    mean_img_sqr = mean_mask ** 2

    # 计算平方的均值
    img_sqr = img_in ** 2
    mean_sqr_img = cv2.filter2D(img_sqr, -1, kernel, borderType=cv2.BORDER_DEFAULT)

    # 计算方差并开方
    var_mask = mean_sqr_img - mean_img_sqr
    variance_val = np.sqrt(var_mask)

    return variance_val