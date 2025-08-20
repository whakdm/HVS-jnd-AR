import numpy as np
import cv2


def func_edge(matin, thre):
    """使用Canny算子估计边缘并进行处理"""
    # 确保输入为uint8类型用于Canny边缘检测
    if matin.dtype != np.uint8:
        matin = cv2.normalize(matin, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Canny边缘检测
    edge_region = cv2.Canny(matin, thre, thre * 2)  # 通常高阈值是低阈值的2-3倍

    # 膨胀操作
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # disk(2)对应5x5椭圆
    img_edge = cv2.dilate(edge_region, se)

    # 计算超边缘
    img_supedge = 1 - 0.8 * img_edge.astype(np.float64) / 255.0  # 归一化处理

    # 高斯滤波
    gaussian_kernal = cv2.getGaussianKernel(5, 0.8)
    gaussian_kernal = gaussian_kernal * gaussian_kernal.T  # 转换为2D核
    matout = cv2.filter2D(img_supedge, -1, gaussian_kernal, borderType=cv2.BORDER_DEFAULT)

    return matout