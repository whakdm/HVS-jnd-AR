import numpy as np
import cv2


def func_contrast_mask_jnd(img):
    """计算基于对比度掩蔽的JND"""
    lum_diff = func_luminance_diff(img)

    thre = 80
    lum_diff_ = thre * np.log10(1 + lum_diff / thre) / np.log10(4)

    bg_lum = func_bg_lum(img)
    LANDA = 1 / 2
    alpha = 0.0001 * bg_lum + 0.115
    belta = LANDA - 0.01 * bg_lum

    jnd_CM = np.abs(lum_diff_ * alpha + belta)
    return jnd_CM


def func_luminance_diff(img):
    """计算亮度差异"""
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    # 定义四个梯度核
    G1 = np.array([
        [0, 0, 0, 0, 0],
        [1, 3, 8, 3, 1],
        [0, 0, 0, 0, 0],
        [-1, -3, -8, -3, -1],
        [0, 0, 0, 0, 0]
    ], dtype=np.float64)

    G2 = np.array([
        [0, 0, 1, 0, 0],
        [0, 8, 3, 0, 0],
        [1, 3, 0, -3, -1],
        [0, 0, -3, -8, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float64)

    G3 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 3, 8, 0],
        [-1, -3, 0, 3, 1],
        [0, -8, -3, 0, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float64)

    G4 = np.array([
        [0, 1, 0, -1, 0],
        [0, 3, 0, -3, 0],
        [0, 8, 0, -8, 0],
        [0, 3, 0, -3, 0],
        [0, 1, 0, -1, 0]
    ], dtype=np.float64)

    # 计算各个方向的梯度
    grad1 = cv2.filter2D(img, -1, G1, borderType=cv2.BORDER_DEFAULT) / 16
    grad2 = cv2.filter2D(img, -1, G2, borderType=cv2.BORDER_DEFAULT) / 16
    grad3 = cv2.filter2D(img, -1, G3, borderType=cv2.BORDER_DEFAULT) / 16
    grad4 = cv2.filter2D(img, -1, G4, borderType=cv2.BORDER_DEFAULT) / 16

    # 取绝对值最大的梯度作为亮度差异
    grad = np.stack([np.abs(grad1), np.abs(grad2), np.abs(grad3), np.abs(grad4)], axis=2)
    lum_diff = np.max(grad, axis=2)

    return lum_diff


def func_bg_lum(matin):
    """计算图像的平均背景亮度"""
    if matin.dtype != np.float64:
        matin = matin.astype(np.float64)

    B = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 0, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1]
    ], dtype=np.float64)

    # 滤波并归一化
    matout = cv2.filter2D(matin, -1, B, borderType=cv2.BORDER_DEFAULT)
    matout = np.floor(matout / 32)

    return matout