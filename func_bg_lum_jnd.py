import numpy as np
import cv2


def func_bg_lum_jnd(img0, min_lum):
    """计算基于亮度适应的JND"""
    alpha = 1.0
    jnd_ld = func_lum_jnd(img0, min_lum)
    jnd_lum_adapt = alpha * jnd_ld
    return jnd_lum_adapt


def func_lum_jnd(matin, min_lum):
    """计算亮度JND"""
    if matin.dtype != np.float64:
        matin = matin.astype(np.float64)

    bg_lum0 = func_bg_lum(matin)
    bg_lum = func_bg_adjust(bg_lum0, min_lum)

    col, row = matin.shape
    bg_jnd = lum_jnd()

    jnd_lum = np.zeros((col, row), dtype=np.float64)
    for x in range(col):
        for y in range(row):
            # 确保索引在有效范围内
            idx = int(bg_lum[x, y]) + 1
            idx = np.clip(idx, 1, 256)
            jnd_lum[x, y] = bg_jnd[idx - 1]  # Python是0索引

    return jnd_lum


def lum_jnd():
    """生成亮度JND查找表"""
    bg_jnd = np.zeros(256, dtype=np.float64)
    T0 = 17.0
    gamma = 3.0 / 128.0

    for k in range(256):
        lum = k  # k从0到255
        if lum <= 127:
            bg_jnd[k] = T0 * (1 - np.sqrt(lum / 127)) + 3
        else:
            bg_jnd[k] = gamma * (lum - 127) + 3

    return bg_jnd


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

    matout = cv2.filter2D(matin, -1, B, borderType=cv2.BORDER_DEFAULT)
    matout = np.floor(matout / 32)

    return matout


def func_bg_adjust(bg_lum0, min_lum):
    """调整背景亮度"""
    row, col = bg_lum0.shape
    bg_lum = bg_lum0.copy()

    for x in range(row):
        for y in range(col):
            if bg_lum[x, y] <= 127:
                bg_lum[x, y] = np.round(min_lum + bg_lum[x, y] * (127 - min_lum) / 127)

    return bg_lum