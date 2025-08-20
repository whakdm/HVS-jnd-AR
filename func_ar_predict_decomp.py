import numpy as np
import cv2
from func_statistic_value import func_statistic_value


def func_ar_predict_decomp(img0, min_thr=None, r=None, R=None):
    """AR预测分解"""
    # 设置默认参数
    if min_thr is None:
        min_thr = 5
    if r is None:
        r = 3
    if R is None:
        R = 10

    min_sigma = func_bg_lum_jnd(img0)
    min_sigma[min_sigma < min_thr] = min_thr

    img_predict = func_ar_nl(img0, min_sigma, r, R)
    return img_predict


def func_bg_lum_jnd(img0):
    """估计背景亮度失真"""
    if img0.dtype != np.float64:
        img0 = img0.astype(np.float64)

    min_lum = 32.0
    bg_lum0 = func_bg_lum(img0)
    bg_lum = func_bg_adjust(bg_lum0, min_lum)

    col, row = img0.shape
    bg_jnd = lum_jnd()

    jnd_lum_adapt = np.zeros((col, row), dtype=np.float64)
    for x in range(col):
        for y in range(row):
            idx = int(bg_lum[x, y]) + 1
            idx = np.clip(idx, 1, 256)
            jnd_lum_adapt[x, y] = bg_jnd[idx - 1]

    return jnd_lum_adapt


def func_bg_lum(matin):
    """计算图像的平均背景亮度"""
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


def lum_jnd():
    """生成亮度JND查找表"""
    bg_jnd = np.zeros(256, dtype=np.float64)
    T0 = 17.0
    gamma = 3.0 / 128.0

    for k in range(256):
        lum = k
        if lum <= 127:
            bg_jnd[k] = T0 * (1 - np.sqrt(lum / 127)) + 3
        else:
            bg_jnd[k] = gamma * (lum - 127) + 3

    return bg_jnd


def func_bg_adjust(bg_lum0, min_lum):
    """调整背景亮度"""
    row, col = bg_lum0.shape
    bg_lum = bg_lum0.copy()

    for x in range(row):
        for y in range(col):
            if bg_lum[x, y] <= 127:
                bg_lum[x, y] = np.round(min_lum + bg_lum[x, y] * (127 - min_lum) / 127)

    return bg_lum


def func_ar_nl(img_in, min_sigma, r, R):
    """非局部AR重建"""
    if img_in.dtype != np.float64:
        img_in = img_in.astype(np.float64)

    row, col = img_in.shape
    vari = func_statistic_value(img_in, r)

    # 计算sigma值
    sigma_value = min_sigma ** 2
    mask = vari > min_sigma
    sigma_value[mask] = (min_sigma[mask] * (min_sigma[mask] / vari[mask]) ** 0.5) ** 2

    # 填充图像用于边缘处理
    pad_size = R + r
    mat_pad = cv2.copyMakeBorder(img_in, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    # 确保img_pad与原始图像大小一致
    img_pad = mat_pad[pad_size:pad_size + row, pad_size:pad_size + col]

    # 创建均值核
    ker = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float64) / ((2 * r + 1) ** 2)

    img_reco = np.zeros((row, col), dtype=np.float64)
    weight_mat = np.zeros((row, col), dtype=np.float64)
    max_weight = np.zeros((row, col), dtype=np.float64)

    # 遍历所有偏移
    for u in range(-R, R + 1):
        for v in range(-R, R + 1):
            if u == 0 and v == 0:
                continue

            # 获取移动后的图像，确保大小与原始图像一致
            start_row = pad_size + u
            end_row = start_row + row
            start_col = pad_size + v
            end_col = start_col + col

            # 确保截取区域在有效范围内
            if start_row < 0 or end_row > mat_pad.shape[0] or start_col < 0 or end_col > mat_pad.shape[1]:
                continue

            img_move = mat_pad[start_row:end_row, start_col:end_col]

            # 计算差异和相似度
            mat_dif = (img_pad - img_move) ** 2
            sum_val = cv2.filter2D(mat_dif, -1, ker, borderType=cv2.BORDER_DEFAULT)
            mat_simi = np.exp(-sum_val / sigma_value)

            # 更新重建值和权重 - 移除错误的切片操作
            img_reco += img_move * mat_simi
            weight_mat += mat_simi

            # 更新最大权重
            max_weight[mat_simi > max_weight] = mat_simi[mat_simi > max_weight]

    # 完成重建
    img_recon = img_reco + max_weight * img_in
    weight_mat += max_weight

    # 防止除以零错误
    weight_mat[weight_mat == 0] = 1

    img_reconst = np.uint8(np.clip(img_recon / weight_mat, 0, 255))

    return img_reconst
