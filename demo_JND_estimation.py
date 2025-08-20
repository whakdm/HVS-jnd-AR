import numpy as np
import cv2
import os
from func_ar_predict_decomp import func_ar_predict_decomp
from func_bg_lum_jnd import func_bg_lum_jnd
from func_contrast_mask_jnd import func_contrast_mask_jnd
from func_disorder_jnd import func_disorder_jnd
from func_randnum import func_randnum


def main():
    # 创建保存结果的目录
    output_dir = "jnd_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载图像（假设为灰度图）
    img0 = cv2.imread('imgs/cat1.png', cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        print("无法加载图像1.png")
        return

    # 保存原始图像
    cv2.imwrite(os.path.join(output_dir, "original_image.png"), img0)

    # JND计算参数
    min_sigma = 8
    min_lum = 32
    r = 3

    # 基于AR模型的自回归
    img_ar = func_ar_predict_decomp(img0, min_sigma)
    cv2.imwrite(os.path.join(output_dir, "ar_predicted_image.png"), img_ar)

    # 计算自由能
    img_free_energy = np.abs(img0.astype(np.float64) - img_ar.astype(np.float64))
    # 归一化自由能图像以便保存
    free_energy_img = cv2.normalize(img_free_energy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(output_dir, "free_energy.png"), free_energy_img)

    # 基于背景亮度的亮度适应JND
    jnd_LA = func_bg_lum_jnd(img_ar, min_lum)
    jnd_LA_img = cv2.normalize(jnd_LA, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(output_dir, "jnd_brightness_adaptation.png"), jnd_LA_img)

    # 基于边缘高度的对比度掩蔽JND
    jnd_CM = func_contrast_mask_jnd(img_ar)
    jnd_CM_img = cv2.normalize(jnd_CM, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(output_dir, "jnd_contrast_masking.png"), jnd_CM_img)

    # 基于无序性的不确定性JND
    jnd_Dis = func_disorder_jnd(img0, img_free_energy, r)
    jnd_Dis_img = cv2.normalize(jnd_Dis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(output_dir, "jnd_disorder.png"), jnd_Dis_img)

    # 计算JND掩码
    jnd_order = jnd_LA + jnd_CM - 0.3 * np.minimum(jnd_LA, jnd_CM)
    jnd_disorder = jnd_Dis
    jnd_map0 = jnd_order + jnd_disorder - 0.3 * np.minimum(jnd_order, jnd_disorder)

    row, col = img0.shape
    valid_mask = np.zeros((row, col), dtype=np.float64)
    valid_mask[r:row - r, r:col - r] = 1
    jnd_map = jnd_map0 * valid_mask

    # 归一化JND图像用于显示和保存
    jnd_img = (jnd_map / np.max(jnd_map)) * 255
    jnd_img = jnd_img.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "jnd_mask.png"), jnd_img)

    # 向图像注入JND噪声
    randmat = func_randnum(row, col)
    alpha = 1.0
    img_distort_jnd = img0.astype(np.float64) + alpha * randmat.astype(np.float64) * jnd_map
    img_distort_jnd = np.clip(img_distort_jnd, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "distorted_image.png"), img_distort_jnd)

    # 计算MSE
    mse_val = np.mean((img0.astype(np.float64) - img_distort_jnd.astype(np.float64)) ** 2)
    print(f"MSE = {mse_val:.2f}")
    # 将MSE值保存到文本文件
    with open(os.path.join(output_dir, "mse_value.txt"), "w") as f:
        f.write(f"MSE = {mse_val:.2f}")

    # 显示结果
    cv2.imshow('original image', img0)
    cv2.imshow('jnd mask', jnd_img)
    cv2.imshow('distorted image', img_distort_jnd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
