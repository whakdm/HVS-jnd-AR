import cv2
import os
# 假设AR预测相关函数与当前脚本在同一目录或已正确安装
from func_ar_predict_decomp import func_ar_predict_decomp  # 导入AR预测分解函数




def run_and_save_ar_prediction(input_image_path, output_dir="ar_predict_results",
                               min_thr=5, r=3, R=10):
    """
    运行AR预测并保存结果图片

    参数:
        input_image_path: 输入图像的路径
        output_dir: 结果保存目录
        min_thr: 最小阈值参数
        r: 局部窗口半径
        R: 搜索窗口半径
    """
    # 检查输入图像是否存在
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图像不存在: {input_image_path}")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像（以灰度模式读取，因为AR预测函数处理的是单通道图像）
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {input_image_path}")

    # 执行AR预测分解
    print(f"开始对图像 {os.path.basename(input_image_path)} 执行AR预测...")
    predicted_img = func_ar_predict_decomp(img, min_thr=min_thr, r=r, R=R)

    # 生成输出文件名
    filename = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_ar_predicted.png")

    # 保存预测结果
    cv2.imwrite(output_path, predicted_img)
    print(f"预测结果已保存至: {output_path}")

    return output_path


if __name__ == "__main__":
    # 示例用法
    # 替换为你的输入图像路径
    input_image = r"E:\rebulid\jnd_wujinjian\jnd_code\imgs\basketball1.png"  # 可以是相对路径或绝对路径

    # 可根据需要调整参数
    result_path = run_and_save_ar_prediction(
        input_image_path=input_image,
        output_dir="ar_predict_results",
        min_thr=5,  # 最小阈值
        r=3,  # 局部窗口半径，控制局部相似度计算范围
        R=10  # 搜索窗口半径，控制非局部相似区域的搜索范围
    )

    # 显示完成信息
    print(f"AR预测完成，结果保存于: {result_path}")
