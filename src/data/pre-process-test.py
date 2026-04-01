import os
import cv2
import numpy as np
import random
from pathlib import Path

def create_breast_mask(image):
    """
    通过传统图像处理方法（Otsu阈值+轮廓提取）去除背景和人工标记
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image 
        
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    return masked_img

def denoise_image(image):
    """
    双边滤波：在去除纹理噪声的同时，保留结节/肿块的边缘及钙化点
    """
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

def enhance_contrast_clahe(image):
    """
    CLAHE (限制对比度自适应直方图均衡化)：提升局部微小病灶的对比度
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_pipeline(image_path, tmp_dir, img_index):
    """
    执行并保存流水线的每一个中间步骤
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    orig_name = image_path.name

    # 步骤 0: 保存原始图像 (强烈建议保留原图作为对比参照)
    cv2.imwrite(str(tmp_dir / f"{img_index}-0-{orig_name}"), img)

    # 步骤 1: 掩膜提取与背景消除
    masked_img = create_breast_mask(img)
    cv2.imwrite(str(tmp_dir / f"{img_index}-1-{orig_name}"), masked_img)

    # 步骤 2: 双边滤波降噪
    denoised_img = denoise_image(masked_img)
    cv2.imwrite(str(tmp_dir / f"{img_index}-2-{orig_name}"), denoised_img)

    # 步骤 3: CLAHE 对比度增强 (最终输出)
    final_img = enhance_contrast_clahe(denoised_img)
    cv2.imwrite(str(tmp_dir / f"{img_index}-3-{orig_name}"), final_img)

    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_images_dir = base_dir / 'data' / 'raw' / 'images_png'
    
    # 设定临时输出文件夹
    tmp_dir = base_dir / 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    if not raw_images_dir.exists():
        print(f"找不到原始图片目录: {raw_images_dir}")
        return

    # 获取所有患者的文件夹列表
    all_patient_dirs = [d for d in raw_images_dir.iterdir() if d.is_dir()]
    
    if len(all_patient_dirs) == 0:
        print("未在 images_png 下找到任何文件夹。")
        return

    # 随机选取3个文件夹 (如果总数不足3个则全选)
    num_to_select = min(3, len(all_patient_dirs))
    selected_dirs = random.sample(all_patient_dirs, num_to_select)
    
    print(f"已随机选取以下 {num_to_select} 个文件夹进行处理:")
    for d in selected_dirs:
        print(f" - {d.name}")
    print("-" * 30)

    img_index = 1
    for patient_dir in selected_dirs:
        # 遍历该文件夹下的所有图片
        for img_path in patient_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                print(f"正在处理图片 {img_index}: {img_path.name}")
                success = preprocess_pipeline(img_path, tmp_dir, img_index)
                if success:
                    img_index += 1

    print("-" * 30)
    print(f"处理完成！所有阶段性结果均已保存至: {tmp_dir.resolve()}")
    print("你可以前往该目录查看流水线处理的效果。")

if __name__ == '__main__':
    main()