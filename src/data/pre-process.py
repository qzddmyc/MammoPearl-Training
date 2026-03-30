import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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

def preprocess_pipeline(image_path, save_path):
    """
    执行完整的图像预处理流水线
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    masked_img = create_breast_mask(img)
    denoised_img = denoise_image(masked_img)
    final_img = enhance_contrast_clahe(denoised_img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(str(save_path), final_img)
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = base_dir / 'data' / 'raw'
    csv_path = raw_data_dir / 'vindr_detection_folds.csv'
    raw_images_dir = raw_data_dir / 'images_png'
    processed_images_dir = base_dir / 'data' / 'processed2' / 'images_png'

    os.makedirs(processed_images_dir, exist_ok=True)

    if not csv_path.exists():
        print(f"找不到标注文件，请检查路径: {csv_path}")
        return

    print("正在加载CSV标注文件...")
    df = pd.read_csv(csv_path, dtype={'finding_birads': str})
    
    unique_images = df[['patient_id', 'image_id']].drop_duplicates()
    total_images = len(unique_images)
    print(f"共发现 {total_images} 张独立的乳腺X光片需要处理。\n")

    success_count = 0
    for _, row in tqdm(unique_images.iterrows(), total=total_images, desc="处理进度"):
        patient_id = str(row['patient_id'])
        image_id = str(row['image_id'])
        
        img_in_path = raw_images_dir / patient_id / image_id
        img_out_path = processed_images_dir / patient_id / image_id
        
        if not img_in_path.exists():
            stem_name = Path(image_id).stem
            possible_files = list((raw_images_dir / patient_id).glob(f"{stem_name}.*"))
            if possible_files:
                img_in_path = possible_files[0]
                img_out_path = processed_images_dir / patient_id / img_in_path.name
        
        if img_in_path.exists():
            if preprocess_pipeline(img_in_path, img_out_path):
                success_count += 1
                if success_count % 100 == 0:
                    tqdm.write(f"[*] 里程碑提示: 已经成功处理了 {success_count} 张图像。")
        else:
            tqdm.write(f"[!] 警告: 未找到对应的图像文件，已跳过 -> {img_in_path}") 

    print(f"\n处理完成！成功处理 {success_count}/{total_images} 张图片。")
    print(f"预处理后的图像已保存至: {processed_images_dir}")

if __name__ == '__main__':
    main()