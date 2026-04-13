import os
import cv2
import numpy as np
import random
import importlib.util
from pathlib import Path


def load_module_from_path(path: Path, name: str = "imgseg"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / 'data' / 'processed' / 'images_png'

    tmp_dir = base_dir / 'tmp' / 'segmentation_test'
    os.makedirs(tmp_dir, exist_ok=True)

    if not processed_dir.exists():
        print(f"找不到预处理后图片目录: {processed_dir}")
        return

    # 查找所有患者文件夹
    all_patient_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    if len(all_patient_dirs) == 0:
        print("未在 processed/images_png 下找到任何患者文件夹。")
        return

    # 随机选取 3 个患者文件夹进行快速测试
    num_to_select = min(3, len(all_patient_dirs))
    selected_dirs = random.sample(all_patient_dirs, num_to_select)

    seg_path = base_dir / 'src' / 'data' / 'img-segmentation.py'
    if not seg_path.exists():
        print(f"找不到分割脚本: {seg_path}")
        return

    segmod = load_module_from_path(seg_path, name='imgseg')
    segment_image = getattr(segmod, 'segment_image')
    overlay_mask = getattr(segmod, 'overlay_mask')

    print(f"已选取 {num_to_select} 个患者进行分割测试:")
    for d in selected_dirs:
        print(f" - {d.name}")

    img_index = 1
    for patient_dir in selected_dirs:
        for img_path in patient_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                continue

            print(f"处理图片 {img_index}: {patient_dir.name}/{img_path.name}")
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                print("  无法读取，已跳过")
                continue

            # 如果为彩色或带 alpha，转换为灰度
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                mask = segment_image(gray, method='otsu')
            except Exception as e:
                print(f"  分割失败: {e}")
                continue

            overlay = overlay_mask(gray, mask)

            out_orig = tmp_dir / f"{img_index}-orig-{img_path.name}"
            out_mask = tmp_dir / f"{img_index}-mask-{img_path.name}"
            out_ovl = tmp_dir / f"{img_index}-overlay-{img_path.name}"

            # 使用 imencode + tofile 以兼容 Windows 路径和 unicode
            cv2.imencode('.png', gray)[1].tofile(str(out_orig))
            cv2.imencode('.png', mask)[1].tofile(str(out_mask))
            cv2.imencode('.png', overlay)[1].tofile(str(out_ovl))

            img_index += 1

    print("-" * 30)
    print(f"分割测试完成，结果保存在: {tmp_dir.resolve()}")


if __name__ == '__main__':
    main()
