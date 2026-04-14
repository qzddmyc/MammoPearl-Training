"""Prompts:

1. 最基础的函数A，入参为(图片的比特信息, xmin, ymin, xmax, ymax)，
该函数返回“该图像的预测框之中的细胞区域中识别出来的细胞的mask图”。 

2. 一个函数B，功能比较多，但是都集成在内：从csv文件中读取所有标记为training训练集的图像，
将其传入函数A，并将产出的mask保存到/data/segmented/mask/<patient_id>/<img_id>中。

3. 一个函数C，给定“原始图片的路径”与“结果保存的目标文件夹”，将图片读取，然后加载models/bbox.pth，
通过该pth文件得到xmin, ymin, xmax, ymax四个值，之后调用函数A，得到mask文件后，
将mask保存至目标文件夹中。保存格式为“目标文件夹的目录/<原始文件名>”。

4. 一个函数D，与函数B类似，但读取的是所有的测试集数据，
需要加载models/bbox.pth并得到xmin, ymin, xmax, ymax值，再送到A中处理（不使用标注文件中的bbox位置值）。
图像保存位置与B函数相同。
"""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn


def extract_cell_mask_from_bbox(
    image_bits: np.ndarray | bytes | bytearray | memoryview,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray:
    """A: Use one bbox to extract a foreground mask from the image.

    The returned mask has the same height/width as the input image, with the
    bbox region segmented by a simple threshold + morphology pipeline.
    """

    if image_bits is None:
        raise ValueError("image_bits is None")

    # Decode / normalize image input.
    if isinstance(image_bits, np.ndarray):
        img = image_bits.copy()
    else:
        raw = np.frombuffer(image_bits, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Unable to decode image_bits")

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] >= 4:
            gray = cv2.cvtColor(img[:, :, :4], cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    gray = np.asarray(gray)
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if any(pd.isna(v) for v in (xmin, ymin, xmax, ymax)):
        return mask

    # Clamp bbox into the image range.
    x1 = int(max(0, min(w - 1, round(float(xmin)))))
    y1 = int(max(0, min(h - 1, round(float(ymin)))))
    x2 = int(max(0, min(w, round(float(xmax)))))
    y2 = int(max(0, min(h, round(float(ymax)))))

    if x2 <= x1 or y2 <= y1:
        return mask

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return mask

    roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    def _post_process(binary: np.ndarray) -> np.ndarray:
        binary = (binary > 0).astype(np.uint8) * 255
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big, iterations=1)

        num_labels, labels = cv2.connectedComponents(binary)
        if num_labels <= 1:
            return binary

        counts = np.bincount(labels.reshape(-1))
        counts[0] = 0
        if counts.max() <= 0:
            return np.zeros_like(binary)
        largest = int(counts.argmax())
        return (labels == largest).astype(np.uint8) * 255

    # Try two common thresholding strategies and keep the more plausible one.
    _, otsu_bin = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_bin = _post_process(otsu_bin)

    adaptive_bin = cv2.adaptiveThreshold(
        roi_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    adaptive_bin = _post_process(adaptive_bin)

    def _foreground_ratio(arr: np.ndarray) -> float:
        return float(np.count_nonzero(arr)) / float(arr.size) if arr.size else 0.0

    candidates = [otsu_bin, adaptive_bin]
    candidates = [c for c in candidates if np.count_nonzero(c) > 0]
    if candidates:
        # Prefer the smaller non-empty foreground, because the lesion/cell area
        # is usually much smaller than the whole bbox.
        chosen = min(candidates, key=_foreground_ratio)
    else:
        chosen = otsu_bin

    mask[y1:y2, x1:x2] = chosen
    return mask


def build_training_masks_from_csv(
    csv_path: str | Path | None = None,
    images_root: str | Path | None = None,
    segmented_root: str | Path | None = None,
) -> int:
    """B: Read the training split from CSV, generate masks with A, and save them.

    Saves base images to:  data/segmented/base/<patient_id>/<img_id>
    Saves masks to:        data/segmented/mask/<patient_id>/<img_id>
    """

    repo_root = Path(__file__).resolve().parents[3]
    csv_path = Path(csv_path) if csv_path is not None else repo_root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = Path(images_root) if images_root is not None else repo_root / "data" / "processed" / "images_png"
    segmented_root = Path(segmented_root) if segmented_root is not None else repo_root / "data" / "segmented"

    base_root = segmented_root / "base"
    mask_root = segmented_root / "mask"
    base_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    df = pd.read_csv(csv_path, low_memory=False)
    if "split" not in df.columns:
        raise ValueError("CSV does not contain a 'split' column")

    train_df = df[df["split"].astype(str).str.lower() == "training"].copy()
    if train_df.empty:
        raise ValueError("No training rows found in CSV")

    processed = 0
    for (patient_id, image_id), group in train_df.groupby(["patient_id", "image_id"], sort=True):
        patient_id = str(patient_id)
        image_id = str(image_id)

        image_path = images_root / patient_id / image_id
        if not image_path.exists():
            alt = images_root / patient_id / Path(image_id).name
            if alt.exists():
                image_path = alt
            else:
                # Try the common PNG fallback.
                alt_png = images_root / patient_id / f"{Path(image_id).stem}.png"
                if alt_png.exists():
                    image_path = alt_png
                else:
                    continue

        raw = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Preserve the original image as backup.
        patient_base_dir = base_root / patient_id
        patient_mask_dir = mask_root / patient_id
        patient_base_dir.mkdir(parents=True, exist_ok=True)
        patient_mask_dir.mkdir(parents=True, exist_ok=True)

        base_out_path = patient_base_dir / image_id
        mask_out_path = patient_mask_dir / image_id

        try:
            copy2(image_path, base_out_path)
        except Exception:
            # Fall back to re-encoding if direct copy fails.
            encoded = cv2.imencode(Path(image_id).suffix or ".png", img)[1]
            encoded.tofile(str(base_out_path))

        # Union masks from all bounding boxes in the image.
        union_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        valid_rows = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
        if not valid_rows.empty:
            for _, row in valid_rows.iterrows():
                one_mask = extract_cell_mask_from_bbox(
                    img,
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                )
                union_mask = cv2.bitwise_or(union_mask, one_mask)

        # If there is no valid bbox, save an empty mask so the file layout stays complete.
        if not cv2.imencode(".png", union_mask)[0]:
            continue
        cv2.imencode(".png", union_mask)[1].tofile(str(mask_out_path))
        processed += 1

    return processed


def predict_mask_for_image(
    image_path: str | Path,
    target_dir: str | Path,
    model_path: str | Path | None = None,
    score_threshold: float = 0.5,
) -> Path:
    """C: Predict bbox from models/bbox.pth, run A, and save the mask.

    The mask is saved to: target_dir/<original_filename>
    """

    repo_root = Path(__file__).resolve().parents[3]
    image_path = Path(image_path)
    target_dir = Path(target_dir)
    model_path = Path(model_path) if model_path is not None else repo_root / "models" / "bbox.pth"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    target_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(num_classes: int = 2) -> Any:
        try:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        except TypeError:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False)  # type: ignore[call-arg]
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model()
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    raw = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu()
    scores = output.get("scores", torch.zeros((0,), device=device)).detach().cpu()
    keep = scores >= float(score_threshold)
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.numel() == 0:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        top_idx = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
        box = boxes[top_idx].tolist()
        mask = extract_cell_mask_from_bbox(rgb, box[0], box[1], box[2], box[3])

    out_path = target_dir / image_path.name
    cv2.imencode(".png", mask)[1].tofile(str(out_path))
    return out_path


def build_test_masks_from_csv(
    csv_path: str | Path | None = None,
    images_root: str | Path | None = None,
    segmented_root: str | Path | None = None,
    model_path: str | Path | None = None,
    score_threshold: float = 0.5,
) -> int:
    """D: Read the test split, predict bbox by model, then create and save masks.

    Saves base images to:  data/segmented/base/<patient_id>/<img_id>
    Saves masks to:        data/segmented/mask/<patient_id>/<img_id>

    注意该函数只会取置信度最高的bbox框，其余标注都会被忽略！
    """

    repo_root = Path(__file__).resolve().parents[3]
    csv_path = Path(csv_path) if csv_path is not None else repo_root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = Path(images_root) if images_root is not None else repo_root / "data" / "processed" / "images_png"
    segmented_root = Path(segmented_root) if segmented_root is not None else repo_root / "data" / "segmented"
    model_path = Path(model_path) if model_path is not None else repo_root / "models" / "bbox.pth"

    base_root = segmented_root / "base"
    mask_root = segmented_root / "mask"
    base_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    def _build_model(num_classes: int = 2) -> Any:
        try:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        except TypeError:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False)  # type: ignore[call-arg]
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model()
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = pd.read_csv(csv_path, low_memory=False)
    if "split" not in df.columns:
        raise ValueError("CSV does not contain a 'split' column")

    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty:
        raise ValueError("No test rows found in CSV")

    processed = 0
    for (patient_id, image_id), group in test_df.groupby(["patient_id", "image_id"], sort=True):
        patient_id = str(patient_id)
        image_id = str(image_id)

        image_path = images_root / patient_id / image_id
        if not image_path.exists():
            alt = images_root / patient_id / Path(image_id).name
            if alt.exists():
                image_path = alt
            else:
                alt_png = images_root / patient_id / f"{Path(image_id).stem}.png"
                if alt_png.exists():
                    image_path = alt_png
                else:
                    continue

        raw = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        patient_base_dir = base_root / patient_id
        patient_mask_dir = mask_root / patient_id
        patient_base_dir.mkdir(parents=True, exist_ok=True)
        patient_mask_dir.mkdir(parents=True, exist_ok=True)

        base_out_path = patient_base_dir / image_id
        mask_out_path = patient_mask_dir / image_id

        try:
            copy2(image_path, base_out_path)
        except Exception:
            encoded = cv2.imencode(Path(image_id).suffix or ".png", img)[1]
            encoded.tofile(str(base_out_path))

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            if img.shape[2] == 4:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            continue

        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().to(device)
        with torch.no_grad():
            output = model([tensor])[0]

        boxes = output.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu()
        scores = output.get("scores", torch.zeros((0,), device=device)).detach().cpu()
        keep = scores >= float(score_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        if boxes.numel() == 0:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            top_idx = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
            box = boxes[top_idx].tolist()
            mask = extract_cell_mask_from_bbox(rgb, box[0], box[1], box[2], box[3])

        cv2.imencode(".png", mask)[1].tofile(str(mask_out_path))
        processed += 1

    return processed
