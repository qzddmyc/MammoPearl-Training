# use:
# python src/data/bounding-box/bbox-test-resnet50.py --ckpt-path models/bbox_resnet50.pth
# or:
# python src/data/bounding-box/bbox-test-resnet50.py --ckpt-path models/bbox_resnet50.pth --save-predictions tmp/bbox_test_matches.csv

# * anchor sizes, internal box filtering, and matching thresholds will be auto-read from checkpoint meta when available.


"""Evaluate a bbox detector on the VinDr test split.

This script loads `models/bbox_resnet50.pth`, runs inference on the test split, and
compares predicted boxes with ground-truth boxes using IoU-based matching.
It reports:

- box precision / recall / F1 at the configured matching thresholds
- image-level presence accuracy
- mean IoU over matched boxes
- mean absolute coordinate error (pixels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
except Exception:  # pragma: no cover
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_v2
    FasterRCNN_ResNet50_FPN_V2_Weights = None
from torchvision.ops import box_iou

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] > 4:
            img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def detect_breast_region(
    img: np.ndarray,
    margin_ratio: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Detect the breast-region crop on a processed mammogram."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0, 0, 0, 0
    if np.max(gray) <= 0:
        return 0, 0, w, h

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(gray > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, w, h
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
    else:
        contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(contour)
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + cw)
        y2 = int(y + ch)

    margin_ratio = max(0.0, float(margin_ratio))
    margin_x = int(round((x2 - x1) * margin_ratio))
    margin_y = int(round((y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def crop_image_and_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    crop_box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and remap boxes into crop-local coordinates."""
    x1, y1, x2, y2 = crop_box
    cropped_img = img[y1:y2, x1:x2]
    if boxes.size == 0:
        return cropped_img, np.zeros((0, 4), dtype=np.float32)

    cropped_boxes = boxes.astype(np.float32).copy()
    cropped_boxes[:, [0, 2]] -= float(x1)
    cropped_boxes[:, [1, 3]] -= float(y1)

    crop_h, crop_w = cropped_img.shape[:2]
    cropped_boxes[:, 0] = np.clip(cropped_boxes[:, 0], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 2] = np.clip(cropped_boxes[:, 2], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 1] = np.clip(cropped_boxes[:, 1], 0, max(crop_h - 1, 0))
    cropped_boxes[:, 3] = np.clip(cropped_boxes[:, 3], 0, max(crop_h - 1, 0))

    keep = (cropped_boxes[:, 2] > cropped_boxes[:, 0] + 1) & (cropped_boxes[:, 3] > cropped_boxes[:, 1] + 1)
    return cropped_img, cropped_boxes[keep]


# 不再需要缩放
# def scale_boxes_to_image(boxes: np.ndarray, orig_size: Tuple[float, float], new_size: Tuple[int, int]) -> np.ndarray:
#     orig_h, orig_w = orig_size
#     new_h, new_w = new_size
#     if orig_h <= 0 or orig_w <= 0:
#         return boxes

#     scale_x = float(new_w) / float(orig_w)
#     scale_y = float(new_h) / float(orig_h)
#     scaled = boxes.copy().astype(np.float32)
#     scaled[:, [0, 2]] *= scale_x
#     scaled[:, [1, 3]] *= scale_y
#     scaled[:, 0::2] = np.clip(scaled[:, 0::2], 0, max(new_w - 1, 0))
#     scaled[:, 1::2] = np.clip(scaled[:, 1::2], 0, max(new_h - 1, 0))
#     keep = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
#     return scaled[keep]


class VinDrBboxDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        split_name: str,
        crop_breast_region: bool = False,
        breast_crop_margin: float = 0.05,
    ) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name
        self.crop_breast_region = crop_breast_region
        self.breast_crop_margin = breast_crop_margin

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()

        if df.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        self.samples: List[Dict[str, Any]] = []
        grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=True)

        for (patient_id, _series_id, image_id), group in grouped:
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            image_path = images_root / str(patient_id) / f"{image_id}"
            self.samples.append(
                {
                    "patient_id": str(patient_id),
                    "image_id": str(image_id),
                    "image_path": image_path,
                    "boxes": boxes,
                    "orig_size": (orig_h, orig_w),
                }
            )

        if not self.samples:
            raise ValueError(f"No samples could be built from {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if not sample["image_path"].exists():
            raise FileNotFoundError(f"Missing image: {sample['image_path']}")

        img = normalize_image(read_image_unicode(sample["image_path"]))
        h, w = img.shape[:2]

        boxes = sample["boxes"].astype(np.float32)

        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        crop_box = (0, 0, w, h)
        if self.crop_breast_region:
            crop_box = detect_breast_region(img, margin_ratio=self.breast_crop_margin)
            img, boxes = crop_image_and_boxes(img, boxes, crop_box)

        target = {
            "boxes": torch.from_numpy(boxes) if boxes.size > 0 else torch.zeros((0, 4), dtype=torch.float32),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        sample = dict(sample)
        sample["crop_box"] = crop_box
        sample["original_image_size"] = (h, w)
        return image_to_tensor(img), target, sample


def collate_fn(batch):
    images, targets, samples = zip(*batch)
    return list(images), list(targets), list(samples)


def build_model(num_classes: int = 2, anchor_sizes: List[Tuple[int, ...]] | None = None,
                box_score_thresh: float = 0.05, box_detections_per_img: int = 100) -> FasterRCNN:
    if anchor_sizes is None:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))

    aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))])
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # prefer to load torchvision v2 default weights when available
    weights = None
    if FasterRCNN_ResNet50_FPN_V2_Weights is not None:
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_anchor_generator=anchor_generator,
            box_score_thresh=box_score_thresh,
            box_detections_per_img=box_detections_per_img,
        )
    except TypeError:
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            # Avoid deprecated `pretrained`/`pretrained_backbone` args by
            # explicitly passing `weights=None` / `weights_backbone=None`.
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, rpn_anchor_generator=anchor_generator,
                                            box_score_thresh=box_score_thresh, box_detections_per_img=box_detections_per_img)  # type: ignore[call-arg]
        except Exception:
            model = fasterrcnn_resnet50_fpn_v2()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model: FasterRCNN, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    meta = ckpt.get("meta", {})
    return meta


def greedy_match(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float):
    """Greedy one-to-one matching of predictions to GT boxes."""
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return []

    order = torch.argsort(pred_scores, descending=True)
    used_gt = set()
    matches = []

    ious = box_iou(pred_boxes, gt_boxes)  # [P, G]
    for pred_idx in order.tolist():
        iou_row = ious[pred_idx]
        best_iou, best_gt = torch.max(iou_row, dim=0)
        gt_idx = int(best_gt.item())
        if float(best_iou.item()) >= iou_threshold and gt_idx not in used_gt:
            used_gt.add(gt_idx)
            matches.append((pred_idx, gt_idx, float(best_iou.item())))
    return matches


def evaluate(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    # save_debug_dir: Optional[Path] = None,
    # max_debug_images: int = 20,
    # nms_iou: float = 0.5,
):
    model.eval()

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    image_correct = 0
    image_total = 0

    matched_ious: List[float] = []
    coord_abs_errors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    # saved_debug = 0
    # printed_debug = 0
    # if save_debug_dir is not None:
    #     save_debug_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, targets, samples in tqdm(loader, desc="evaluating", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target, sample in zip(outputs, targets, samples):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                gt_boxes = target["boxes"].detach().cpu()

                matches = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold)
                tp = len(matches)
                fp = int(pred_boxes.shape[0] - tp)
                fn = int(gt_boxes.shape[0] - tp)

                total_gt += int(gt_boxes.shape[0])
                total_pred += int(pred_boxes.shape[0])
                total_tp += tp
                total_fp += fp
                total_fn += fn

                matched_ious.extend([m[2] for m in matches])

                for pred_idx, gt_idx, iou_val in matches:
                    pred = pred_boxes[pred_idx].numpy()
                    gt = gt_boxes[gt_idx].numpy()
                    coord_abs_errors.append(np.abs(pred - gt))
                    crop_x1, crop_y1, crop_x2, crop_y2 = sample.get("crop_box", (0, 0, 0, 0))
                    pred_orig = pred.copy()
                    gt_orig = gt.copy()
                    pred_orig[[0, 2]] += float(crop_x1)
                    pred_orig[[1, 3]] += float(crop_y1)
                    gt_orig[[0, 2]] += float(crop_x1)
                    gt_orig[[1, 3]] += float(crop_y1)
                    rows.append(
                        {
                            "patient_id": sample["patient_id"],
                            "image_id": sample["image_id"],
                            "crop_xmin": float(crop_x1),
                            "crop_ymin": float(crop_y1),
                            "crop_xmax": float(crop_x2),
                            "crop_ymax": float(crop_y2),
                            "pred_xmin": float(pred_orig[0]),
                            "pred_ymin": float(pred_orig[1]),
                            "pred_xmax": float(pred_orig[2]),
                            "pred_ymax": float(pred_orig[3]),
                            "gt_xmin": float(gt_orig[0]),
                            "gt_ymin": float(gt_orig[1]),
                            "gt_xmax": float(gt_orig[2]),
                            "gt_ymax": float(gt_orig[3]),
                            "iou": float(iou_val),
                            "score": float(pred_scores[pred_idx].item()),
                        }
                    )

                gt_present = gt_boxes.shape[0] > 0
                pred_present = pred_boxes.shape[0] > 0
                image_correct += int(gt_present == pred_present)
                image_total += 1

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    image_accuracy = image_correct / image_total if image_total else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_abs_error = (
        np.mean(np.stack(coord_abs_errors), axis=0).tolist() if coord_abs_errors else [0.0, 0.0, 0.0, 0.0]
    )

    metrics = {
        "images": image_total,
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "image_accuracy": image_accuracy,
        "mean_iou": mean_iou,
        "mean_abs_error": {
            "xmin": float(mean_abs_error[0]),
            "ymin": float(mean_abs_error[1]),
            "xmax": float(mean_abs_error[2]),
            "ymax": float(mean_abs_error[3]),
        },
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bbox detector on VinDr test split.")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=None, help="Score threshold used for matching; auto-read from checkpoint val_score_threshold if omitted")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold used for matching; auto-read from checkpoint val_iou_threshold if omitted")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", type=Path, default=None, help="Optional CSV path for matched predictions.")
    parser.add_argument("--anchor-sizes", type=str, default=None, help="Comma-separated anchor sizes; auto-read from checkpoint meta if omitted")
    parser.add_argument("--box-score-thresh", type=float, default=None, help="Score threshold for model-internal box filtering; auto-read from checkpoint meta if omitted")
    parser.add_argument("--box-detections-per-img", type=int, default=None, help="Max detections per image at inference time; auto-read from checkpoint meta if omitted")
    parser.add_argument("--force-breast-crop", action="store_true", help="Force breast-region cropping even if checkpoint meta does not request it")
    parser.add_argument("--disable-breast-crop", action="store_true", help="Disable breast-region cropping even if checkpoint meta requests it")
    parser.add_argument("--breast-crop-margin", type=float, default=None, help="Override breast crop padding ratio; defaults to checkpoint meta when available")
    # parser.add_argument(
    #     "--save-debug-dir",
    #     type=Path,
    #     default=None,
    #     help="Optional folder to save debug images with GT (green) and predictions (red).",
    # )
    # parser.add_argument("--max-debug-images", type=int, default=50, help="Maximum number of debug images to save.")
    # parser.add_argument(
    #     "--nms-iou",
    #     type=float,
    #     default=0.5,
    #     help="IoU threshold for per-image NMS applied before matching (set <=0 to disable).",
    # )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    ckpt_path = args.ckpt_path or (root / "models" / "bbox_resnet50.pth")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to inspect meta first (may contain anchor sizes / config)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})

    anchor_str = args.anchor_sizes if args.anchor_sizes is not None else meta.get("anchor_sizes", "8,16,32,64,128")
    anchor_sizes = tuple((int(s.strip()),) for s in str(anchor_str).split(",") if s.strip())
    print(f"[Info] Using anchor sizes: {anchor_str}")

    box_score_thresh = args.box_score_thresh if args.box_score_thresh is not None else float(meta.get("box_score_thresh", 0.05))
    box_detections_per_img = args.box_detections_per_img if args.box_detections_per_img is not None else int(meta.get("box_detections_per_img", 100))
    print(f"[Info] box_score_thresh={box_score_thresh}, box_detections_per_img={box_detections_per_img}")
    score_threshold = args.score_threshold if args.score_threshold is not None else float(meta.get("val_score_threshold", 0.5))
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else float(meta.get("val_iou_threshold", 0.5))
    print(f"[Info] eval score_threshold={score_threshold}, iou_threshold={iou_threshold}")

    crop_from_meta = bool(meta.get("crop_breast_region", False))
    if args.force_breast_crop:
        crop_breast_region = True
    elif args.disable_breast_crop:
        crop_breast_region = False
    else:
        crop_breast_region = crop_from_meta
    breast_crop_margin = float(args.breast_crop_margin) if args.breast_crop_margin is not None else float(meta.get("breast_crop_margin", 0.05))
    print(f"[Info] breast_crop_region={crop_breast_region}, breast_crop_margin={breast_crop_margin}")

    dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="test",
        crop_breast_region=crop_breast_region,
        breast_crop_margin=breast_crop_margin,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=2, anchor_sizes=anchor_sizes,
                        box_score_thresh=box_score_thresh, box_detections_per_img=box_detections_per_img)
    model.to(device)

    # load weights
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    metrics, rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        # save_debug_dir=args.save_debug_dir,
        # max_debug_images=args.max_debug_images,
        # nms_iou=args.nms_iou,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if meta:
        print("Checkpoint meta:")
        print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_predictions, index=False)
        print(f"Saved matched predictions to: {args.save_predictions}")


if __name__ == "__main__":
    main()


"""output:


"""
