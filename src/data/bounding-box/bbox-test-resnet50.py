# use: 
# python src/data/bounding-box/bbox-test-resnet50.py --ckpt-path models/bbox_resnet50.pth --score-threshold 0.5

# * anchor-sizes will be auto-read from checkpoint meta if omitted.


"""Evaluate a bbox detector on the VinDr test split.

This script loads `models/bbox_resnet50.pth`, runs inference on the test split, and
compares predicted boxes with ground-truth boxes using IoU-based matching.
It reports:

- box precision / recall / F1 at IoU threshold 0.5
- image-level presence accuracy
- mean IoU over matched boxes
- mean absolute coordinate error (pixels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
    def __init__(self, csv_path: Path, images_root: Path, split_name: str) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name

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

        boxes = sample["boxes"].astype(np.float32)

        if boxes.size > 0:
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        target = {
            "boxes": torch.from_numpy(boxes) if boxes.size > 0 else torch.zeros((0, 4), dtype=torch.float32),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_to_tensor(img), target, sample


def collate_fn(batch):
    images, targets, samples = zip(*batch)
    return list(images), list(targets), list(samples)


def build_model(num_classes: int = 2, anchor_sizes: List[Tuple[int, ...]] | None = None) -> FasterRCNN:
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
        )
    except TypeError:
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            # Avoid deprecated `pretrained`/`pretrained_backbone` args by
            # explicitly passing `weights=None` / `weights_backbone=None`.
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, rpn_anchor_generator=anchor_generator)  # type: ignore[call-arg]
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
                matched_pred = {m[0] for m in matches}
                matched_gt = {m[1] for m in matches}

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
                    rows.append(
                        {
                            "patient_id": sample["patient_id"],
                            "image_id": sample["image_id"],
                            "pred_xmin": float(pred[0]),
                            "pred_ymin": float(pred[1]),
                            "pred_xmax": float(pred[2]),
                            "pred_ymax": float(pred[3]),
                            "gt_xmin": float(gt[0]),
                            "gt_ymin": float(gt[1]),
                            "gt_xmax": float(gt[2]),
                            "gt_ymax": float(gt[3]),
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
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", type=Path, default=None, help="Optional CSV path for matched predictions.")
    parser.add_argument("--anchor-sizes", type=str, default=None, help="Comma-separated anchor sizes; auto-read from checkpoint meta if omitted")
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

    dataset = VinDrBboxDataset(csv_path=csv_path, images_root=images_root, split_name="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Load checkpoint to inspect meta first (may contain anchor sizes / config)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})

    anchor_str = args.anchor_sizes if args.anchor_sizes is not None else meta.get("anchor_sizes", "8,16,32,64,128")
    anchor_sizes = tuple((int(s.strip()),) for s in str(anchor_str).split(",") if s.strip())
    print(f"[Info] Using anchor sizes: {anchor_str}")

    model = build_model(num_classes=2, anchor_sizes=anchor_sizes)
    model.to(device)

    # load weights
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    metrics, rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
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