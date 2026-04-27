"""
Test script for CopyPasteWrapper visualization.

Saves a 2x2 grid image for each tested sample to tmp/copy_paste_test/:
  Top-left:     Original negative sample (raw)
  Top-right:    Random donor positive sample with GT boxes drawn
  Bottom-left:  Pasted result image (raw)
  Bottom-right: Pasted result image with synthesized GT boxes drawn

Usage (from repo root, Git Bash):
    python src/data/bounding-box/test_copy_paste.py
"""

import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2

# ---------------------------------------------------------------------------
# Add repo root to sys.path so we can import bbox-train helpers.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "data" / "bounding-box"))

# bbox-train.py uses a hyphen in its name, so we must import it via importlib.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bbox_train",
    REPO_ROOT / "src" / "data" / "bounding-box" / "bbox-train.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["bbox_train"] = _mod  # register before exec so @dataclass can resolve __module__
_spec.loader.exec_module(_mod)

VinDrBboxDataset = _mod.VinDrBboxDataset
TrainAugmentWrapper = _mod.TrainAugmentWrapper
CopyPasteWrapper = _mod.CopyPasteWrapper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv"
IMAGES_ROOT = REPO_ROOT / "data" / "processed" / "images_png"
OUT_DIR = REPO_ROOT / "tmp" / "copy_paste_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 8          # number of negative samples to visualize
PASTE_PROB = 1.0         # force paste for every negative sample in test
MAX_PASTES = 2
SEED = 42

BOX_COLOR_DONOR = (0, 220, 0)    # green for donor GT boxes
BOX_COLOR_PASTE = (220, 50, 50)  # red for pasted GT boxes
BOX_THICKNESS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert [C, H, W] float tensor in [0,1] to RGB uint8 HxWxC."""
    arr = t.permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def draw_boxes(img_rgb: np.ndarray, boxes: torch.Tensor, color, thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on an RGB image (in-place copy)."""
    out = img_rgb.copy()
    if boxes.numel() == 0:
        return out
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a text label in the top-left corner."""
    out = img.copy()
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_grid(tl, tr, bl, br) -> np.ndarray:
    """Stack four images (same size) into a 2x2 grid."""
    # Resize all to the same height/width (the maximum).
    imgs = [tl, tr, bl, br]
    max_h = max(im.shape[0] for im in imgs)
    max_w = max(im.shape[1] for im in imgs)

    def pad(im):
        h, w = im.shape[:2]
        padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        padded[:h, :w] = im
        return padded

    imgs = [pad(im) for im in imgs]
    top = np.concatenate([imgs[0], imgs[1]], axis=1)
    bot = np.concatenate([imgs[2], imgs[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)

print("[test_copy_paste] Loading dataset ...")
base_dataset = VinDrBboxDataset(
    csv_path=CSV_PATH,
    images_root=IMAGES_ROOT,
    split_name="training",
    crop_breast_region=True,
)

# Separate indices
all_indices = list(range(len(base_dataset)))
pos_indices = [i for i in all_indices if base_dataset.samples[i].boxes.size > 0]
neg_indices = [i for i in all_indices if base_dataset.samples[i].boxes.size == 0]

print(f"  Total: {len(all_indices)}, positive: {len(pos_indices)}, negative: {len(neg_indices)}")

# We wrap the full dataset with augmentation so donors also get augmented.
augmented_dataset = TrainAugmentWrapper(
    torch.utils.data.Subset(base_dataset, all_indices),
    hflip_prob=0.5,
    brightness_delta=0.2,
    rotation_max_deg=8.0,
)

# pos/neg indices are relative to augmented_dataset (same ordering as all_indices).
pos_in_aug = [i for i, idx in enumerate(all_indices) if base_dataset.samples[idx].boxes.size > 0]
neg_in_aug = [i for i, idx in enumerate(all_indices) if base_dataset.samples[idx].boxes.size == 0]

# ---------------------------------------------------------------------------
# Generate visualizations
# (We manually replicate the CopyPasteWrapper paste logic here so the
#  displayed donor image matches the actual crop source, giving an accurate
#  2x2 comparison grid.)
chosen_neg = random.sample(neg_in_aug, min(NUM_SAMPLES, len(neg_in_aug)))

for sample_num, neg_idx in enumerate(chosen_neg):
    # 1. Original negative sample
    orig_neg_img, orig_neg_target = augmented_dataset[neg_idx]
    neg_rgb = tensor_to_uint8(orig_neg_img)

    # 2. Manually replicate CopyPasteWrapper tissue detection so we know
    #    exactly which donor was chosen (for a correct 2x2 display).
    target_img = orig_neg_img.clone()
    _, H, W = target_img.shape
    mean_img = target_img.mean(dim=0)
    tissue_mask = mean_img > 0.05
    tissue_rows = tissue_mask.any(dim=1).nonzero(as_tuple=False)
    tissue_cols = tissue_mask.any(dim=0).nonzero(as_tuple=False)
    if tissue_rows.numel() == 0 or tissue_cols.numel() == 0:
        t_y1, t_x1, t_y2, t_x2 = 0, 0, H, W
    else:
        t_y1 = int(tissue_rows[0].item())
        t_y2 = int(tissue_rows[-1].item()) + 1
        t_x1 = int(tissue_cols[0].item())
        t_x2 = int(tissue_cols[-1].item()) + 1

    pasted_boxes_list: List = []
    actual_donors: List[int] = []

    n_paste = random.randint(1, MAX_PASTES)
    for _ in range(n_paste):
        donor_aug_idx = random.choice(pos_in_aug)
        donor_img, donor_target = augmented_dataset[donor_aug_idx]
        if donor_target["boxes"].shape[0] == 0:
            continue
        bi = random.randrange(donor_target["boxes"].shape[0])
        x1, y1, x2, y2 = donor_target["boxes"][bi].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        _, dH, dW = donor_img.shape
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, dW), min(y2, dH)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = donor_img[:, y1c:y2c, x1c:x2c]
        cH, cW = crop.shape[1], crop.shape[2]
        avail_w = (t_x2 - t_x1) - cW
        avail_h = (t_y2 - t_y1) - cH
        if avail_w < 0 or avail_h < 0:
            continue
        placed = False
        # Build lesion mask via largest connected component (same as CopyPasteWrapper).
        crop_gray = (crop.mean(dim=0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        _, thresh_crop = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_crop = cv2.dilate(thresh_crop, kernel, iterations=1)
        n_labels, labels_map = cv2.connectedComponents(thresh_crop, connectivity=8)
        if n_labels > 1:
            label_sizes = np.bincount(labels_map.ravel())
            label_sizes[0] = 0
            largest_label = int(label_sizes.argmax())
            lesion_mask = (labels_map == largest_label).astype(np.uint8)
        else:
            lesion_mask = (thresh_crop > 0).astype(np.uint8)
        if lesion_mask.sum() == 0:
            continue

        _feather_w = 5
        dist_map = cv2.distanceTransform(lesion_mask, cv2.DIST_L2, 3)
        alpha_map = np.clip(dist_map / float(_feather_w), 0.0, 1.0).astype(np.float32)
        alpha_t = torch.from_numpy(alpha_map).unsqueeze(0)  # [1, cH, cW]

        for _attempt in range(5):
            px = t_x1 + random.randint(0, avail_w)
            py = t_y1 + random.randint(0, avail_h)
            region_mean = mean_img[py:py + cH, px:px + cW]
            tissue_ratio = float((region_mean > 0.05).float().mean().item())
            if tissue_ratio >= 0.70:
                placed = True
                break
        if not placed:
            continue
        # Feathered paste.
        target_region = target_img[:, py:py + cH, px:px + cW]
        # Step A: brightness alignment (P95).
        _lesion_px_crop   = crop[:, lesion_mask.astype(bool)].reshape(-1)
        _lesion_px_target = target_region[:, lesion_mask.astype(bool)].reshape(-1)
        crop_p95   = float(torch.quantile(_lesion_px_crop,   0.95)) if _lesion_px_crop.numel() > 0 else 1e-6
        target_p95 = float(torch.quantile(_lesion_px_target, 0.95)) if _lesion_px_target.numel() > 0 else 1e-6
        brightness_scale = float(np.clip(target_p95 / max(crop_p95, 1e-6), 0.5, 2.0))
        crop_adj = (crop * brightness_scale).clamp(0.0, 1.0)
        # Step B: center darkening.
        _gamma_center = 0.75
        cy, cx = cH / 2.0, cW / 2.0
        sigma = max(cy, cx) * 0.5
        ys = torch.arange(cH, dtype=torch.float32)
        xs = torch.arange(cW, dtype=torch.float32)
        dist2 = ((ys.unsqueeze(1) - cy) ** 2 + (xs.unsqueeze(0) - cx) ** 2)
        gaussian = torch.exp(-dist2 / (2 * sigma ** 2))
        gamma_map = (1.0 - gaussian * (1.0 - _gamma_center)).unsqueeze(0)
        crop_adj = torch.pow(crop_adj.clamp(1e-6, 1.0), gamma_map)
        # Step C: blend.
        blended = crop_adj * alpha_t + target_region * (1.0 - alpha_t)
        target_img[:, py:py + cH, px:px + cW] = blended
        pasted_boxes_list.append(torch.tensor([px, py, px + cW, py + cH], dtype=torch.float32))
        actual_donors.append(donor_aug_idx)

    # Use the last actual donor for display (or first, whichever was placed).
    if actual_donors:
        show_donor_idx = actual_donors[0]
    else:
        show_donor_idx = random.choice(pos_in_aug)
    donor_img, donor_target = augmented_dataset[show_donor_idx]
    donor_rgb = tensor_to_uint8(donor_img)
    donor_rgb_boxed = draw_boxes(donor_rgb, donor_target["boxes"], BOX_COLOR_DONOR, BOX_THICKNESS)

    pasted_rgb = tensor_to_uint8(target_img)
    pasted_boxes = torch.stack(pasted_boxes_list, dim=0) if pasted_boxes_list else torch.zeros((0, 4))
    pasted_rgb_boxed = draw_boxes(pasted_rgb, pasted_boxes, BOX_COLOR_PASTE, BOX_THICKNESS)

    n_pasted = len(pasted_boxes_list)
    tl = add_label(neg_rgb, "Negative (original)")
    tr = add_label(donor_rgb_boxed, f"Donor pos (GT boxes={donor_target['boxes'].shape[0]})")
    bl = add_label(pasted_rgb, "Pasted result (no boxes drawn)")
    br = add_label(pasted_rgb_boxed, f"Pasted result ({n_pasted} GT boxes)")

    grid = make_grid(tl, tr, bl, br)

    out_path = OUT_DIR / f"sample_{sample_num:02d}.png"
    # cv2 expects BGR
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path}  (pasted boxes: {n_pasted})")

print(f"\n[test_copy_paste] Done. {len(chosen_neg)} grids saved to: {OUT_DIR}")
