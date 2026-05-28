"""
MammoPearl Deep-Learning — Dataset utilities

提供图像级二分类（has_lesion）的 PyTorch Dataset 和数据增强管线。

CSV 格式（vindr_detection_folds.csv）：
  patient_id, image_id, split, xmin, ymin, xmax, ymax, ...
  - xmin 为 NaN 表示无标注框 → 阴性
  - split 取值：training / test
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# 默认路径
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = _REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv"
DEFAULT_IMAGES_ROOT = _REPO_ROOT / "data" / "processed" / "images_png"

# ──────────────────────────────────────────────────────────────────────────────
# 辅助：从 CSV 提取图像级标签
# ──────────────────────────────────────────────────────────────────────────────

def build_image_label_df(
    csv_path: str | Path = DEFAULT_CSV,
    split: str | None = None,          # "training" / "test" / None = 全部
    fold_val: int | None = None,       # 若不为 None，用 fold 列做 train/val 划分
    is_val: bool = False,              # 配合 fold_val：True=取 fold==fold_val 作验证集
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (img_df, bbox_df)。

    img_df  列：patient_id, image_id, label (0/1), fold
    bbox_df 列：patient_id, image_id, xmin, ymin, xmax, ymax
              （原始坐标，未缩放；阴性图像无对应行）
    """

    df = pd.read_csv(csv_path, low_memory=False)

    if split is not None:
        df = df[df["split"].str.lower() == split.lower()]

    # 图像级聚合：只要有一行 xmin 不为 NaN 即为阳性
    img_df = (
        df.groupby(["patient_id", "image_id"], sort=False)
        .agg(
            label=("xmin", lambda x: int(x.notna().any())),
            fold=("fold", "first"),
        )
        .reset_index()
    )

    if fold_val is not None:
        if is_val:
            img_df = img_df[img_df["fold"] == fold_val]
        else:
            img_df = img_df[img_df["fold"] != fold_val]

    # bbox_df：只保留有效框行，供 GT mask 生成使用
    bbox_cols = ["patient_id", "image_id", "xmin", "ymin", "xmax", "ymax"]
    bbox_df = (
        df[bbox_cols].dropna(subset=["xmin", "ymin", "xmax", "ymax"])
        .copy()
        .reset_index(drop=True)
    )
    # 过滤到与 img_df 相同的图像集合
    valid_keys = set(zip(img_df["patient_id"], img_df["image_id"]))
    mask = bbox_df.apply(
        lambda r: (r["patient_id"], r["image_id"]) in valid_keys, axis=1
    )
    bbox_df = bbox_df[mask].reset_index(drop=True)

    return img_df.reset_index(drop=True), bbox_df


# ──────────────────────────────────────────────────────────────────────────────
# 图像加载
# ──────────────────────────────────────────────────────────────────────────────

def _load_gray(
    patient_id: str,
    image_id: str,
    images_root: Path,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """加载单张图像为 (H, W, 3) uint8，等比填充到 target_h × target_w。"""

    for candidate in [
        images_root / patient_id / image_id,
        images_root / patient_id / Path(image_id).name,
        images_root / patient_id / f"{Path(image_id).stem}.png",
    ]:
        if candidate.exists():
            path = candidate
            break
    else:
        raise FileNotFoundError(f"Image not found: {patient_id}/{image_id}")

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cv2 failed to decode: {path}")

    # 转灰度
    if img.ndim == 2:
        gray = img
    elif img.shape[2] >= 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 等比缩放，短边用 0 填充（AR-preserving letterbox）
    h, w = gray.shape
    scale = min(target_h / h, target_w / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=resized.dtype)
    pad_top = (target_h - nh) // 2
    pad_left = (target_w - nw) // 2
    canvas[pad_top: pad_top + nh, pad_left: pad_left + nw] = resized

    # 归一化到 0–255 uint8，确保一致
    if canvas.dtype != np.uint8:
        canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 复制为 3 通道（EfficientNet 需要 RGB）
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return rgb  # (H, W, 3) uint8


def _load_original_hw(
    patient_id: str,
    image_id: str,
    images_root: Path,
) -> tuple[int, int]:
    """返回原始图像的 (height, width)，不做任何缩放。"""
    for candidate in [
        images_root / patient_id / image_id,
        images_root / patient_id / Path(image_id).name,
        images_root / patient_id / f"{Path(image_id).stem}.png",
    ]:
        if candidate.exists():
            raw = np.fromfile(str(candidate), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.shape[0], img.shape[1]
    raise FileNotFoundError(f"Image not found: {patient_id}/{image_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _build_gt_mask(
    patient_id: str,
    image_id: str,
    bbox_df: pd.DataFrame,
    orig_h: int,
    orig_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """根据原始 bbox 坐标生成与 letterbox 缩放后图像对齐的 GT mask。

    返回 (target_h, target_w) float32，值域 [0, 1]，bbox 内为 1，外为 0。
    """
    scale = min(target_h / orig_h, target_w / orig_w)
    nh = int(round(orig_h * scale))
    nw = int(round(orig_w * scale))
    pad_top  = (target_h - nh) // 2
    pad_left = (target_w - nw) // 2

    mask = np.zeros((target_h, target_w), dtype=np.float32)

    rows = bbox_df[
        (bbox_df["patient_id"] == patient_id) &
        (bbox_df["image_id"]   == image_id)
    ]
    for _, r in rows.iterrows():
        x1 = int(round(float(r["xmin"]) * scale)) + pad_left
        y1 = int(round(float(r["ymin"]) * scale)) + pad_top
        x2 = int(round(float(r["xmax"]) * scale)) + pad_left
        y2 = int(round(float(r["ymax"]) * scale)) + pad_top
        x1, x2 = max(0, x1), min(target_w, x2)
        y1, y2 = max(0, y1), min(target_h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    return mask


class MammoDataset(Dataset):
    """图像级二分类 Dataset。

    Parameters
    ----------
    img_df      : build_image_label_df() 返回的第一个元素
    images_root : 预处理图像根目录
    input_h/w   : 网络输入尺寸
    augment     : 是否启用训练增强
    use_gt_mask : 若为 True，在训练集上将 GT bbox 作为第 4 通道附加到输入
                  （仅对 split=training 有效；test 时无 GT，自动退化为 3 通道）
    bbox_df     : build_image_label_df() 返回的第二个元素，use_gt_mask=True 时必填
    mean/std    : ImageNet 归一化参数（3 通道）
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        img_df: pd.DataFrame,
        images_root: str | Path = DEFAULT_IMAGES_ROOT,
        input_h: int = 512,
        input_w: int = 512,
        augment: bool = False,
        use_gt_mask: bool = False,
        bbox_df: pd.DataFrame | None = None,
    ) -> None:
        self.img_df = img_df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment
        self.use_gt_mask = use_gt_mask
        self.bbox_df = bbox_df if bbox_df is not None else pd.DataFrame()

        if use_gt_mask and self.bbox_df.empty:
            raise ValueError("use_gt_mask=True 时必须提供 bbox_df。")

    @property
    def in_channels(self) -> int:
        """返回实际输入通道数（3 或 4）。"""
        return 4 if self.use_gt_mask else 3

    def __len__(self) -> int:
        return len(self.img_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.img_df.iloc[idx]
        pid = str(row["patient_id"])
        iid = str(row["image_id"])

        rgb = _load_gray(
            pid, iid, self.images_root, self.input_h, self.input_w,
        )  # (H, W, 3) uint8

        # 生成 GT mask 通道（缩放前需要原始图像尺寸）
        if self.use_gt_mask:
            orig = _load_original_hw(pid, iid, self.images_root)
            gt_mask = _build_gt_mask(
                pid, iid, self.bbox_df,
                orig[0], orig[1],
                self.input_h, self.input_w,
            )  # (H, W) float32 [0,1]
        else:
            gt_mask = None

        if self.augment:
            rgb, gt_mask = self._augment(rgb, gt_mask)

        # Normalize RGB to float32 [C, H, W]
        img = rgb.astype(np.float32) / 255.0
        mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(self.IMAGENET_STD,  dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        if gt_mask is not None:
            mask_t = torch.from_numpy(gt_mask[np.newaxis, :, :])  # (1, H, W)
            tensor = torch.cat([tensor, mask_t], dim=0)           # (4, H, W)

        label = int(row["label"])
        return tensor, label

    # ------------------------------------------------------------------
    def _augment(
        self,
        rgb: np.ndarray,
        gt_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """轻量级训练增强（对 rgb 和 gt_mask 同步变换）。"""
        flip_h = np.random.rand() < 0.5
        flip_v = np.random.rand() < 0.3
        do_rot = np.random.rand() < 0.4
        angle  = np.random.uniform(-10, 10) if do_rot else 0.0

        # 随机水平翻转
        if flip_h:
            rgb = cv2.flip(rgb, 1)
            if gt_mask is not None:
                gt_mask = cv2.flip(gt_mask, 1)

        # 随机垂直翻转
        if flip_v:
            rgb = cv2.flip(rgb, 0)
            if gt_mask is not None:
                gt_mask = cv2.flip(gt_mask, 0)

        # 随机亮度 / 对比度抖动（仅 RGB）
        alpha = np.random.uniform(0.85, 1.15)
        beta  = np.random.uniform(-15.0, 15.0)
        rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 随机小角度旋转
        if do_rot:
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            if gt_mask is not None:
                gt_mask = cv2.warpAffine(
                    gt_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
                )

        return rgb, gt_mask


# ──────────────────────────────────────────────────────────────────────────────
# 正样本权重（用于 WeightedRandomSampler / pos_weight）
# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(img_df: pd.DataFrame) -> float:
    """返回 BCEWithLogitsLoss pos_weight = neg_count / pos_count。"""
    pos = int(img_df["label"].sum())
    neg = len(img_df) - pos
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def compute_sample_weights(img_df: pd.DataFrame) -> list[float]:
    """返回每个样本的采样权重，用于 WeightedRandomSampler（平衡正负）。"""
    pos = int(img_df["label"].sum())
    neg = len(img_df) - pos
    w_pos = 1.0 / pos if pos > 0 else 1.0
    w_neg = 1.0 / neg if neg > 0 else 1.0
    return [w_pos if lbl == 1 else w_neg for lbl in img_df["label"]]
