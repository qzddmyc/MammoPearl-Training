r"""
MammoPearl Deep-Learning — Stage 1 测试集评估脚本

功能：
  1. 在 test split 上计算多阈值 Recall / Precision / F1 / F2
  2. 可选：输出每张图的预测概率 CSV
  3. 可选：使用 GradCAM 生成热图并保存到指定目录
     需要安装：pip install grad-cam  (pytorch-grad-cam)

─────────────────────────────────────────────────────────────────────────────
运行命令（基础）：

python src/data/deep-learning/clf-test.py \
    --ckpt-path models/clf_efficientnet_b4.pth

附带预测 CSV 输出：

python src/data/deep-learning/clf-test.py \
    --ckpt-path models/clf_efficientnet_b4.pth \
    --output-csv tmp/clf_preds.csv

附带 GradCAM 可视化：

python src/data/deep-learning/clf-test.py \
    --ckpt-path models/clf_efficientnet_b4.pth \
    --gradcam-output-dir tmp/clf_gradcam \
    --gradcam-max-samples 50

─────────────────────────────────────────────────────────────────────────────
参数注意事项：

--ckpt-path
    必填。指定 clf-train.py 保存的 .pth checkpoint 路径。
    脚本自动从 checkpoint 读取 input_h/w、amp、in_channels，无需手动指定。

--thresholds 0.1 0.2 0.3 0.5 0.7
    评估的概率阈值列表，默认 0.1–0.9 全评。
    召回率关键区间一般在 0.1–0.3，可缩小范围加速评估。

--output-csv tmp/clf_preds.csv
    保存每张测试图的 patient_id、image_id、GT 标签和预测概率，
    供后续分析（如 ROC 曲线、最优阈值搜索）使用。
    **若需要将 Stage 1 结果送入 Stage 2（clf2-test.py --stage1-pred-csv），
    必须显式传入此参数；默认不生成 CSV。**

--gradcam-output-dir tmp/clf_gradcam
    对得分 ≥ --gradcam-score-threshold 的图生成 GradCAM 叠加可视化。
    输出文件名格式：<patient_id>_<image_stem>_prob<x.xx>_gt<0/1>.jpg
    注意：需先安装 grad-cam（pip install grad-cam）。

--gradcam-score-threshold 0.5
    只对此阈值以上的图生成 GradCAM（避免对全部阴性图生成热图浪费时间）。

注意：
  1. 测试集无 GT bbox，不可使用 --use-gt-mask 模式训练的模型在此处手动指定 4
     通道——脚本会自动从 checkpoint 读取 in_channels 并正确重建模型。
  2. 若 checkpoint 由 --use-gt-mask 训练得到，测试时 in_channels=4，
     但 test split 无 GT mask，第 4 通道将全为 0（即"无先验"状态），
     这是评估 GT mask 辅助的泛化性的合理方式。

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import importlib.util as _ilu

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset import (
    DEFAULT_CSV,
    DEFAULT_IMAGES_ROOT,
    MammoDataset,
    build_image_label_df,
)

# clf-train.py 含连字符，不能直接 import，用 spec 加载
_spec = _ilu.spec_from_file_location("clf_train", _HERE / "clf-train.py")
_clf_train = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_clf_train)  # type: ignore[union-attr]
build_model = _clf_train.build_model
evaluate    = _clf_train.evaluate

# ──────────────────────────────────────────────────────────────────────────────
# GradCAM（可选依赖）
# ──────────────────────────────────────────────────────────────────────────────

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    # Reverse ImageNet normalization: (H, W, 3) uint8 RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_gradcam(
    model: torch.nn.Module,
    dataset: MammoDataset,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 50,
    score_threshold: float = 0.5,
) -> None:
    # Generate and save GradCAM overlays for images with score >= score_threshold

    if not _GRADCAM_AVAILABLE:
        print("[GradCAM] pytorch-grad-cam 未安装，跳过可视化。"
              "  安装方法：pip install grad-cam")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # EfficientNet-B4 最后一个 Conv block
    target_layers = [model.features[-1][0]]
    cam = GradCAM(model=model, target_layers=target_layers)

    model.eval()
    saved = 0

    for idx in range(len(dataset)):
        if saved >= max_samples:
            break

        tensor, label = dataset[idx]
        input_tensor = tensor.unsqueeze(0).to(device)

        logit = model(input_tensor).squeeze().item()
        prob  = 1.0 / (1.0 + np.exp(-logit))

        if prob < score_threshold:
            continue

        grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)
        rgb_img = _denormalize(tensor).astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        row = dataset.img_df.iloc[idx]
        fname = f"{row['patient_id']}_{Path(str(row['image_id'])).stem}_prob{prob:.2f}_gt{label}.jpg"
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"[GradCAM] Saved {saved} images to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MammoPearl clf-test")

    p.add_argument("--ckpt-path",      required=True)
    p.add_argument("--csv-path",       default=str(DEFAULT_CSV))
    p.add_argument("--images-root",    default=str(DEFAULT_IMAGES_ROOT))
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--num-workers",    type=int, default=4)

    # 评估阈值
    p.add_argument("--thresholds",     nargs="+", type=float,
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   help="评估用的概率阈值列表。")

    # 输出
    p.add_argument("--output-csv",     default=None,
                   help="若指定，保存每张图的预测概率到此 CSV 文件。")

    # GradCAM
    p.add_argument("--gradcam-output-dir", default=None,
                   help="若指定，保存 GradCAM 热图到此目录。")
    p.add_argument("--gradcam-max-samples", type=int, default=50)
    p.add_argument("--gradcam-score-threshold", type=float, default=0.5,
                   help="只对得分 ≥ 此阈值的图生成 GradCAM。")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载 checkpoint ─────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    input_h    = saved_args.get("input_h", 512)
    input_w    = saved_args.get("input_w", 512)
    amp        = saved_args.get("amp", False)
    in_channels = ckpt.get("in_channels", 3)

    model = build_model(pretrained=False, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    best_ep = ckpt.get("epoch", "?")
    best_f2 = ckpt.get("best_f2", float("nan"))
    print(f"Checkpoint: epoch={best_ep}, best_val_F2={best_f2:.4f}")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    test_df, _ = build_image_label_df(args.csv_path, split="test")
    print(f"Test images: {len(test_df)} "
          f"(pos={test_df['label'].sum()}, neg={len(test_df)-test_df['label'].sum()})")

    test_ds = MammoDataset(
        test_df, args.images_root, input_h, input_w, augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 评估 ──────────────────────────────────────────────────────────────
    results = evaluate(model, test_loader, device, args.thresholds, amp)

    _SEP = "─" * 66
    print(f"\n{_SEP}")
    print("Test-set evaluation")
    print(f"  {'Thr':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'F2':>7}  "
          f"{'TP':>5}  {'FP':>6}  {'FN':>4}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  "
          f"{'─'*5}  {'─'*6}  {'─'*4}")
    for thr in sorted(results.keys()):
        m = results[thr]
        print(f"  {thr:>5.2f}  {m['recall']:>7.4f}  {m['prec']:>7.4f}  "
              f"{m['f1']:>7.4f}  {m['f2']:>7.4f}  "
              f"{m['tp']:>5}  {m['fp']:>6}  {m['fn']:>4}")
    print(_SEP)

    # ── 每图预测 CSV ─────────────────────────────────────────────────────
    if args.output_csv:
        _save_pred_csv(model, test_ds, device, amp, Path(args.output_csv))

    # ── GradCAM ───────────────────────────────────────────────────────────
    if args.gradcam_output_dir:
        save_gradcam(
            model, test_ds, device,
            Path(args.gradcam_output_dir),
            args.gradcam_max_samples,
            args.gradcam_score_threshold,
        )


def _save_pred_csv(
    model: torch.nn.Module,
    dataset: MammoDataset,
    device: torch.device,
    amp: bool,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            tensor, label = dataset[idx]
            inp = tensor.unsqueeze(0).to(device)
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logit = model(inp).squeeze().item()
            else:
                logit = model(inp).squeeze().item()
            prob = 1.0 / (1.0 + np.exp(-logit))
            row_data = dataset.img_df.iloc[idx]
            rows.append({
                "patient_id": row_data["patient_id"],
                "image_id":   row_data["image_id"],
                "label":      label,
                "prob":       f"{prob:.6f}",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "image_id", "label", "prob"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Output CSV] Saved {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
