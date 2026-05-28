r"""
MammoPearl Deep-Learning — Stage 1 图像级病灶筛查训练脚本

模型：EfficientNet-B4（torchvision，ImageNet 预训练）
任务：二分类 has_lesion（0/1），BCEWithLogitsLoss + pos_weight
监控指标：val F2（召回率权重高，漏检惩罚大）

硬件环境：NVIDIA GeForce RTX 4090，24 GB VRAM

─────────────────────────────────────────────────────────────────────────────
运行命令（基础）：

python src/data/deep-learning/clf-train.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 512 \
    --input-w 512 \
    --fold-val 0 \
    --patience 8 \
    --save-path models/clf_efficientnet_b4.pth \
    --amp \
    --augment \
    --hide-progress-bar

运行命令（附带 GT mask 第 4 通道）：

python src/data/deep-learning/clf-train.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 512 \
    --input-w 512 \
    --fold-val 0 \
    --patience 8 \
    --save-path models/clf_efficientnet_b4_mask.pth \
    --amp \
    --augment \
    --use-gt-mask \
    --hide-progress-bar

─────────────────────────────────────────────────────────────────────────────
参数注意事项：

--fold-val 0
    从 training split 中把 fold==0 的图像作为验证集，其余 fold 用于训练。
    VinDr CSV 共 5 个 fold（0–4），默认使用 fold 0 验证。

--ref-score 0.5
    以此概率阈值对应的 F2 作为 early stop 监控指标。
    设置高阈值（如 0.5–0.7）可使 checkpoint 对应较精确的模型；
    设置低阈值（如 0.2–0.3）可使 checkpoint 对应高召回模型。
    推荐在首次训练时先用 0.5，再根据测试集结果调整。

--encoder-lr-multiplier 0.1
    backbone 的学习率为 lr × multiplier，分类头使用完整 lr。
    过大会破坏预训练权重；过小会导致 backbone 拟合慢。

--use-gt-mask
    启用后将 GT bbox 生成的二值 mask 作为第 4 输入通道附加到图像。
    优点：给模型提供"应关注哪里"的空间先验，可能提升 F2。
    注意：
      1. 此模式仅在 training split 有效（test split 无 GT bbox，不可使用）；
      2. 启用后首层卷积变为 4 通道，保存的 .pth 与 3 通道版本不互通；
      3. 测试时 clf-test.py 自动从 checkpoint 读取 in_channels，不需要手动指定；
      4. 推理阶段（真实部署）无 GT bbox，此模式只适用于衡量 GT 引导效果的上界实验。

--amp
    启用 bfloat16 混合精度训练，可减少约 30–40% 显存占用和训练时间。
    需要 CUDA 设备支持 bfloat16（RTX 30/40 系及以上）。

--augment
    启用随机翻转、对比度抖动、小角度旋转等轻量数据增强。
    乳腺钼靶影像通常只做水平翻转（LR 对称），但垂直翻转和旋转在本数据集
    上影响有限，可视需要通过修改 _augment() 函数来关闭垂直翻转。

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# 本地模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (
    DEFAULT_CSV,
    DEFAULT_IMAGES_ROOT,
    MammoDataset,
    build_image_label_df,
    compute_pos_weight,
    compute_sample_weights,
)

# ──────────────────────────────────────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    pretrained: bool = True,
    medical_backbone_path: str | None = None,
    in_channels: int = 3,
) -> nn.Module:
    """构建 EfficientNet-B4，替换分类头为单输出（logit）。

    Parameters
    ----------
    in_channels : 3 表示普通 RGB 输入；4 表示 RGB + GT mask 输入。
                  当为 4 时，首层卷积核展展至 4 通道，预训练权重的前 3 项保留。
    """

    weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = efficientnet_b4(weights=weights)

    if in_channels != 3:
        # 获取首层卷积层（原始为 3 通道输入）
        first_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            # 复制预训练的前 3 通道权重
            new_conv.weight[:, :3, :, :] = first_conv.weight
            if in_channels > 3:
                # 新增的通道用前 3 项的均均初始化
                new_conv.weight[:, 3:, :, :] = first_conv.weight[:, :in_channels - 3, :, :]
        model.features[0][0] = new_conv

    # 替换最终分类层
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 1),
    )

    if medical_backbone_path:
        path = Path(medical_backbone_path)
        if path.exists():
            state = torch.load(str(path), map_location="cpu")
            # 尝试加载 backbone 权重（仅 features 部分）
            backbone_state = {k: v for k, v in state.items() if k.startswith("features.")}
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            print(f"[Medical backbone] loaded {len(backbone_state)} keys "
                  f"| missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[Warning] Medical backbone not found: {path}")

    return model


def get_param_groups(model: nn.Module, lr: float, encoder_lr_multiplier: float) -> list[dict]:
    """分组：分类头用 lr，backbone 用 lr * multiplier。"""

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": lr * encoder_lr_multiplier},
        {"params": head_params,     "lr": lr},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────────────────────────────────

def fbeta_score(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float] | None = None,
    amp: bool = False,
) -> dict:
    """在 loader 上评估模型，返回多阈值指标字典。"""

    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(images).squeeze(1)
            else:
                logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    results: dict = {}
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = fbeta_score(tp, fp, fn, beta=1.0)
        f2   = fbeta_score(tp, fp, fn, beta=2.0)
        results[thr] = dict(tp=tp, fp=fp, fn=fn, prec=prec, recall=rec, f1=f1, f2=f2)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    hide_progress: bool,
) -> float:
    model.train()
    total_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        if amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        if not hide_progress:
            # 每 25% 进度打印一次
            milestones = {max(1, len(loader) * k // 4) for k in range(1, 5)}
            if (i + 1) in milestones:
                pct = (i + 1) / len(loader) * 100
                bar = "=" * int(30 * (i + 1) / len(loader))
                print(f"  [{i+1:4d}/{len(loader)}] [{bar:<30}]  {pct:5.1f}%  loss={loss.item():.4f}", flush=True)

    return total_loss / max(len(loader), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MammoPearl clf-train")

    # 数据
    p.add_argument("--csv-path",       default=str(DEFAULT_CSV))
    p.add_argument("--images-root",    default=str(DEFAULT_IMAGES_ROOT))
    p.add_argument("--input-h",        type=int, default=512)
    p.add_argument("--input-w",        type=int, default=512)
    p.add_argument("--fold-val",       type=int, default=0,
                   help="用 fold==fold_val 的图像作为验证集（默认 0）。")

    # 训练
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch-size",     type=int, default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--patience",       type=int, default=8,
                   help="val F2 连续 N 个 epoch 无提升则 early stop。")
    p.add_argument("--augment",        action="store_true")
    p.add_argument("--amp",            action="store_true",
                   help="启用 bfloat16 混合精度训练。")

    # 模型
    p.add_argument("--medical-backbone-path", default=None,
                   help="可选：预训练医疗影像 backbone 权重路径。")
    p.add_argument("--save-path",      default="models/clf_efficientnet_b4.pth")

    # 其他
    p.add_argument("--num-workers",    type=int, default=4)
    p.add_argument("--hide-progress-bar", action="store_true")

    # 评估阈值（监控 ref-score 对应的 F2）
    p.add_argument("--ref-score",      type=float, default=0.5,
                   help="用于 early stop 监控的概率阈值（默认 0.5）。")
    # GT mask 辅助输入
    p.add_argument("--use-gt-mask", action="store_true",
                   help="开启后将 GT bbox 生成的 mask 作为第 4 通道附加到模型输入。"
                        "仅对 training split 有效；test 时无 GT，不可开启。")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    train_df, train_bbox_df = build_image_label_df(args.csv_path, split="training",
                                                   fold_val=args.fold_val, is_val=False)
    val_df,   val_bbox_df   = build_image_label_df(args.csv_path, split="training",
                                                   fold_val=args.fold_val, is_val=True)

    print(f"Train images: {len(train_df)} (pos={train_df['label'].sum()}, "
          f"neg={len(train_df)-train_df['label'].sum()})")
    print(f"Val   images: {len(val_df)}   (pos={val_df['label'].sum()}, "
          f"neg={len(val_df)-val_df['label'].sum()})")
    if args.use_gt_mask:
        print("[GT mask] 开启：将 GT bbox 作为第 4 通道附加到模型输入。")

    train_ds = MammoDataset(
        train_df, args.images_root,
        args.input_h, args.input_w,
        augment=args.augment,
        use_gt_mask=args.use_gt_mask,
        bbox_df=train_bbox_df if args.use_gt_mask else None,
    )
    val_ds = MammoDataset(
        val_df, args.images_root,
        args.input_h, args.input_w,
        augment=False,
        use_gt_mask=args.use_gt_mask,
        bbox_df=val_bbox_df if args.use_gt_mask else None,
    )

    sample_weights = compute_sample_weights(train_df)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    in_channels = train_ds.in_channels
    model = build_model(
        pretrained=True,
        medical_backbone_path=args.medical_backbone_path,
        in_channels=in_channels,
    )
    model.to(device)

    # ── 损失函数 ────────────────────────────────────────────────────────────
    # WeightedRandomSampler 已在采样层面平衡正负，pos_weight 设为 1.0 避免双重过补偿
    # （若未使用 sampler，可将 pos_weight 改回 compute_pos_weight(train_df)）
    pos_weight = torch.tensor([1.0], device=device)
    print(f"pos_weight = {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── 优化器 ──────────────────────────────────────────────────────────────
    param_groups = get_param_groups(model, args.lr, args.encoder_lr_multiplier)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # ── 训练循环 ────────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 全正基线 F2（Recall=1, FP=val_neg, FN=0），用于过滤退化态
    _val_pos = int(val_df["label"].sum())
    _val_neg = len(val_df) - _val_pos
    trivial_f2 = 5 * _val_pos / (5 * _val_pos + _val_neg)  # beta=2

    best_f2 = -1.0
    no_improve = 0
    eval_thresholds = [0.1, 0.2, 0.3, 0.5, args.ref_score]
    eval_thresholds = sorted(set(eval_thresholds))

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, args.amp, args.hide_progress_bar,
        )
        scheduler.step()

        results = evaluate(model, val_loader, device, eval_thresholds, args.amp)
        elapsed = time.time() - t0

        # ── 表格输出 ────────────────────────────────────────────────
        _SEP = "─" * 66
        print(f"\n{_SEP}", flush=True)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d}  "
            f"loss={train_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[1]:.2e}  "
            f"{elapsed:.0f}s",
            flush=True,
        )
        print(f"  {'Thr':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'F2':>7}  "
              f"{'TP':>5}  {'FP':>6}  {'FN':>4}", flush=True)
        print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  "
              f"{'─'*5}  {'─'*6}  {'─'*4}", flush=True)
        for thr, m in results.items():
            marker = "  <- ref" if thr == args.ref_score else ""
            print(
                f"  {thr:>5.2f}  {m['recall']:>7.4f}  {m['prec']:>7.4f}  "
                f"{m['f1']:>7.4f}  {m['f2']:>7.4f}  "
                f"{m['tp']:>5}  {m['fp']:>6}  {m['fn']:>4}{marker}",
                flush=True,
            )

        # 只取 F2 > 全正基线的阈值（真正有区分能力），全部退化则计入 patience
        valid_results = {thr: m for thr, m in results.items() if m["f2"] > trivial_f2}

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[1],
            **{f"val_recall@{thr}": results[thr]["recall"] for thr in eval_thresholds},
            **{f"val_f2@{thr}": results[thr]["f2"] for thr in eval_thresholds},
        }
        history.append(epoch_log)

        if valid_results:
            best_thr, best_thr_metrics = max(valid_results.items(), key=lambda kv: kv[1]["f2"])
            best_epoch_f2 = best_thr_metrics["f2"]
            if best_epoch_f2 > best_f2:
                best_f2 = best_epoch_f2
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f2": best_f2,
                    "best_thr": best_thr,
                    "in_channels": in_channels,
                    "args": vars(args),
                    "history": history,
                }, str(save_path))
                print(f"  ★ New best F2={best_f2:.4f} @{best_thr:.2f}  → Saved: {save_path}", flush=True)
            else:
                no_improve += 1
                print(f"  patience {no_improve}/{args.patience}", flush=True)
        else:
            no_improve += 1
            print(f"  (degenerate)  patience {no_improve}/{args.patience}", flush=True)

        if no_improve >= args.patience:
            print(f"  Early stop: no improvement for {args.patience} epochs.")
            break
        print(_SEP, flush=True)

    print(f"\nTraining complete. Best F2={best_f2:.4f}")
    print(f"Checkpoint: {save_path}")

    # 保存完整历史
    hist_path = save_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_path}")


if __name__ == "__main__":
    main()
