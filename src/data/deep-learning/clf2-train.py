r"""
MammoPearl Deep-Learning -- Stage 2 病变类型分类训练脚本

Stage 2 对图像进行四分类病变类型识别（基于 finding_categories 标签）：
  - 类别 0：No Finding（无病变；也用于接收 Stage 1 的误报）
  - 类别 1：Mass（肿块）
  - 类别 2：Calcification（可疑钙化）
  - 类别 3：Asymmetry_Distortion（不对称 / 结构扭曲，含 Skin_Other）

多病变图像按优先级取主要类型：Mass > Calcification > Asymmetry_Distortion。

在完整流水线中，Stage 1 负责粗筛（高召回），Stage 2 对疑似阳性图像进行
精细的病变类型分类。训练时使用全部 training split（含阴性），使得 Stage 2
也能将 Stage 1 的误报重新归类为 No Finding。

------------------------------------------------------------------
运行命令（基础）：

python src/data/deep-learning/clf2-train.py \
    --epochs 30 --batch-size 16 --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 512 --input-w 512 \
    --fold-val 0 --patience 10 \
    --save-path models/clf2_efficientnet_b4.pth \
    --amp --augment

------------------------------------------------------------------
参数注意事项：

--save-path
    Stage 2 模型保存路径，请与 Stage 1 的路径区分（默认名称不同）。

--patience 10
    Stage 2 收敛通常比 Stage 1 慢（类别更多、样本更少），
    建议 patience >= 10。

--encoder-lr-multiplier 0.1
    backbone 学习率 = lr * multiplier；分类头使用全速 lr。
    较小的 multiplier 可防止预训练权重被破坏。

注意：
  1. Stage 2 在全部 training split 上训练，不预先过滤 Stage 1 输出。
     在推理阶段（clf2-test.py）再结合 Stage 1 阈值过滤。
  2. 类别极度不平衡（No Finding:Mass:Calc:Asym ≈ 44:2.5:0.7:1），
     WeightedRandomSampler 在代码中处理，不额外使用 class weight。
  3. Checkpoint 判据：类别 1/2/3（病变类型）的 macro F1，
     No Finding（类别 0）不参与判据计算。

------------------------------------------------------------------
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
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset import (
    DEFAULT_CSV,
    DEFAULT_IMAGES_ROOT,
    MammoDataset,
    build_lesion_type_df,
    compute_sample_weights_multiclass,
)

# ──────────────────────────────────────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 4
CLASS_NAMES = {0: "None", 1: "Mass", 2: "Calc", 3: "Asym"}


def build_stage2_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """构建 EfficientNet-B4，替换分类头为 num_classes 输出的 logit 向量。"""
    weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_param_groups(model: nn.Module, lr: float, encoder_lr_multiplier: float) -> list[dict]:
    """backbone 用 lr * multiplier，分类头用 lr。"""
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
# 评估
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_stage2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> dict:
    """在 loader 上做三分类评估，返回每类的 TP/FP/FN/Recall/Prec/F1。"""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(images)
            else:
                logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    results: dict = {}
    for c in range(NUM_CLASSES):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = dict(tp=tp, fp=fp, fn=fn, n=tp + fn, prec=prec, recall=rec, f1=f1)

    results["macro_f1"]     = sum(results[c]["f1"] for c in range(NUM_CLASSES)) / NUM_CLASSES
    # 目标指标：病变类型 1/2/3 的 macro F1（排除 No Finding）
    results["target_score"] = sum(results[c]["f1"] for c in range(1, NUM_CLASSES)) / (NUM_CLASSES - 1)
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
        labels = labels.long().to(device)   # CrossEntropyLoss 需要 long

        optimizer.zero_grad()

        if amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        if not hide_progress:
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
    p = argparse.ArgumentParser(description="MammoPearl clf2-train (Stage 2)")

    # 数据
    p.add_argument("--csv-path",    default=str(DEFAULT_CSV))
    p.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    p.add_argument("--input-h",     type=int, default=512)
    p.add_argument("--input-w",     type=int, default=512)
    p.add_argument("--fold-val",    type=int, default=0)

    # 训练
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--augment",      action="store_true")
    p.add_argument("--amp",          action="store_true")

    # 模型
    p.add_argument("--save-path",    default="models/clf2_efficientnet_b4.pth")

    # 其他
    p.add_argument("--num-workers",        type=int, default=4)
    p.add_argument("--hide-progress-bar",  action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    train_df, _ = build_lesion_type_df(args.csv_path, split="training",
                                       fold_val=args.fold_val, is_val=False)
    val_df,   _ = build_lesion_type_df(args.csv_path, split="training",
                                       fold_val=args.fold_val, is_val=True)

    def _dist(df: pd.DataFrame) -> str:
        return (f"none={int((df['label']==0).sum())}  "
                f"mass={int((df['label']==1).sum())}  "
                f"calc={int((df['label']==2).sum())}  "
                f"asym={int((df['label']==3).sum())}")

    print(f"Train: {len(train_df)}  ({_dist(train_df)})")
    print(f"Val:   {len(val_df)}    ({_dist(val_df)})")

    train_ds = MammoDataset(train_df, args.images_root, args.input_h, args.input_w,
                            augment=args.augment)
    val_ds   = MammoDataset(val_df,   args.images_root, args.input_h, args.input_w,
                            augment=False)

    sample_weights = compute_sample_weights_multiclass(train_df)
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

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = build_stage2_model(pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    # WeightedRandomSampler 已平衡三类，CrossEntropyLoss 使用均匀 weight
    criterion = nn.CrossEntropyLoss()

    # ── 优化器 ──────────────────────────────────────────────────────────────
    param_groups = get_param_groups(model, args.lr, args.encoder_lr_multiplier)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    no_improve  = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, args.amp, args.hide_progress_bar,
        )
        scheduler.step()

        results = evaluate_stage2(model, val_loader, device, args.amp)
        elapsed = time.time() - t0

        _SEP = "─" * 66
        print(f"\n{_SEP}", flush=True)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d}  "
            f"loss={train_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[1]:.2e}  "
            f"{elapsed:.0f}s",
            flush=True,
        )
        print(f"  {'Class':>5}  {'N':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  "
              f"{'TP':>5}  {'FP':>6}  {'FN':>4}", flush=True)
        print(f"  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  "
              f"{'─'*5}  {'─'*6}  {'─'*4}", flush=True)
        for c in range(NUM_CLASSES):
            m = results[c]
            print(
                f"  {CLASS_NAMES[c]:>5}  {m['n']:>5}  {m['recall']:>7.4f}  {m['prec']:>7.4f}  "
                f"{m['f1']:>7.4f}  {m['tp']:>5}  {m['fp']:>6}  {m['fn']:>4}",
                flush=True,
            )
        macro  = results["macro_f1"]
        target = results["target_score"]
        print(f"  Macro F1={macro:.4f}  |  Target (macro F1 of lesion types)={target:.4f}", flush=True)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[1],
            **{f"val_recall_c{c}": results[c]["recall"] for c in range(NUM_CLASSES)},
            **{f"val_f1_c{c}": results[c]["f1"] for c in range(NUM_CLASSES)},
            "val_macro_f1": macro,
            "val_target_score": target,
        }
        history.append(epoch_log)

        if target > best_score:
            best_score = target
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "num_classes": NUM_CLASSES,
                "args": vars(args),
                "history": history,
            }, str(save_path))
            print(f"  ★ New best score={best_score:.4f}  → Saved: {save_path}", flush=True)
        else:
            no_improve += 1
            print(f"  patience {no_improve}/{args.patience}", flush=True)
            if no_improve >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs.")
                break
        print(_SEP, flush=True)

    print(f"\nTraining complete. Best target score={best_score:.4f}")
    print(f"Checkpoint: {save_path}")

    hist_path = save_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_path}")


# ── 延迟导入（避免顶层导入 pandas 影响启动速度）──────────────────────────────
import pandas as pd  # noqa: E402

if __name__ == "__main__":
    main()
