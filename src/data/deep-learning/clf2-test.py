r"""
MammoPearl Deep-Learning -- Stage 2 条件分类器测试集评估脚本

对测试集阳性图像（有病变）进行 3 类条件病变类型评估，可选：
  1. 结合 Stage 1 预测 CSV，只评估 Stage 1 通过的阳性图像
  2. 保存每张图的预测结果 CSV

类别（条件分类器，仅阳性图像）：0=Mass, 1=Calcification, 2=Asymmetry_Distortion

------------------------------------------------------------------
运行命令（基础，全量阳性图评估）：

python src/data/deep-learning/clf2-test.py \
    --ckpt-path models/clf2_cond_efficientnet_b4.pth

结合 Stage 1 过滤（推荐，模拟完整流水线）：

python src/data/deep-learning/clf2-test.py \
    --ckpt-path models/clf2_cond_efficientnet_b4.pth \
    --stage1-pred-csv tmp/clf_preds.csv \
    --stage1-threshold 0.1

附带预测 CSV 输出：

python src/data/deep-learning/clf2-test.py \
    --ckpt-path models/clf2_cond_efficientnet_b4.pth \
    --output-csv tmp/clf2_preds.csv

------------------------------------------------------------------
参数注意事项：

--stage1-pred-csv tmp/clf_preds.csv
    Stage 1 预测 CSV（由 clf-test.py --output-csv 生成）。
    包含列：patient_id, image_id, label, prob
    提供后脚本只评估 prob >= --stage1-threshold 且本身有病变的图像，
    模拟完整 Stage 1 -> Stage 2 流水线中 Stage 2 的分类性能。

--stage1-threshold 0.1
    Stage 1 过滤阈值，默认 0.1（高召回）。
    只在提供 --stage1-pred-csv 时生效。

--output-csv tmp/clf2_preds.csv
    保存每张图的预测类别和各类别概率，供后续分析使用。
    列：gt_type, pred_type, prob_mass, prob_calc, prob_asym

------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset import DEFAULT_CSV, DEFAULT_IMAGES_ROOT, MammoDataset, build_lesion_type_df

# 动态导入 clf2-train.py（文件名含连字符）
_spec = _ilu.spec_from_file_location("clf2_train", _HERE / "clf2-train.py")
_clf2_train = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_clf2_train)
build_stage2_model = _clf2_train.build_stage2_model
evaluate_stage2    = _clf2_train.evaluate_stage2
NUM_CLASSES        = _clf2_train.NUM_CLASSES
CLASS_NAMES        = _clf2_train.CLASS_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MammoPearl clf2-test (Stage 2)")
    p.add_argument("--ckpt-path",         required=True)
    p.add_argument("--csv-path",          default=str(DEFAULT_CSV))
    p.add_argument("--images-root",       default=str(DEFAULT_IMAGES_ROOT))
    p.add_argument("--stage1-pred-csv",   default=None,
                   help="Stage 1 预测 CSV（clf-test.py --output-csv 输出）。"
                        "若提供则只评估 Stage 1 通过的图像。")
    p.add_argument("--stage1-threshold",  type=float, default=0.1)
    p.add_argument("--output-csv",        default=None)
    p.add_argument("--batch-size",        type=int, default=32)
    p.add_argument("--num-workers",       type=int, default=4)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载 checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    ckpt_args   = ckpt.get("args", {})
    input_h     = ckpt_args.get("input_h", 512)
    input_w     = ckpt_args.get("input_w", 512)
    amp         = ckpt_args.get("amp", False)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    best_ep     = ckpt.get("epoch", "?")
    best_score  = ckpt.get("best_score", 0.0)

    print(f"Checkpoint: epoch={best_ep}, best_score={best_score:.4f}")

    model = build_stage2_model(pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── 测试集 ────────────────────────────────────────────────────────────────
    test_df, _ = build_lesion_type_df(args.csv_path, split="test")

    # 可选：按 Stage 1 预测过滤
    if args.stage1_pred_csv:
        s1 = pd.read_csv(args.stage1_pred_csv)
        # 兼容 clf-test.py 输出列：patient_id, image_id, label, prob
        if "prob" not in s1.columns:
            raise ValueError("stage1-pred-csv 缺少 'prob' 列，请用 clf-test.py --output-csv 重新生成。")
        s1_pass = s1[s1["prob"] >= args.stage1_threshold][["patient_id", "image_id"]]
        before = len(test_df)
        test_df = test_df.merge(s1_pass, on=["patient_id", "image_id"])
        print(f"Stage 1 filter @{args.stage1_threshold}: {before} -> {len(test_df)} images")

    print(
        f"Test images: {len(test_df)}  "
        f"(none={int((test_df['label']==0).sum())}  "
        f"mass={int((test_df['label']==1).sum())}  "
        f"calc={int((test_df['label']==2).sum())}  "
        f"asym={int((test_df['label']==3).sum())})"
    )

    test_ds = MammoDataset(test_df, args.images_root, input_h, input_w, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 评估 ──────────────────────────────────────────────────────────────────
    results = evaluate_stage2(model, test_loader, device, amp)

    _SEP = "─" * 66
    print(f"\n{_SEP}")
    print("Stage 2 Test-set evaluation")
    print(f"  {'Class':>5}  {'N':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  "
          f"{'TP':>5}  {'FP':>6}  {'FN':>4}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  "
          f"{'─'*5}  {'─'*6}  {'─'*4}")
    for c in range(num_classes):
        m = results[c]
        print(
            f"  {CLASS_NAMES[c]:>5}  {m['n']:>5}  {m['recall']:>7.4f}  {m['prec']:>7.4f}  "
            f"{m['f1']:>7.4f}  {m['tp']:>5}  {m['fp']:>6}  {m['fn']:>4}"
        )
    print(
        f"  Macro F1={results['macro_f1']:.4f}  |  "
        f"Target (macro F1 Mass/Calc/Asym)={results['target_score']:.4f}"
    )
    print(_SEP)

    # ── 可选：保存预测 CSV ───────────────────────────────────────────────────
    if args.output_csv:
        _save_pred_csv(model, test_ds, test_df, device, amp, Path(args.output_csv))


def _save_pred_csv(
    model: torch.nn.Module,
    dataset: MammoDataset,
    img_df: pd.DataFrame,
    device: torch.device,
    amp: bool,
    output_path: Path,
) -> None:
    """保存每张图的预测类别和各类别概率到 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            tensor, label = dataset[idx]
            inp = tensor.unsqueeze(0).to(device)
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(inp)
            else:
                logits = model(inp)
            probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))
            row = img_df.iloc[idx]
            rows.append({
                "patient_id":  row["patient_id"],
                "image_id":    row["image_id"],
                "gt_type":     int(label),
                "pred_type":   pred_class,
                "prob_mass":   float(probs[0]),
                "prob_calc":   float(probs[1]),
                "prob_asym":   float(probs[2]),
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions saved: {output_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
