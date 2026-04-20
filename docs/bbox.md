# bbox-train-resnet50.py 训练策略改动文档

本文档描述了针对 `src/data/bounding-box/bbox-train-resnet50.py` 新增的训练策略改动，
用于解决乳腺病灶检测中"误报过多、正负样本极度不平衡、背景样本稀释损失"的问题。

所有改动均以可配置命令行参数的形式实现，默认参数值保持与原有行为一致（即不传新参数时训练流程不变）。

---

## 一、可切换的 ROI 分类损失策略

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cls-loss-type` | str | `ce` | 可选 `ce` / `weighted_ce` / `focal` |
| `--cls-weight-bg` | float | `0.1` | Weighted CE 模式下背景类权重 |
| `--cls-weight-lesion` | float | `1.0` | Weighted CE 模式下病灶类权重 |
| `--focal-gamma` | float | `2.0` | Focal Loss 的 gamma 参数 |
| `--focal-alpha-bg` | float | `0.25` | Focal Loss 背景类的 alpha |
| `--focal-alpha-lesion` | float | `0.75` | Focal Loss 病灶类的 alpha |

### 说明

- **标准 CE (`ce`)**：与原始 torchvision Faster R-CNN 行为一致。
- **Weighted Cross-Entropy (`weighted_ce`)**：为背景和病灶类别分别设定不同权重。
  背景权重低（如 0.1）可削弱大量简单背景样本的 loss 稀释效应；
  病灶权重高（如 1.0+）可加重漏检病灶的惩罚。
  如果需要抑制假阳性，也可将背景权重设为大于 1.0（如 2.0），病灶权重设为 1.0。
- **Focal Loss (`focal`)**：通过 `(1-pt)^gamma` 因子自动降低高置信度（简单）样本的损失贡献。
  gamma 越大，对简单样本的抑制越强。alpha 参数提供额外的类别平衡能力。

### 实现方式

通过 monkey-patch 替换 `torchvision.models.detection.roi_heads.fastrcnn_loss`，
在 ROI Head 内部使用自定义的 `_custom_fastrcnn_loss` 函数。该函数：
1. 逐样本计算分类 loss（`reduction="none"`）
2. 根据 `cls_loss_type` 选择对应的计算逻辑
3. 保持 box regression loss 计算与原始实现完全一致
4. 不影响 RPN 的 objectness / rpn_box_reg 损失

### 示例

```bash
# Weighted CE: 降低背景权重
--cls-loss-type weighted_ce --cls-weight-bg 0.1 --cls-weight-lesion 1.0

# Weighted CE: 提高背景权重抑制 FP
--cls-loss-type weighted_ce --cls-weight-bg 2.0 --cls-weight-lesion 1.0

# Focal Loss
--cls-loss-type focal --focal-gamma 2.0 --focal-alpha-bg 0.25 --focal-alpha-lesion 0.75
```

---

## 二、在线难例挖掘（OHEM）

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ohem` | flag | 关闭 | 启用 OHEM |
| `--ohem-ratio` | float | `0.2` | 保留最困难样本的比例（前 20%） |
| `--ohem-min-samples` | int | `128` | 至少保留的样本数 |

### 说明

启用后，在每次 ROI Head 计算损失时：
1. 先计算所有候选框的独立分类 loss 和回归 loss
2. 以分类 loss + 回归 loss 作为困难度指标，降序排列
3. 取 `max(ohem_min_samples, int(N * ohem_ratio))` 个最困难样本
4. 只有这些样本参与最终的损失均值计算和反向传播
5. 简单背景样本（loss 极低）被自动排除

OHEM 可与任意损失类型组合使用（CE / Weighted CE / Focal Loss）。

### 示例

```bash
# OHEM + 标准 CE
--ohem --ohem-ratio 0.2 --ohem-min-samples 128

# OHEM + Focal Loss
--cls-loss-type focal --focal-gamma 2.0 --ohem
```

---

## 三、训练阶段 IoU 阈值配置

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--box-fg-iou-thresh` | float | `0.5` | ROI Head 正样本 IoU 阈值 |
| `--box-bg-iou-thresh` | float | `0.5` | ROI Head 背景 IoU 阈值 |
| `--rpn-fg-iou-thresh` | float | `0.7` | RPN 正样本 IoU 阈值 |
| `--rpn-bg-iou-thresh` | float | `0.3` | RPN 背景 IoU 阈值 |

### 说明

- `box-fg-iou-thresh` 是影响最大的参数：将其从 0.5 提高到 0.6 或 0.7，
  可以让模型只把定位足够准确的候选框视为正样本，减少边界模糊导致的误检。
- RPN 阈值通常不需要修改（默认 0.7/0.3 已经比较合理），但也已暴露为可配置参数。
- 这些阈值仅影响训练阶段的正负样本分配，不影响验证/推理阶段的评估 IoU 阈值
  （评估 IoU 阈值由 `--val-iou-threshold` 控制）。

### 示例

```bash
# 提高 ROI 正样本 IoU 要求
--box-fg-iou-thresh 0.6

# 更严格
--box-fg-iou-thresh 0.7
```

---

## 四、Positive-Only Warmup（两阶段训练）

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warmup-positive-epochs` | int | `0` | 正样本热身 epoch 数（0=禁用） |

### 说明

- **阶段 1（前 N 个 epoch）**：仅使用包含病灶的正样本图像训练，让模型先学会"什么是病灶"。
- **阶段 2（第 N+1 个 epoch 起）**：恢复全量样本训练（正负样本混合），继续训练剩余 epoch。
- 验证集始终使用真实分布，不受 warmup 影响。
- 如果同时指定了 `--positive-only`，则 warmup 无效（因为整个训练本身就是 positive-only）。
- 总 epoch 数仍为 `--epochs`，其中前 N 个为 warmup。

### 示例

```bash
# 前 10 个 epoch 仅用正样本训练，共训练 30 个 epoch
--epochs 30 --warmup-positive-epochs 10
```

---

## 五、ROI Positive Fraction 调整

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--roi-positive-fraction` | float | `0.25` | ROI 采样中正样本比例 |

### 说明

已有参数。将其降低到 0.05~0.1 可以让 ROI 采样更偏向背景难例，
进一步抑制假阳性。配合上取整策略确保正样本不会因取整而丢失。

### 示例

```bash
--roi-positive-fraction 0.1
--roi-positive-fraction 0.05
```

---

## 六、修改的函数/类说明

| 新增项 | 类型 | 说明 |
|--------|------|------|
| `FocalLoss` | `torch.nn.Module` | 独立的 Focal Loss 模块 |
| `_custom_fastrcnn_loss()` | 函数 | 替换 torchvision 内部 `fastrcnn_loss` 的自定义实现 |
| `apply_custom_roi_loss()` | 函数 | Monkey-patch 入口，将自定义 loss 注入模型 |

| 修改项 | 说明 |
|--------|------|
| `build_model()` | 新增 4 个 IoU 阈值参数，传递给 Faster R-CNN 工厂函数 |
| `parse_args()` | 新增 15 个命令行参数 |
| `main()` | 新增 custom loss 应用、warmup 子集准备、epoch 循环中的 warmup 切换 |

未修改的函数/逻辑：数据读取、patient-level split、checkpoint 保存、早停、
验证流程、freeze/unfreeze backbone、gradient accumulation、LR scheduling。

---

## 七、推荐实验配置

### 实验 A：Focal Loss + OHEM

```bash
python src/data/bounding-box/bbox-train-resnet50.py \
    --epochs 30 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 \
    --cls-loss-type focal --focal-gamma 2.0 --focal-alpha-bg 0.25 --focal-alpha-lesion 0.75 \
    --ohem --ohem-ratio 0.2 --ohem-min-samples 128
```

### 实验 B：Weighted CE + IoU 上调

```bash
python src/data/bounding-box/bbox-train-resnet50.py \
    --epochs 30 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 \
    --cls-loss-type weighted_ce --cls-weight-bg 0.1 --cls-weight-lesion 1.0 \
    --box-fg-iou-thresh 0.6
```

### 实验 C：Focal Loss + Warmup + 低 positive fraction

```bash
python src/data/bounding-box/bbox-train-resnet50.py \
    --epochs 30 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 \
    --cls-loss-type focal --focal-gamma 2.0 \
    --warmup-positive-epochs 10 \
    --roi-positive-fraction 0.1 --roi-batch-size-per-image 256
```

### 实验 D：全策略组合

```bash
python src/data/bounding-box/bbox-train-resnet50.py \
    --epochs 30 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 \
    --cls-loss-type focal --focal-gamma 2.0 --focal-alpha-bg 0.25 --focal-alpha-lesion 0.75 \
    --ohem --ohem-ratio 0.2 \
    --box-fg-iou-thresh 0.6 \
    --warmup-positive-epochs 10 \
    --roi-positive-fraction 0.1
```
