# bbox-train-resnet50.py 功能文档

> 训练脚本路径：`src/data/bounding-box/bbox-train-resnet50.py`
> 测试脚本路径：`src/data/bounding-box/bbox-test-resnet50.py`

## 概述

基于 torchvision `fasterrcnn_resnet50_fpn_v2` 模型，对 VinDr 乳腺 X 光数据集训练病灶边界框检测器。输入为 912×1520 处理后的 PNG 图像，输出为 `(xmin, ymin, xmax, ymax)` 格式的病灶框。

## 数据流

```
data/raw/vindr_detection_folds.csv
    ↓ 按 split="training" 筛选
    ↓ 按 (patient_id, series_id, image_id) 分组
VinDrBboxDataset (全量样本)
    ↓ 剔除 bad_data_record_resnet50.csv 中的坏样本
    ↓ split_train_val_by_patient() 按病人级别划分
train_indices / val_indices
    ↓
Subset → DataLoader → 训练/验证
```

## 功能列表

### 1. 模型构建 (`build_model`)

- 使用 `fasterrcnn_resnet50_fpn_v2` + `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` 预训练权重
- 自定义 `AnchorGenerator`，支持通过 `--anchor-sizes` 配置每个 FPN 层级的 anchor 尺寸（默认 `8,16,32,64,128`）
- 支持配置 RPN/ROI IoU 阈值：`--box-fg-iou-thresh`、`--box-bg-iou-thresh`、`--rpn-fg-iou-thresh`、`--rpn-bg-iou-thresh`
- 支持配置推理时框过滤：`--box-score-thresh`（默认 0.05）、`--box-detections-per-img`（默认 100）

### 2. 数据集划分 (`split_train_val_by_patient`)

- 按 `patient_id` 级别划分，防止同一病人出现在训练集和验证集两侧
- 正样本病人和负样本病人分别按比例抽取，保持原始正负分布
- 验证集比例由 `--val-ratio`（默认 0.15）控制
- 验证集不参与任何训练或采样干预

### 3. 梯度累积 (`--fuck-running` / `--accumulation-steps`)

- 默认模式（无 `--fuck-running`）：`batch_size=2`，每 4 步累积一次梯度，等效 `batch_size=8`
- 大算力模式（`--fuck-running`）：直接使用指定的 `batch_size`，无梯度累积
- 每次 `optimizer.step()` 前执行梯度裁剪 `clip_grad_norm_(max_norm=1.0)`
- 非有限值（NaN/Inf）的 loss 会被自动跳过，不参与反向传播

### 4. 渐冻层训练 (`--freeze-epochs`)

- 前 N 个 epoch 冻结 ResNet backbone 的 `layer1` 和 `layer2`，仅训练 FPN 和检测头
- 解冻时自动重建 optimizer 和 lr_scheduler

### 5. 学习率策略

- **Iter-level warmup**：第一个 epoch 内部使用 `LinearLR(start_factor=0.01)` 做 1000 步线性预热
- **Epoch-level scheduler**：
  - 默认：`CosineAnnealingLR`，`T_max` = 总 epoch 数 − balanced warmup epoch 数
  - 可选：`StepLR`（通过 `--lr-step-size` 启用）
- warmup 阶段跳过 epoch-level scheduler step，防止 cosine 过度拉低 LR

### 6. ROI 分类损失策略 (`--cls-loss-type`)

通过 monkey-patch `torchvision.models.detection.roi_heads.fastrcnn_loss` 实现，支持三种模式：

| 模式 | 参数 | 说明 |
|------|------|------|
| `ce` | — | 标准 CrossEntropy（默认） |
| `weighted_ce` | `--cls-weight-bg`, `--cls-weight-lesion` | 加权 CE，调节正负样本 loss 贡献比 |
| `focal` | `--focal-gamma`, `--focal-alpha-bg`, `--focal-alpha-lesion` | Focal Loss，自动降低简单样本 loss |

### 7. 在线难例挖掘 OHEM (`--ohem`)

- 逐样本计算 loss，按困难度降序排列，仅保留前 K 个最难样本参与反向传播
- K = max(`--ohem-min-samples`, N × `--ohem-ratio`)
- **与 Focal Loss 互斥**：同时启用时自动禁用 OHEM 并打印警告

### 8. Balanced Sampling Warmup (`--warmup-balanced-epochs`)

- 前 N 个 epoch 使用 `WeightedRandomSampler` 对训练集进行加权采样
- 正样本权重 = `--warmup-pos-weight-ratio`（默认 10.0），负样本权重 = 1.0
- RPN 在 warmup 阶段仍能学到背景抑制（正负样本均可见）
- warmup 结束后自动重建 optimizer 和 lr_scheduler，切换回全量训练

### 9. Legacy Positive-Only Warmup (`--warmup-positive-epochs`)

- 前 N 个 epoch 仅使用正样本图像训练（**不推荐**，会破坏 RPN 背景抑制能力）
- 当同时配置了 `--warmup-balanced-epochs` 时，balanced warmup 优先

### 10. Epoch 子采样 (`--only-use`)

- 每个 epoch 仅使用训练集总量的指定比例（0.0-1.0）
- **仅在正式训练阶段（非 warmup 阶段）生效**；balanced warmup 阶段始终使用全量训练集 + 加权采样
- 正负样本按原始比例等比缩减，正样本有最低保护（不低于原始正样本比例对应数量）
- 跨 epoch **轮转采样**：通过 cycle 机制确保所有图像在多个 epoch 中均被训练到
- 典型用途：将每 epoch 时间从 17min 缩短到 ~5min（`--only-use 0.3`）

### 11. 训练子 Loss 追踪

- 每个 epoch 结束后输出四项子 loss 的均值：`loss_classifier`、`loss_box_reg`、`loss_objectness`、`loss_rpn_box_reg`
- 用于诊断模型各组件（ROI 分类、框回归、RPN 前景判断、RPN 框回归）的学习状态

### 12. 验证策略

- 每个 epoch 结束后在验证集上评估（不参与反向传播）
- 主指标：F1 score（`--val-score-threshold` 和 `--val-iou-threshold` 控制匹配逻辑）
- **多阈值报告**：同时输出 threshold=0.1/0.3/0.5/0.7/0.9 的 TP/FP 统计
- **RPN proposal 监控**：记录模型原始输出框数量（threshold 前），超过 200/图时打印警告
- **Early Stopping**：连续 `--patience` 个 epoch F1 无提升则终止训练

### 13. Checkpoint 保存

- 仅保存 F1 最佳的一轮模型（非最后一轮）
- Meta 信息包含：anchor_sizes、所有超参数、split_summary、训练历史、box_score_thresh、box_detections_per_img
- 路径：`models/bbox_resnet50.pth`

### 14. ROI/RPN 超参数调优

- `--roi-batch-size-per-image`（默认 512）：每张图 ROI 采样总数
- `--roi-positive-fraction`（默认 0.25）：ROI 正样本比例，使用上取整策略防止正样本丢失
- `--rpn-pre-nms-top-n-train`（默认 2000）/ `--rpn-post-nms-top-n-train`（默认 1000）

## 完整参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv-path` | Path | auto | CSV 路径 |
| `--images-root` | Path | auto | 图像根目录 |
| `--save-path` | Path | `models/bbox_resnet50.pth` | 模型保存路径 |
| `--epochs` | int | 12 | 总训练轮数 |
| `--batch-size` | int | 2 | 批次大小 |
| `--val-batch-size` | int | 1 | 验证批次大小 |
| `--val-ratio` | float | 0.15 | 验证集比例 |
| `--val-score-threshold` | float | 0.5 | 验证时 score 阈值 |
| `--val-iou-threshold` | float | 0.5 | 验证时 IoU 阈值 |
| `--patience` | int | 5 | Early Stopping 耐心值 |
| `--min-delta` | float | 0.0 | F1 最小提升量 |
| `--lr` | float | 0.005 | 基础学习率 |
| `--momentum` | float | 0.9 | SGD 动量 |
| `--weight-decay` | float | 0.0005 | 权重衰减 |
| `--num-workers` | int | 0 | DataLoader 工作线程 |
| `--seed` | int | 42 | 随机种子 |
| `--positive-only` | flag | off | 仅使用正样本训练 |
| `--fuck-running` | flag | off | 大算力模式 |
| `--accumulation-steps` | int | 4 | 梯度累积步数 |
| `--freeze-epochs` | int | 2 | 冻结 backbone 的 epoch 数 |
| `--anchor-sizes` | str | `8,16,32,64,128` | Anchor 尺寸 |
| `--roi-batch-size-per-image` | int | 512 | ROI 采样总数/图 |
| `--roi-positive-fraction` | float | 0.25 | ROI 正样本比例 |
| `--rpn-pre-nms-top-n-train` | int | 2000 | RPN NMS 前保留数 |
| `--rpn-post-nms-top-n-train` | int | 1000 | RPN NMS 后保留数 |
| `--lr-gamma` | float | 0.1 | StepLR gamma |
| `--lr-step-size` | int | 0 | StepLR 步长（0=用 Cosine） |
| `--box-fg-iou-thresh` | float | 0.5 | ROI 前景 IoU 阈值 |
| `--box-bg-iou-thresh` | float | 0.5 | ROI 背景 IoU 阈值 |
| `--rpn-fg-iou-thresh` | float | 0.7 | RPN 前景 IoU 阈值 |
| `--rpn-bg-iou-thresh` | float | 0.3 | RPN 背景 IoU 阈值 |
| `--cls-loss-type` | str | `ce` | 损失类型：ce/weighted_ce/focal |
| `--cls-weight-bg` | float | 0.1 | weighted_ce 背景权重 |
| `--cls-weight-lesion` | float | 1.0 | weighted_ce 病灶权重 |
| `--focal-gamma` | float | 2.0 | Focal Loss gamma |
| `--focal-alpha-bg` | float | 0.25 | Focal Loss 背景 alpha |
| `--focal-alpha-lesion` | float | 0.75 | Focal Loss 病灶 alpha |
| `--ohem` | flag | off | 启用 OHEM |
| `--ohem-ratio` | float | 0.2 | OHEM 保留比例 |
| `--ohem-min-samples` | int | 128 | OHEM 最小样本数 |
| `--warmup-positive-epochs` | int | 0 | 正样本 warmup 轮数（不推荐） |
| `--warmup-balanced-epochs` | int | 0 | 平衡采样 warmup 轮数 |
| `--warmup-pos-weight-ratio` | float | 10.0 | warmup 正样本采样权重比 |
| `--only-use` | float | 1.0 | 每 epoch 使用数据比例 |
| `--box-score-thresh` | float | 0.05 | 推理时 score 过滤阈值 |
| `--box-detections-per-img` | int | 100 | 每图最大检测数 |
