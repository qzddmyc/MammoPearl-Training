# bbox-train-resnet50.py / bbox-test-resnet50.py 功能文档

> 训练脚本：`src/data/bounding-box/bbox-train-resnet50.py`  
> 测试脚本：`src/data/bounding-box/bbox-test-resnet50.py`

## 1. 设计目标

当前 bbox 工作流的目标是：**对乳腺 X 光图像中的可疑病灶区域进行初步框选**，而不是做更复杂的病灶细分类、多阶段检测或分割引导检测。

因此当前实现遵循以下原则：

1. **保持单类检测**：所有有效病灶框统一视为 `lesion`
2. **优先减少背景干扰**：默认先定位乳房主体区域，再进行裁剪和检测
3. **保证 train / val / test 口径尽量一致**：checkpoint 会记录训练侧的关键推理配置，测试脚本优先读取这些配置
4. **优先服务粗框选**：强调“先把病灶大致框出来”，不主动把任务升级成更复杂的系统

---

## 2. 当前工作流

### 2.1 训练流程

```text
data/raw/vindr_detection_folds.csv
    ↓ 按 split="training" 过滤
    ↓ 按 (patient_id, series_id, image_id) 分组
VinDrBboxDataset
    ↓ 读取 data/processed/images_png/<patient_id>/<image_id>
    ↓ 裁剪非法 / 越界 bbox
    ↓ 检测乳房主体区域（默认开启）
    ↓ 对图像裁剪，并将 bbox 同步重映射到裁剪后的局部坐标
    ↓ 读取 bad_data_record_resnet50.csv（若存在）
    ↓ patient-level 划分 train / val
    ↓ 可选 balanced warmup / legacy positive-only warmup / only-use 子采样
    ↓ Faster R-CNN 训练
    ↓ 每个 epoch 做验证、记录 F1、按最佳 F1 保存 checkpoint
```

### 2.2 测试流程

```text
models/bbox_resnet50.pth
    ↓ 读取 checkpoint meta
    ↓ 恢复训练时的 anchor / box filtering / breast crop / val threshold 配置
test split 图像
    ↓ 检测乳房主体区域（默认跟随 checkpoint）
    ↓ 裁剪并重映射 GT bbox
    ↓ 模型推理
    ↓ 在裁剪坐标系内做匹配
    ↓ 输出 precision / recall / F1 / image_accuracy / mean IoU / 坐标误差
    ↓ 若保存预测结果，则回写为原始 processed 图像坐标
```

---

## 3. 训练脚本完整功能

### 3.1 数据读取与清洗

- 训练数据来自 `data/raw/vindr_detection_folds.csv` 中的 `split="training"`
- 以 `(patient_id, series_id, image_id)` 为单位聚合，一张图像保留其全部 bbox
- 训练前会先对 bbox 做越界裁剪与非法框过滤
- 若存在 `bad_data_record_resnet50.csv`，会在 train / val 划分前剔除对应样本
- 支持 `--positive-only`，只在训练侧保留有病灶框的图像；验证集仍保持真实分布

### 3.2 乳房主体裁剪

- 默认开启
- 使用 Otsu 阈值 + 开运算 + 最大轮廓，估计乳房主体区域
- 对图像做带 margin 的裁剪，并将 bbox 同步映射到裁剪后的局部坐标系
- 可通过 `--disable-breast-crop` 关闭
- 可通过 `--breast-crop-margin` 控制裁剪留白

### 3.3 训练 / 验证划分

- 训练脚本会从 training split 中再切出验证集
- 划分优先按 `patient_id` 进行，避免同一病人同时出现在 train 和 val
- 划分后尽量保持正负图像分布稳定
- 若 patient-level 划分结果异常为空，脚本会打印告警并做兜底处理

### 3.4 模型构建

- 使用 `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`
- 优先加载 `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`
- 支持自定义 anchor sizes
- 支持配置 ROI / RPN 的 IoU 匹配阈值
- 支持配置模型内部的 `box_score_thresh` 与 `box_detections_per_img`
- 最终类别固定为 2 类：`background` + `lesion`

### 3.5 优化器、学习率与 batch 策略

- 使用 SGD + Momentum
- 支持 `weight_decay`
- 默认可在小 batch 下使用梯度累积
- `--fuck-running` 打开后，按真实 batch_size 直接训练，不做梯度累积
- 首个 epoch 支持 iter-level LinearLR warmup
- epoch-level scheduler 支持：
  - `StepLR`
  - `CosineAnnealingLR`
- balanced warmup 结束后，支持通过 `--post-warmup-lr` 以更低 LR 重建 optimizer
- 支持冻结 / 解冻 backbone 的 `layer1`、`layer2`

### 3.6 样本采样与 warmup

- 支持 legacy `--warmup-positive-epochs`（仅正样本 warmup）
- 支持 `--warmup-balanced-epochs` + `--warmup-pos-weight-ratio`
  - 使用 `WeightedRandomSampler` 对正样本图像过采样
  - warmup 结束后切回正常训练
- 支持 `--only-use`
  - 每轮只使用一部分训练数据
  - 对正样本数量做下限保护
  - 通过跨 epoch 轮转采样，避免长期只看到同一批图像

### 3.7 损失函数与难例策略

- `--cls-loss-type` 支持：
  - `ce`
  - `weighted_ce`
  - `focal`
- `weighted_ce` 支持背景 / 病灶类别权重
- `focal` 支持 `gamma` 与 `alpha`
- 支持 `--ohem`
- 禁止 `focal + ohem` 同时启用；若同时传入，会自动关闭 OHEM 并告警
- 支持通过 `--roi-batch-size-per-image` 与 `--roi-positive-fraction` 调整 ROI 采样
- 支持通过 `--rpn-pre-nms-top-n-train` / `--rpn-post-nms-top-n-train` 调整 RPN proposal 数量

### 3.8 稳定性保护与日志

- 若 `freeze_epochs` 落在 balanced warmup 中间，会主动打印冲突告警
- 在 `weighted_ce` 下，如果训练集负样本图像显著多于正样本图像，且 `--cls-weight-bg < 1.0`：
  - 默认会钳制到 `1.0`
  - 只有传入 `--allow-low-bg-weight` 才保留较低背景权重
- 启动时打印 train / val 的 neg/pos image ratio
- 当 `--only-use` 导致每轮正样本图像过少时，会打印预警
- 每个 epoch 起始打印分割线，便于长日志定位
- 对 scheduler 的日志分为四类：
  - warmup 阶段跳过
  - 本轮刚重建 optimizer / scheduler
  - 正常执行 `lr_scheduler.step()`
  - 本轮没有执行任何 `optimizer.step()` 的异常情况

### 3.9 验证、早停与 best checkpoint

- 主指标为验证集 F1
- 每个 epoch 会输出：
  - train loss
  - train sub-loss（classifier / box_reg / objectness / rpn_box_reg）
  - val precision / recall / F1
  - val TP / FP / FN
  - 当前 best F1 与 best epoch
- `validate_one_epoch` 还会：
  - 统计 `threshold=0.1/0.3/0.5/0.7/0.9` 下的 TP / FP
  - 监控平均 raw predictions 数量
  - 当 `avg_raw_preds_per_image > 200` 时打印告警
- 仅保存 **best checkpoint**
- 支持基于 `patience + min_delta` 的 early stopping

### 3.10 checkpoint meta 中记录的信息

训练脚本保存 checkpoint 时，会记录包括但不限于以下信息：

- 数据路径
- `history`
- `anchor_sizes`
- `roi_batch_size_per_image`
- `roi_positive_fraction`
- `val_score_threshold`
- `val_iou_threshold`
- `cls_loss_type`
- `ohem_enabled`
- `effective_cls_weight_bg`
- `effective_cls_weight_lesion`
- `low_bg_weight_guard_applied`
- `crop_breast_region`
- `breast_crop_margin`
- `box_score_thresh`
- `box_detections_per_img`
- `train_neg_to_pos_ratio`
- `val_neg_to_pos_ratio`
- `split_summary`

---

## 4. 测试脚本完整功能

### 4.1 自动恢复训练侧评估配置

测试脚本会优先从 checkpoint meta 恢复：

1. `anchor_sizes`
2. `box_score_thresh`
3. `box_detections_per_img`
4. `crop_breast_region`
5. `breast_crop_margin`
6. `val_score_threshold`
7. `val_iou_threshold`

这意味着：**只要不手动 override，test 会尽量复现 train 侧验证时的推理与匹配口径。**

### 4.2 测试阶段的数据处理

- 从 `split="test"` 读取数据
- 对 bbox 做越界裁剪与非法框过滤
- 若启用乳房主体裁剪，则对 GT bbox 同步重映射
- 推理和匹配发生在裁剪后的局部坐标系中

### 4.3 输出指标

测试脚本会输出：

- bbox precision
- bbox recall
- bbox F1
- image-level presence accuracy
- mean IoU
- mean absolute coordinate error

### 4.4 保存预测结果

若传入 `--save-predictions`，则输出 CSV，包含：

- `patient_id`
- `image_id`
- 裁剪框坐标
- 预测框坐标
- GT 框坐标
- IoU
- 预测分数

其中预测框和 GT 框会被写回 **原始 processed 图像坐标系**，方便人工排查。

### 4.5 手动覆盖行为

测试脚本支持以下覆盖：

- `--anchor-sizes`
- `--box-score-thresh`
- `--box-detections-per-img`
- `--score-threshold`
- `--iou-threshold`
- `--force-breast-crop`
- `--disable-breast-crop`
- `--breast-crop-margin`

一旦手动覆盖，就不再完全等价于训练侧默认验证口径。

---

## 5. train / test 一致性审查结论

当前两份脚本在以下关键逻辑上是一致的：

1. **模型骨架一致**：都使用 `fasterrcnn_resnet50_fpn_v2`
2. **anchor 配置一致**：test 会从 checkpoint meta 自动恢复
3. **内部 box filtering 一致**：`box_score_thresh` / `box_detections_per_img` 会自动恢复
4. **乳房裁剪一致**：`crop_breast_region` / `breast_crop_margin` 会自动恢复
5. **评估阈值一致**：test 现在会优先读取训练侧的 `val_score_threshold` / `val_iou_threshold`
6. **坐标口径一致**：匹配在裁剪后坐标系内完成，但保存结果时回写原图坐标

因此，就“**能否较好反映 train 文件训练出的 detector 效果**”而言，当前 test 文件是**基本可以的**。

但仍需明确：

- test 反映的是 **best checkpoint 在 test split 上的检测效果**
- 它不会复现训练中的 warmup、loss 计算、冻结/解冻、OHEM 等训练行为
- 如果你手动覆盖 test 参数，评估结果就不再与 train 的验证口径完全一致

---

## 6. 关键参数索引

#### 6.1 训练脚本：数据与验证相关

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--csv-path` | `None` | 自定义标注 CSV |
| `--images-root` | `None` | 自定义 processed 图像根目录 |
| `--save-path` | `None` | best checkpoint 输出路径 |
| `--val-ratio` | `0.15` | training 内部划分到验证集的比例 |
| `--val-batch-size` | `1` | 验证 batch size |
| `--val-score-threshold` | `0.5` | 验证匹配时的分数阈值 |
| `--val-iou-threshold` | `0.5` | 验证匹配时的 IoU 阈值 |
| `--seed` | `42` | 随机种子 |
| `--num-workers` | `0` | DataLoader worker 数 |
| `--positive-only` | off | 训练侧只保留正样本图像 |

#### 6.2 训练脚本：优化与调度相关

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--epochs` | `12` | 总训练轮数 |
| `--batch-size` | `2` | 训练 batch size |
| `--fuck-running` | off | 打开后直接按 batch-size 训练，不做梯度累积 |
| `--accumulation-steps` | `4` | 未开启 `--fuck-running` 时的累积步数 |
| `--lr` | `0.005` | 基础学习率 |
| `--post-warmup-lr` | `None` | balanced warmup 结束后重建 optimizer 时的新 LR |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `0.0005` | 权重衰减 |
| `--freeze-epochs` | `2` | 冻结 backbone layer1/2 的 epoch 数 |
| `--lr-step-size` | `0` | `>0` 时使用 StepLR，否则使用 CosineAnnealingLR |
| `--lr-gamma` | `0.1` | StepLR 的 gamma |
| `--patience` | `5` | early stopping patience |
| `--min-delta` | `0.0` | 视为“提升”的最小 F1 增量 |

#### 6.3 训练脚本：检测头、损失与采样相关

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--anchor-sizes` | `8,16,32,64,128` | FPN anchor sizes |
| `--roi-batch-size-per-image` | `512` | ROI 每图采样数 |
| `--roi-positive-fraction` | `0.25` | ROI 正样本比例 |
| `--rpn-pre-nms-top-n-train` | `2000` | RPN 训练前 NMS proposal 数 |
| `--rpn-post-nms-top-n-train` | `1000` | RPN 训练后 NMS proposal 数 |
| `--box-fg-iou-thresh` | `0.5` | ROI 前景阈值 |
| `--box-bg-iou-thresh` | `0.5` | ROI 背景阈值 |
| `--rpn-fg-iou-thresh` | `0.7` | RPN 前景阈值 |
| `--rpn-bg-iou-thresh` | `0.3` | RPN 背景阈值 |
| `--cls-loss-type` | `ce` | `ce / weighted_ce / focal` |
| `--cls-weight-bg` | `1.0` | `weighted_ce` 的背景类权重 |
| `--cls-weight-lesion` | `1.0` | `weighted_ce` 的病灶类权重 |
| `--allow-low-bg-weight` | off | 允许在严重失衡时保留 `<1.0` 的背景权重 |
| `--focal-gamma` | `2.0` | Focal Loss gamma |
| `--focal-alpha-bg` | `0.25` | Focal Loss 背景 alpha |
| `--focal-alpha-lesion` | `0.75` | Focal Loss 病灶 alpha |
| `--ohem` | off | 启用 OHEM |
| `--ohem-ratio` | `0.2` | OHEM 保留样本比例 |
| `--ohem-min-samples` | `128` | OHEM 最少保留样本数 |
| `--warmup-positive-epochs` | `0` | legacy positive-only warmup |
| `--warmup-balanced-epochs` | `0` | balanced warmup epoch 数 |
| `--warmup-pos-weight-ratio` | `10.0` | balanced warmup 正样本权重倍率 |
| `--only-use` | `1.0` | 每个 epoch 使用训练集的比例 |
| `--disable-breast-crop` | off | 关闭乳房主体裁剪 |
| `--breast-crop-margin` | `0.05` | 乳房裁剪边缘留白比例 |
| `--box-score-thresh` | `0.05` | 模型内部低分框过滤阈值 |
| `--box-detections-per-img` | `100` | 每图最大检测框数量 |

#### 6.4 测试脚本重点参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--ckpt-path` | `None` | checkpoint 路径 |
| `--score-threshold` | `None` | 匹配用分数阈值；默认优先读取 checkpoint 的 `val_score_threshold` |
| `--iou-threshold` | `None` | 匹配用 IoU 阈值；默认优先读取 checkpoint 的 `val_iou_threshold` |
| `--save-predictions` | `None` | 保存匹配结果 CSV |
| `--anchor-sizes` | `None` | 覆盖 checkpoint 中的 anchor 配置 |
| `--box-score-thresh` | `None` | 覆盖 checkpoint 中的模型内部 box 过滤阈值 |
| `--box-detections-per-img` | `None` | 覆盖 checkpoint 中的最大检测数 |
| `--force-breast-crop` | off | 强制开启乳房裁剪 |
| `--disable-breast-crop` | off | 强制关闭乳房裁剪 |
| `--breast-crop-margin` | `None` | 覆盖 checkpoint 中的裁剪留白 |

---

## 7. 推荐命令

#### 7.1 推荐训练命令

下面这条命令适合当前“病灶初步框选”目标，并按 **Git Bash** 写法给出：

```bash
python src/data/bounding-box/bbox-train-resnet50.py \
    --epochs 50 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --lr 0.005 \
    --post-warmup-lr 0.001 \
    --freeze-epochs 0 \
    --roi-batch-size-per-image 512 \
    --roi-positive-fraction 0.25 \
    --anchor-sizes 16,32,64,128,256 \
    --cls-loss-type weighted_ce \
    --cls-weight-bg 1.0 \
    --cls-weight-lesion 1.0 \
    --box-fg-iou-thresh 0.5 \
    --warmup-balanced-epochs 8 \
    --warmup-pos-weight-ratio 10.0 \
    --patience 15
```

说明：

- 默认会启用乳房主体裁剪
- 这条命令不再推荐 `--only-use 0.3`
- 这条命令不再推荐把背景权重压到 `0.3`
- `--anchor-sizes 16,32,64,128,256` 是当前“粗框选”偏向下的推荐值，不等同于脚本内置默认值

#### 7.2 推荐测试命令

若希望 test 尽量跟随 train 的 checkpoint 配置，可直接：

```bash
python src/data/bounding-box/bbox-test-resnet50.py \
    --ckpt-path models/bbox_resnet50.pth
```

若你想手动覆盖匹配阈值并保存结果：

```bash
python src/data/bounding-box/bbox-test-resnet50.py \
    --ckpt-path models/bbox_resnet50.pth \
    --score-threshold 0.5 \
    --iou-threshold 0.5 \
    --save-predictions tmp/bbox_test_matches.csv
```

---

## 8. 什么叫“良好的训练效果”

在当前“病灶区域初步框选”目标下，**良好的训练效果**不是单看 train loss 降低，而应同时满足下面几类现象：

1. **验证 / 测试 F1 有实质提升**：不是只比旧结果略高一点，而是 precision 和 recall 都不再长期塌陷
2. **FP 得到控制**：每张图的误检框数量明显下降，`val_fp` 与 `avg_raw_preds_per_image` 不再异常膨胀
3. **病灶大致能框住**：mean IoU 与坐标误差处于可接受范围，预测框大体落在真实病灶区域附近
4. **best checkpoint 与 test 表现一致性较好**：验证集表现不是偶然峰值，test 上也能大致延续
5. **符合“初步框选”目标**：能把大部分可疑区域先框出来，而不是追求极细粒度边界或病灶分类解释

换句话说，对你当前任务来说，“良好”更接近：

- **较少漏掉明显病灶**
- **不要产生过量假阳性**
- **框的位置大致正确**
- **验证集和测试集的行为一致**

---

## 9. 适用边界

当前实现适用于：

- 病灶区域的单类初步框选
- train / val / test 口径一致的 bbox 检测实验
- 更关注“先框出来”，而不是更细粒度病灶类别解释

当前实现**不主动追求**：

- 多类别病灶检测
- 分割引导的多阶段检测
- 复杂病灶类型专门建模

如果后续目标从“初步框选”升级为“更精细的病灶识别与分类”，再考虑继续扩展工作流会更合适。
