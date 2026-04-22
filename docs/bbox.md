# bbox-train.py / bbox-test.py 功能文档

> 训练脚本：`src/data/bounding-box/bbox-train.py`  
> 测试脚本：`src/data/bounding-box/bbox-test.py`

## 1. 设计目标

当前 bbox 工作流的目标是：

- 对 VinDr processed PNG 图像中的病灶区域做**单类粗框选**
- 使用 Faster R-CNN 输出 `(xmin, ymin, xmax, ymax)` 检测框
- 让 train / val / test 的输入口径、裁剪口径、匹配口径尽量一致

当前实现的边界：

- 类别固定为 `background + lesion`
- 不做多类别病灶检测
- 不做分割引导检测
- 不做级联检测

---

## 2. 坐标体系与裁剪逻辑

## 2.1 坐标体系

当前流程中有两套坐标：

1. **processed 原图坐标**
   - 对应 `data/processed/images_png/<patient_id>/<image_id>`
   - CSV 中的 bbox 按这套坐标读取
   - test 导出 CSV 时最终会写回这套坐标

2. **裁剪后的局部坐标**
   - 当启用乳房主体裁剪时，图像先裁成 `crop_box=(x1, y1, x2, y2)`
   - bbox 会同步减去 `(x1, y1)`，进入裁剪后的局部坐标系
   - train 的训练、val 的匹配、test 的匹配都在这套坐标里进行

## 2.2 图像标准化

train 与 test 都先对图像做统一标准化：

- 灰度图转 RGB
- BGRA 转 RGB
- BGR 转 RGB
- 若通道数大于 4，仅保留前 3 个通道后再转 RGB
- 最终再转成 `[0, 1]` 范围的 float tensor

## 2.3 乳房主体区域检测 `detect_breast_region()`

train 与 test 使用同一套裁剪框检测逻辑：

1. 转灰度
2. 若灰度图不是 `uint8`，归一化到 `0~255`
3. 若图像尺寸非法，返回 `(0, 0, 0, 0)`
4. 若整图全黑或最大值 `<= 0`，返回整图 `(0, 0, w, h)`
5. 对灰度图做 `Otsu` 自动阈值二值化
6. 用 `5x5` 椭圆核做一次开运算
7. 查找外轮廓
8. 若找到轮廓：
   - 取面积最大的轮廓
   - 使用其 bounding rect 作为裁剪框
9. 若没找到轮廓：
   - 退回到所有非零像素的最小外接矩形
10. 按 `margin_ratio` 给裁剪框四周扩展留白
11. 将裁剪框 clamp 到图像边界内
12. 若扩展后 `x2 <= x1` 或 `y2 <= y1`，退回整图 `(0, 0, w, h)`

## 2.4 图像与 bbox 同步裁剪 `crop_image_and_boxes()`

train 与 test 使用同一套裁剪与框重映射逻辑：

1. 按 `img[y1:y2, x1:x2]` 裁图
2. 若当前没有 bbox，返回裁剪图像和空框数组
3. 对 bbox 做局部坐标重映射：
   - `xmin/xmax -= x1`
   - `ymin/ymax -= y1`
4. 将重映射后的框 clamp 到裁剪图边界内
5. 过滤掉宽或高不超过 1 像素的框：
   - `xmax > xmin + 1`
   - `ymax > ymin + 1`

## 2.5 train 中的裁剪行为

训练数据集 `VinDrBboxDataset` 的默认行为：

- `crop_breast_region=True`
- `breast_crop_margin=0.05`

训练主流程中又进一步规定：

- 默认开启乳房主体裁剪
- 只有传入 `--disable-breast-crop` 才关闭

train 数据取样时的顺序是：

1. 读 processed 图像
2. 取该图像全部 bbox
3. 对 bbox 做整图边界 clamp
4. 过滤非法框
5. 若开启裁剪：
   - 先检测 `crop_box`
   - 再同步裁图和裁框
6. 把裁剪后的图像与裁剪后的 bbox 送进模型

## 2.6 test 中的裁剪行为

测试数据集 `VinDrBboxDataset` 的类默认值是：

- `crop_breast_region=False`
- `breast_crop_margin=0.05`

但 test 主流程不会直接使用类默认值，而是按以下优先级决定裁剪是否开启：

1. `--force-breast-crop`
2. `--disable-breast-crop`
3. checkpoint meta 中的 `crop_breast_region`
4. 若 checkpoint meta 没有该字段，则回退为 `False`

test 数据取样时的顺序是：

1. 读 processed 图像
2. 取 GT bbox
3. 对 GT bbox 做整图边界 clamp
4. 过滤非法框
5. 默认先把 `crop_box` 设成整图 `(0, 0, w, h)`
6. 若开启裁剪：
   - 先检测 `crop_box`
   - 再同步裁图和裁框
7. 返回：
   - 裁剪后的图像
   - 裁剪后的 GT
   - 样本信息
   - `crop_box`
   - `original_image_size`

## 2.7 test 导出 CSV 时的坐标回写

test 的匹配发生在裁剪后的局部坐标中。

若传入 `--save-predictions`：

- 对每个成功匹配的预测-真值对
- 会把预测框和 GT 框分别加回 `crop_x1`、`crop_y1`
- 再写成原始 processed 图像坐标

因此导出 CSV 中：

- `crop_*` 是裁剪框坐标
- `pred_*` 是原图坐标
- `gt_*` 也是原图坐标

---

## 3. 训练脚本工作流

```text
data/raw/vindr_detection_folds.csv
    ↓ 过滤 split="training"
    ↓ 按 (patient_id, series_id, image_id) 分组
VinDrBboxDataset
    ↓ 读取 processed PNG
    ↓ bbox 越界裁剪与非法框过滤
    ↓ 可选乳房主体裁剪，并将 bbox 重映射到裁剪后的局部坐标
    ↓ bad_data_record_resnet50.csv 过滤
    ↓ patient-level train / val 划分
    ↓ 可选 positive-only / balanced warmup / only-use 子采样
    ↓ Faster R-CNN 训练
    ↓ 每个 epoch 进行验证
    ↓ 按最佳 val_F1 保存 checkpoint
```

---

## 4. 测试脚本工作流

```text
models/bbox_resnet50.pth
    ↓ 读取 checkpoint meta
    ↓ 恢复 anchor / box filtering / crop / val threshold 配置
test split 图像
    ↓ 可选乳房主体裁剪，并将 GT 重映射到裁剪后的局部坐标
    ↓ 模型推理
    ↓ 按 score + IoU 做 greedy one-to-one matching
    ↓ 输出 precision / recall / F1 / image_accuracy / mean IoU / 坐标误差
    ↓ 若保存预测结果，则回写到原图坐标
```

---

## 5. 训练脚本完整功能

## 5.1 数据读取与样本组织

- 从 `vindr_detection_folds.csv` 读取标注
- 仅保留 `split="training"`
- 以 `(patient_id, series_id, image_id)` 分组
- 每张图像保留全部 bbox
- 图像路径组织为：
  - `data/processed/images_png/<patient_id>/<image_id>`

## 5.2 bbox 清洗

当前对 bbox 的处理分两层：

1. 构建样本时：
   - 若发现原始框满足 `xmax <= xmin` 或 `ymax <= ymin`
   - 打印 warning

2. 取样时：
   - 先把 bbox clamp 到当前图像边界
   - 再过滤掉宽或高不超过 1 像素的框

## 5.3 bad data 过滤

训练主流程会尝试读取：

- `src/data/bounding-box/bad_data_record_resnet50.csv`

若存在：

- 读取 `(patient_id, image_id)` 组合
- 在 train / val 划分前剔除对应样本

若不存在：

- 打印信息并使用空集合

## 5.4 patient-level train / val 划分

训练不会直接使用 CSV 中的 validation split，而是从 training 中再切出验证集。

当前逻辑：

- 按 `patient_id` 划分，避免病人泄漏
- 对“含正样本 patient”和“纯负样本 patient”分别选取验证子集
- 目标是让验证集图像量大致接近 `val_ratio`

当前兜底逻辑：

- 若 train 侧为空：
  - 回退到全部 usable 样本做训练
- 若 val 侧为空：
  - 强制把一个完整 patient 移到验证集

## 5.5 `--positive-only`

当前行为是：

- 只对训练侧保留正样本图像
- 验证集保持真实分布

实现顺序：

1. 先构建完整 training 数据集
2. 先做 patient-level train / val 划分
3. 再仅对 `train_indices` 做 positive-only 过滤

## 5.6 模型构建

训练脚本使用：

- `fasterrcnn_resnet50_fpn_v2`

当前支持：

- 优先加载 `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`
- 自定义 `anchor_sizes`
- 自定义 ROI / RPN 的 IoU 匹配阈值
- 自定义 `box_score_thresh`
- 自定义 `box_detections_per_img`
- 最终类别固定为 2 类

若 v2 构造函数签名不兼容：

- 回退到 `fasterrcnn_resnet50_fpn`
- 再不行才退回默认最简构造

## 5.7 ROI 分类损失

`--cls-loss-type` 当前支持：

- `ce`
- `weighted_ce`
- `focal`

### `ce`

- 标准交叉熵

### `weighted_ce`

- 支持分别设置 background / lesion 权重
- 若训练集负样本图像明显多于正样本图像
- 且 `--cls-weight-bg < 1.0`
- 且未传 `--allow-low-bg-weight`
- 会自动把背景权重钳制到 `1.0`

### `focal`

- 使用 focal loss
- 支持 `gamma`
- 支持背景和病灶两类的 `alpha`

## 5.8 OHEM

若启用 `--ohem`：

- 先逐样本估计困难度
- 仅保留 top-K 最困难样本参与训练
- `K = max(ohem_min_samples, int(N * ohem_ratio))`

若同时启用 `focal + ohem`：

- 启动时自动关闭 OHEM
- 并打印 warning

## 5.9 优化器与参数分组

训练当前使用：

- `SGD(momentum=args.momentum)`

参数分组规则：

- bias / BN / norm 参数不加 weight decay
- 其余参数使用 `args.weight_decay`

## 5.10 冻结与解冻 backbone

若 `--freeze-epochs > 0`：

- 初始冻结 backbone 的 `layer1` 与 `layer2`

当训练进行到 `freeze_epochs` 对应轮次后：

- 解冻 `layer1` 与 `layer2`
- 用当前 LR 重建 optimizer
- 重新创建 scheduler

若 `0 < freeze_epochs < warmup_balanced_epochs`：

- 启动时打印 warning

## 5.11 梯度累积

当前行为：

- 默认启用梯度累积
- 若不传 `--fuck-running`
  - `accumulation_steps = max(1, args.accumulation_steps)`
- 若传入 `--fuck-running`
  - `accumulation_steps = 1`

训练时：

- loss 先除以 `accumulation_steps`
- 累积够步数后再 `optimizer.step()`

## 5.12 iter-level warmup

只在第一个 epoch 创建 `LinearLR`：

- `warmup_iters = min(1000, len(train_loader) - 1)`

并且：

- 只有在实际执行 `optimizer.step()` 时，warmup scheduler 才会 step

## 5.13 epoch-level scheduler

当前支持两种：

- `StepLR`
- `CosineAnnealingLR`

选择规则：

- `--lr-step-size > 0` 时使用 `StepLR`
- 否则使用 `CosineAnnealingLR`

balanced warmup 存在时：

- cosine 的 `T_max` 会扣掉 warmup epoch 数

## 5.14 warmup 行为

### legacy positive-only warmup

- 参数：`--warmup-positive-epochs`
- 仅在 balanced warmup 未启用时参与
- 前若干个 epoch 仅用正样本图像训练

### balanced warmup

- 参数：
  - `--warmup-balanced-epochs`
  - `--warmup-pos-weight-ratio`
- 使用 `WeightedRandomSampler`
- 对正样本图像赋更高采样权重
- warmup 结束后：
  - 切回正式训练集
  - 重建 optimizer
  - 可选用 `--post-warmup-lr`
  - 重建 scheduler

若训练集中没有正样本：

- 自动关闭相关 warmup
- 并打印 warning

## 5.15 `--only-use`

当 `only_use < 1.0` 时：

- 正式训练阶段每个 epoch 只使用训练集的一部分
- 不会固定使用同一批图像
- 对正样本和负样本分别轮转抽取
- 保证正样本数量不低于原始比例对应数量

若估算出的每轮正样本图像过少：

- 打印 warning

## 5.16 train_one_epoch 的异常 batch 处理

若某个 batch 中：

- loss dict 内存在非有限值，或
- 总 loss 非有限

则：

- 清零梯度
- 跳过该 batch

训练日志会额外打印：

- `[Sum] count`
- `[Sum] bad data count`

## 5.17 验证逻辑

每个 epoch 后都会做验证。

当前验证行为：

- 在 val 集上运行模型
- 使用 `score_threshold` 与 `iou_threshold` 做 greedy one-to-one matching
- 输出：
  - precision
  - recall
  - F1
  - TP / FP / FN
  - `pred_boxes`
  - `raw_preds`
  - `avg_raw_preds_per_image`

并额外统计：

- `threshold=0.1/0.3/0.5/0.7/0.9` 下的 TP / FP

若 `avg_raw_preds_per_image > 200`：

- 打印 warning

## 5.18 scheduler 日志分支

epoch 结束时，scheduler 日志当前分四种情况：

1. balanced warmup 阶段跳过
2. 当前 epoch 刚重建 optimizer / scheduler，跳过
3. 当前 epoch 正常执行 `lr_scheduler.step()`
4. 当前 epoch 没有任何 `optimizer.step()`，打印 warning

## 5.19 best checkpoint 与 early stopping

主指标：

- `val_f1`

保存逻辑：

- 只有当 `val_f1 > best_val_f1 + min_delta`
- 才覆盖保存 best checkpoint

停止逻辑：

- 若连续 `patience` 个 epoch 没有提升
- 提前停止训练

## 5.20 训练时写入 checkpoint 的 meta 字段

当前会写入的字段包括：

- `task`
- `num_classes`
- `class_names`
- `csv_path`
- `images_root`
- `positive_only`
- `history`
- `torchvision_model`
- `anchor_sizes`
- `roi_batch_size_per_image`
- `roi_positive_fraction`
- `val_ratio`
- `val_score_threshold`
- `val_iou_threshold`
- `patience`
- `min_delta`
- `best_epoch`
- `best_val_precision`
- `best_val_recall`
- `best_val_f1`
- `cls_loss_type`
- `ohem_enabled`
- `effective_cls_weight_bg`
- `effective_cls_weight_lesion`
- `low_bg_weight_guard_applied`
- `crop_breast_region`
- `breast_crop_margin`
- `box_fg_iou_thresh`
- `warmup_positive_epochs`
- `warmup_balanced_epochs`
- `only_use`
- `train_neg_to_pos_ratio`
- `val_neg_to_pos_ratio`
- `box_score_thresh`
- `box_detections_per_img`
- `split_summary`

---

## 6. 测试脚本完整功能

## 6.1 启动时的默认路径

若命令行未传参：

- `csv_path` 默认：
  - `data/raw/vindr_detection_folds.csv`
- `images_root` 默认：
  - `data/processed/images_png`
- `ckpt_path` 默认：
  - `models/bbox_resnet50.pth`

## 6.2 启动时的检查

test 会检查：

- CSV 文件存在
- processed 图像目录存在
- checkpoint 存在

任一不存在都会直接抛错。

## 6.3 checkpoint meta 恢复逻辑

test 会优先从 checkpoint meta 恢复以下设置：

1. `anchor_sizes`
2. `box_score_thresh`
3. `box_detections_per_img`
4. `val_score_threshold`
5. `val_iou_threshold`
6. `crop_breast_region`
7. `breast_crop_margin`

当前优先级是：

1. 命令行显式传入
2. checkpoint meta
3. 代码 fallback 默认值

## 6.4 test 数据集行为

当前 test 数据集会：

- 读取 `split="test"`
- 按图像聚合 GT bbox
- 对 GT bbox 做整图边界 clamp
- 过滤非法框
- 若启用乳房裁剪：
  - 将图像与 GT 一起转换到裁剪后的局部坐标系

## 6.5 匹配逻辑

当前 test 的匹配规则是：

1. 模型输出 `boxes` 与 `scores`
2. 先按 `score_threshold` 过滤预测框
3. 计算预测框与 GT 的 IoU 矩阵
4. 按 score 从高到低排序
5. 做 greedy one-to-one matching
6. 统计 TP / FP / FN

当前不是：

- Hungarian matching
- mAP 全曲线评估
- COCO API 风格评估

## 6.6 输出指标

当前 test 会输出：

- `images`
- `gt_boxes`
- `pred_boxes`
- `tp`
- `fp`
- `fn`
- `precision`
- `recall`
- `f1`
- `image_accuracy`
- `mean_iou`
- `mean_abs_error`
  - `xmin`
  - `ymin`
  - `xmax`
  - `ymax`

其中：

- `image_accuracy` 是图像级 presence accuracy
- `mean_iou` 只统计匹配成功的框
- `mean_abs_error` 也只统计匹配成功的框

## 6.7 `--save-predictions` 导出的列

若传入 `--save-predictions`，当前会输出 CSV，包含：

- `patient_id`
- `image_id`
- `crop_xmin`
- `crop_ymin`
- `crop_xmax`
- `crop_ymax`
- `pred_xmin`
- `pred_ymin`
- `pred_xmax`
- `pred_ymax`
- `gt_xmin`
- `gt_ymin`
- `gt_xmax`
- `gt_ymax`
- `iou`
- `score`

注意：

- 当前只导出**匹配成功**的预测-真值对
- 不额外导出全部 FP 明细

## 6.8 当前未启用的调试参数

test 文件里存在被注释掉、当前未启用的参数定义：

- `--save-debug-dir`
- `--max-debug-images`
- `--nms-iou`

它们当前不是可用 CLI 参数。

---

## 7. 训练脚本参数总表

### 7.1 数据、路径、验证

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--csv-path` | `None` | 自定义标注 CSV 路径 |
| `--images-root` | `None` | 自定义 processed 图像根目录 |
| `--save-path` | `None` | best checkpoint 输出路径 |
| `--epochs` | `12` | 总训练 epoch 数 |
| `--batch-size` | `2` | 训练 batch size |
| `--val-batch-size` | `1` | 验证 batch size |
| `--val-ratio` | `0.15` | 从 training 中切出验证集的比例 |
| `--val-score-threshold` | `0.5` | 验证匹配分数阈值 |
| `--val-iou-threshold` | `0.5` | 验证匹配 IoU 阈值 |
| `--patience` | `5` | early stopping patience |
| `--min-delta` | `0.0` | 判定“提升”的最小 F1 增量 |
| `--num-workers` | `0` | DataLoader worker 数 |
| `--seed` | `42` | 随机种子 |
| `--positive-only` | off | 训练侧只保留正样本图像 |

### 7.2 优化与调度

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--lr` | `0.005` | 基础学习率 |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `0.0005` | 权重衰减 |
| `--fuck-running` | off | 打开后不做梯度累积 |
| `--accumulation-steps` | `4` | 未开启 `--fuck-running` 时的累积步数 |
| `--freeze-epochs` | `2` | 初期冻结 backbone `layer1/layer2` 的 epoch 数 |
| `--lr-gamma` | `0.1` | StepLR 的 gamma |
| `--lr-step-size` | `0` | `>0` 使用 StepLR，否则使用 CosineAnnealingLR |
| `--post-warmup-lr` | `None` | balanced warmup 结束后重建 optimizer 时的新 LR |

### 7.3 模型、proposal、ROI

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--anchor-sizes` | `8,16,32,64,128` | FPN anchor sizes |
| `--roi-batch-size-per-image` | `512` | ROI 每图采样数 |
| `--roi-positive-fraction` | `0.25` | ROI 正样本比例目标 |
| `--rpn-pre-nms-top-n-train` | `2000` | 训练时 RPN pre-NMS proposal 数 |
| `--rpn-post-nms-top-n-train` | `1000` | 训练时 RPN post-NMS proposal 数 |
| `--box-fg-iou-thresh` | `0.5` | ROI 前景 IoU 阈值 |
| `--box-bg-iou-thresh` | `0.5` | ROI 背景 IoU 阈值 |
| `--rpn-fg-iou-thresh` | `0.7` | RPN 前景 IoU 阈值 |
| `--rpn-bg-iou-thresh` | `0.3` | RPN 背景 IoU 阈值 |
| `--box-score-thresh` | `0.05` | 模型内部低分框过滤阈值 |
| `--box-detections-per-img` | `100` | 模型内部每图最大检测框数 |

### 7.4 分类损失与难例策略

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--cls-loss-type` | `ce` | `ce / weighted_ce / focal` |
| `--cls-weight-bg` | `1.0` | `weighted_ce` 的背景类权重 |
| `--cls-weight-lesion` | `1.0` | `weighted_ce` 的病灶类权重 |
| `--allow-low-bg-weight` | off | 允许在严重失衡时保留 `<1.0` 的背景权重 |
| `--focal-gamma` | `2.0` | focal loss gamma |
| `--focal-alpha-bg` | `0.25` | focal loss 背景 alpha |
| `--focal-alpha-lesion` | `0.75` | focal loss 病灶 alpha |
| `--ohem` | off | 启用 OHEM |
| `--ohem-ratio` | `0.2` | OHEM top-K 比例 |
| `--ohem-min-samples` | `128` | OHEM 最少保留样本数 |

### 7.5 warmup、子采样、裁剪

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--warmup-positive-epochs` | `0` | legacy positive-only warmup epoch 数 |
| `--warmup-balanced-epochs` | `0` | balanced warmup epoch 数 |
| `--warmup-pos-weight-ratio` | `10.0` | balanced warmup 中正样本采样权重倍率 |
| `--only-use` | `1.0` | 正式训练阶段每个 epoch 使用训练集的比例 |
| `--disable-breast-crop` | off | 关闭乳房主体裁剪 |
| `--breast-crop-margin` | `0.05` | 裁剪框相对留白比例 |
| `--hide-progress-bar` | off | 不输出 tqdm 训练/验证进度条（仅训练脚本；测试脚本无此参数） |

---

## 8. 测试脚本参数总表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--csv-path` | `None` | 自定义标注 CSV 路径 |
| `--images-root` | `None` | 自定义 processed 图像根目录 |
| `--ckpt-path` | `None` | checkpoint 路径 |
| `--score-threshold` | `None` | 匹配分数阈值；默认优先读 checkpoint `val_score_threshold` |
| `--iou-threshold` | `None` | 匹配 IoU 阈值；默认优先读 checkpoint `val_iou_threshold` |
| `--num-workers` | `0` | DataLoader worker 数 |
| `--save-predictions` | `None` | 导出 matched predictions CSV |
| `--anchor-sizes` | `None` | 覆盖 checkpoint 中的 anchor 配置 |
| `--box-score-thresh` | `None` | 覆盖 checkpoint 中的模型内部 box 过滤阈值 |
| `--box-detections-per-img` | `None` | 覆盖 checkpoint 中的模型内部最大检测数 |
| `--force-breast-crop` | off | 强制开启乳房主体裁剪 |
| `--disable-breast-crop` | off | 强制关闭乳房主体裁剪 |
| `--breast-crop-margin` | `None` | 覆盖 checkpoint 中的裁剪留白比例 |

### 8.1 test 中的裁剪优先级

当前 test 的裁剪优先级是：

1. `--force-breast-crop`
2. `--disable-breast-crop`
3. checkpoint meta 中的 `crop_breast_region`
4. fallback 默认值 `False`

若同时传入 `--force-breast-crop` 与 `--disable-breast-crop`：

- 当前代码会优先采用 `--force-breast-crop`

---

## 9. 使用示例

## 9.1 train 文件开头当前给出的训练示例

```bash
python src/data/bounding-box/bbox-train.py \
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

## 9.2 test 文件开头当前给出的测试示例

```bash
python src/data/bounding-box/bbox-test.py \
    --ckpt-path models/bbox_resnet50.pth
```

```bash
python src/data/bounding-box/bbox-test.py \
    --ckpt-path models/bbox_resnet50.pth \
    --save-predictions tmp/bbox_test_matches.csv
```

---

## 10. 一致性结论

当前 train 与 test 在以下方面是一致的：

1. 都使用相同的乳房裁剪检测函数
2. 都使用相同的 bbox 裁剪与局部坐标重映射函数
3. train 的验证与 test 的评估都在裁剪后的局部坐标内完成
4. test 若导出 CSV，会把匹配框回写到原图坐标

因此，当前实现下：

- train / val / test 的裁剪坐标口径是统一的
- checkpoint meta 能把训练侧的关键推理配置传递给测试侧
