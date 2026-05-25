# 使用 retinanet_resnet50_fpn_v2 模型对数据集进行训练（rec_34 起从 Faster R-CNN 切换至 RetinaNet）

# 使用这个模型去训练会耗费很长的时间，需要注意

# If your computer is GREAT, you can use "--fuck-running" to run in a big batch-size (8).

r"""

Use this to run in Git Bash:

python src/data/bounding-box/bbox-train.py \
    --clf-epochs 10 \
    --clf-lr 1e-3 \
    --clf-pos-weight 10.0 \
    --clf-threshold 0.5 \
    --epochs 50 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --lr 0.005 \
    --post-warmup-lr 0.001 \
    --warmup-balanced-epochs 5 \
    --warmup-pos-weight-ratio 3.0 \
    --full-train-pos-weight-ratio 0.0 \
    --freeze-epochs 0 \
    --augment \
    --aug-hflip-prob 0.5 \
    --aug-brightness-delta 0.2 \
    --aug-rotation-max-deg 8.0 \
    --focal-alpha 0.5 \
    --focal-gamma 2.0 \
    --box-fg-iou-thresh 0.4 \
    --box-bg-iou-thresh 0.3 \
    --box-nms-thresh 0.5 \
    --patience 15 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --hide-progress-bar

本文件介绍：

Train a breast lesion bounding-box detector from VinDr-Mammo detection CSV.

This script reads `data/raw/vindr_detection_folds.csv`, matches each row to
`data/processed/images_png/<patient_id>/<image_id>`, and trains a RetinaNet
(retinanet_resnet50_fpn_v2) detector to predict lesion bounding boxes
(xmin, ymin, xmax, ymax).

Model checkpoint is saved to `models/bbox_resnet50.pth`.

============================================================================================================

每次改进使用的提示词，以单横线分隔。

Prompts for improvement:

1.  针对训练后期 Loss 卡在 0.19 左右无法下降的问题，需从学习率策略、
    模型结构和优化器等方面进行系统性干预，打破局部最优解。
2.  引入学习率 Warmup 机制，防止初始训练时因梯度过大破坏预训练权重，并提供合理的初始学习率设定。
3.  增加权重衰减（Weight Decay）的配置参数，通过正则化手段有效防止模型在较小数据集上过拟合。
4.  新增命令行参数 "--fuck-running" 作为算力切换开关：
    当不含此参数时，代码需在 batch_size=2 的前提下通过累积 4 个 step 再执行 optimizer.step()
    来变相实现 batch_size=8 的梯度累积；
    当存在该参数时，直接使用配置的较大 batch_size 进行正常训练，
    --同时两种模式下都必须保持每个 batch 内合理的正负样本混合比例。--(此行需要剔除，逻辑已删)
 *  fix: 在算得正样本时，需要按照比例上取整，以防止正样本丢失。
5.  针对 912x1520 的高分辨率医疗影像数据，修改模型的 AnchorGenerator，为其添加 8 和 16 这
    样更小的 scale 尺寸，以强化微小病灶的检测能力。6. 优化训练策略，除了在 DataLoader 端保持正
    负样本比之外，还需通过调整模型内部的 ROI 采样比例等参数，变相实现 Hard Negative Mining（挖掘难例）。
7.  实现渐冻层训练策略：在训练初期主动冻结 ResNet 的 layer1 和 layer2 层，仅训练 FPN 和检测头；
    在设定的几轮 Epoch 之后，全量解冻这些底层网络进行全局微调。
8.  强制采用 torchvision 中的 fasterrcnn_resnet50_fpn_v2 版本模型，以利用其更先进的
    数据增强策略和优化过的 FPN 特征提取结构。
  * feat: 需要使用 FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 权重。 
-----------
9.  取消对每一个 epoch、batch 的样本强制比例分配，保留原始的正负样本比例。
    即，对每个 epoch 使用全量样本（除了被分为验证集的）直接进行训练，
    另外，每个 epoch 中，在样本内部通过 shuffle=True 来打乱，以防止模型的过拟合。
10. 从 training 中自行划出约 15% 作为验证集；划分优先按 patient_id 进行，避免同一 patient 同时
    出现在 train 和 val；划分后尽量保持原始正负分布；验证集完全不参与训练，不参与反向传播。不做任何
    采样干预，保持真实分布。不使用 shuffle；使用验证集指标作为保存最佳模型的标准，优先推荐 F1；
    每个 epoch 后，如果当前指标优于历史最佳，则保存 best checkpoint；若验证集指标连续 N 个 epoch 没
    有提升，则停止训练，N 由参数控制。
    [注意] 这个版本保存的模型是效果最佳的一轮模型，而不是最终一轮的训练成果。
-----------
11. 引入可切换的 ROI 分类损失策略（通过 --cls-loss-type 切换）：支持标准 CE、Weighted
    Cross-Entropy 和 Focal Loss 三种模式。Weighted CE 为背景和病灶分别设定不同的类别权重，
    用于调节正负样本对分类损失的贡献比例；Focal Loss 通过 gamma 参数自动降低简单样本的损失
    贡献，迫使模型聚焦于困难样本的学习。
12. 引入在线难例挖掘（OHEM）机制（通过 --ohem 开关启用）：先逐样本计算独立 loss，按困难度
    降序排列后仅保留前 K 个最困难样本参与反向传播。K 取 --ohem-ratio 和 --ohem-min-samples
    二者较大值。该策略可与任意损失函数组合使用。
13. 将训练阶段 RPN 和 ROI Head 的 IoU 匹配阈值提取为命令行可配置参数
    （--box-fg-iou-thresh / --box-bg-iou-thresh / --rpn-fg-iou-thresh / --rpn-bg-iou-thresh），
    支持将 ROI 正样本 IoU 阈值从默认 0.5 上调至 0.6/0.7 以减少边界模糊导致的误检。
14. 新增两阶段训练策略（positive-only warmup，通过 --warmup-positive-epochs 配置）：前 N 个
    epoch 仅使用正样本图像训练，让模型先学习病灶特征；之后恢复全量样本训练。验证集保持真实分布。
-----------
15. 废弃 Positive-Only Warmup，改用 Balanced Sampling Warmup（通过 --warmup-balanced-epochs
    和 --warmup-pos-weight-ratio 配置）：前 N 个 epoch 使用 WeightedRandomSampler 对训练集进行
    加权采样，使正样本被过采样至与负样本接近的数量，同时 RPN 仍能学到背景抑制。warmup 结束后重建
    optimizer 和 lr_scheduler，切换回全量训练。
16. 禁止 Focal Loss + OHEM 同时启用：两者功能高度重叠，同时使用会过度稀释梯度信号。新增启动时
    互斥检查，当同时启用时自动禁用 OHEM 并打印警告。
17. 修复 CosineAnnealingLR 的 T_max 与 warmup 阶段冲突：将 T_max 设为实际全量训练的 epoch 数
    （total - warmup），并在 warmup 阶段跳过 epoch-level scheduler step，防止 LR 在 warmup
    阶段被 cosine 过度拉低。warmup 结束后重建 optimizer 和 scheduler 以获得干净的 LR 曲线。
18. 在 validate_one_epoch 中增加多阈值验证报告（threshold=0.1/0.3/0.5/0.7/0.9），输出各阈值
    下的 TP/FP 统计，帮助更全面地理解模型行为，同时不影响 best checkpoint 的选择逻辑。
19. 在 validate_one_epoch 中增加 RPN proposal 数量监控：记录模型输出的原始 box 数量（threshold
    之前），当每图平均值超过 200 时输出警告，用于检测 RPN 是否有效抑制背景。
20. 在 build_model 中暴露 box_score_thresh 和 box_detections_per_img 参数（对应命令行
    --box-score-thresh 和 --box-detections-per-img），控制推理时的低分框过滤和每图最大检测数。
21. 新增 --only-use 参数（float，默认 1.0）：在正式训练阶段每个 epoch 仅使用分配数据总量的指定
    比例。正负样本按原始比例等比缩减，并对正样本设最低保护（不低于原始比例对应的数量）。通过跨
    epoch 轮转采样机制，确保所有图像在多个 epoch 中均被训练到。
-----------
22. 修复误导性警告：将 scheduler 更新判断从二分支（warmup | else）扩展为四分支，明确区分
    "warmup 阶段跳过"、"本 epoch 刚重建 optimizer（正常行为）"、"正常 step"、"真正 0 步（异
    常）"四种情况，避免重建后误报 [Warning] No optimizer.step()。
23. 新增 --post-warmup-lr 参数（float，默认 None 即沿用 --lr）：balanced warmup 结束后重建
    optimizer 时使用该 LR 代替原始 --lr，允许以较低学习率进入正式训练阶段，减少 warmup 结束
    后的梯度震荡。若未传该参数，行为与之前完全一致。
24. 新增 freeze_epochs / warmup 冲突检测：若 0 < freeze_epochs < warmup_balanced_epochs，
    则在启动时打印 [Warning]，提示用户该配置会在 warmup 内部触发 optimizer 重建，可能引发
    FP 反弹，建议设为 0 或 >= warmup_balanced_epochs。
25. 在每个 epoch 循环开始处打印分割线（"=" * 60 + "Epoch n/N start" + "=" * 60），便于在
    长日志中快速定位各 epoch 的起止边界。
26. 在 weighted_ce 模式下新增背景权重保护：若训练集负样本图像明显多于正样本图像，且
    --cls-weight-bg < 1.0，则默认将背景权重钳制到 1.0 并打印告警；只有显式传入
    --allow-low-bg-weight 时才允许保留更低的背景权重，以避免背景分类损失被过度削弱、导致
    bbox 误检（FP）爆炸。
27. 启动时打印 train/val 的 neg/pos image ratio，并对 --only-use < 1.0 做阳性样本量预警：
    若估算每轮可见正样本图像过少，则提示该配置可能导致正式训练阶段学习不足与 FP 反弹。同步
    更新推荐命令：默认不再推荐 --only-use 0.3，并将 --cls-weight-bg 推荐值调整为 1.0。
28. 为了更符合“病灶区域的初步框选”目标，在训练和验证阶段默认启用乳房主体区域裁剪：
    先在 processed 图像上检测乳房主体轮廓，再对图像进行裁剪，并将 bbox 同步重映射到裁剪后
    的局部坐标系，减少大面积黑色背景和非乳房区域对 detector 的干扰。
29. 将乳房主体裁剪的开关和边缘留白提取为参数（--disable-breast-crop / --breast-crop-margin），
    并把裁剪配置写入 checkpoint meta，要求测试脚本优先读取同一配置，保持训练/测试口径一致。
-----------
30. 新增随机数据增强支持（通过 --augment 启用），仅作用于训练集：
    - 随机水平翻转（--aug-hflip-prob，默认 0.5）；
    - 随机亮度扰动（--aug-brightness-delta，默认 ±0.2）；
    - 随机小角度旋转（--aug-rotation-max-deg，默认 0.0，推荐 5~10）：
      只对单框图像执行（多框跳过以避免 pivot 歧义）；旋转轴为 bbox 中心；
      bbox 以 center-preserve 方式更新（中心旋转，尺寸不变）；
      旋转后调用 detect_breast_region 二次裁剪去除黑角。
    用 TrainAugmentWrapper 包裹训练 Subset，不创建额外数据集实例。
31. 修复 RPN 正样本 IoU 阈值：将 --rpn-fg-iou-thresh 默认值从 0.7 降低到 0.5。
    背景：处理后图像中典型病灶（resize 后约 225×167px）与最优 anchor（256px）的最大 IoU 仅
    约 0.57，低于原始默认值 0.7，导致病灶 anchor 落入"灰色区间"被忽略，RPN 无法有效学习
    病灶位置。降低阈值后这些 anchor 将被标记为正样本，显著改善 RPN 召回。
32. 新增全训练阶段持续加权采样（--full-train-pos-weight-ratio，默认 0.0 即禁用）：在
    balanced warmup 结束进入全量训练后，若该参数 > 0，则继续使用 WeightedRandomSampler
    对正样本保持轻微过采样（如 3.0 表示正样本被采到的概率是负样本的 3 倍），防止模型因
    图像级 10:1 负正比而坍缩至"什么都不预测"的极端状态。
-----------
33. 针对"FP 极高、置信度普遍 > 0.9、F1 长期低于 0.02"的训练失效问题，进行系统性修复：
    a. 参数调整：将 --warmup-balanced-epochs 从 8 缩短至 5、--warmup-pos-weight-ratio 从
       10.0 降至 3.0（减少 warmup 过采压力）；将 --full-train-pos-weight-ratio 从 3.0 改
       为 0.0（全量训练阶段不再强制过采样，让背景梯度充分抑制 FP）；将
       --cls-weight-bg 从 1.0 提高至 2.5（直接加大 FP 的 loss 惩罚）；将
       --cls-weight-lesion 提高至 2.0（加大 FN 惩罚）；将 --box-detections-per-img 从
       100 降至 10（推理端限制每图最大输出框数）。
    b. 新增 --box-nms-thresh 参数（默认 0.5，推荐 0.3）：暴露 Faster R-CNN 的 NMS IoU
       阈值，降低后可更积极地去除空间相近的重复 FP 框，直接减少推理输出量。
    c. 新增 --rpn-objectness-loss-scale 参数（默认 1.0，推荐 3.0）：在 train_one_epoch
       中将 loss_objectness 乘以该系数后再计入总 loss，强化 RPN 拒绝背景区域的训练信号，
       从 proposal 生成端压低进入 ROI head 的背景框数量。
    d. 修复 checkpoint 选择逻辑：validate_one_epoch 额外追踪每个阈值（0.1/0.3/0.5/0.7/
       0.9）的 FN，计算各阈值 F1，返回最优阈值及 best_thresh_f1；main() 改用
       best_thresh_f1 代替固定阈值 F1 来判断是否保存最佳 checkpoint，避免因 score
       threshold 选取不当而丢弃真正最优的 epoch。
-----------
34. 彻底切换检测框架：从 Faster R-CNN 迁移至 RetinaNet_ResNet50_FPN_V2，同时完成
    医学影像灰度输入适配，从根本上解决两阶段 RPN 瓶颈与 Focal Loss 无法原生作用于
    proposal 生成阶段的双重缺陷：
    a. 架构替换：删除 fasterrcnn_resnet50_fpn_v2 / FastRCNNPredictor / roi_heads
       monkey-patch 及相关 import；新增 retinanet_resnet50_fpn_v2 import。RetinaNet
       是单阶段检测器，直接在 FPN 各尺度特征图上预测，彻底消除了 RPN "漏斗" 瓶颈
       （Faster R-CNN 中约 70% GT box 从未进入 ROI Head 的根本原因）。
    b. 删除自定义 loss 模块：移除 FocalLoss 类、_custom_fastrcnn_loss 函数、
       apply_custom_roi_loss 函数（约 120 行），因为 RetinaNet 已原生集成
       Focal Loss，无需 monkey-patch。
    c. 重写 build_model()：加载 COCO 预训练 backbone 权重 →
       构建 2 类 RetinaNet（num_classes=2）→ 通过
       model.head.classification_head.focal_loss_alpha/gamma 直接设置 Focal Loss
       参数，无需外部注入。新增参数 --focal-alpha（默认 0.75，面向 10:1 不平衡数据
       比 RetinaNet 原始论文 0.25 更高，确保病灶梯度足够）。
    d. 新增灰度 conv1 均值初始化（方向二轻量实现）：钼靶图像为灰度图，以 R=G=B
       形式加载为 3 通道。ImageNet 预训练的 conv1 三个通道权重来自不同颜色语义，
       对等值输入不适合。在 build_model() 中对 conv1 的 3 通道权重取均值后广播
       覆盖，使三个通道初始化完全一致，减少训练初期的通道间梯度不平衡。
    e. 简化 train_one_epoch()：移除 rpn_objectness_loss_scale 参数及相关加权逻辑；
       subloss_keys 改为 RetinaNet 的 ("classification", "bbox_regression")。
    f. 精简 parse_args()：删除 Faster R-CNN 专有参数（--roi-batch-size-per-image、
       --roi-positive-fraction、--rpn-*、--cls-loss-type、--cls-weight-*、
       --ohem、--allow-low-bg-weight、--rpn-objectness-loss-scale，共约 13 个）；
       新增 --focal-alpha（默认 0.75）；保留 --focal-gamma（默认 2.0）；
       --box-fg-iou-thresh 和 --box-bg-iou-thresh 现在直接映射到 RetinaNet 的
       anchor 正负样本 IoU 阈值。
    g. 更新推荐命令：简化至 24 行，去掉所有 RPN/ROI 参数，直接通过
       --focal-alpha 0.75 --focal-gamma 2.0 控制 Focal Loss。
-----------
35. 修复两项影响 rec_34 收敛的根因缺陷，无架构改动：
    a. 正样本权重恢复：将 --full-train-pos-weight-ratio 从 0.0 改为 2.0。
       rec_34 中该参数意外为 0，导致暖机阶段结束后所有正样本权重变为 0，约 49%
       的有效 batch（8 张图）完全由负样本组成，分类头在这些 batch 中梯度为 0，
       模型在主训练阶段实际上无法从正样本中学习。
    b. Anchor 尺寸上移：将 --anchor-sizes 从 16,32,64,128,256 改为
       32,64,128,256,512。数据验证显示 VinDr-Mammo 病灶在模型内部分辨率
       （800×1333px）下 p90 宽度为 419px，而原最大 anchor 仅 256px，导致大病灶
       无有效 anchor 匹配；同时 16px/32px anchor 在数据集中覆盖不足 5% 的病灶，
       属于无效噪声。新配置将 IoU≥0.4 覆盖率从 91.0% 提升至 97.1%。
    ** 注：rec_35 实际结果证明，仅改这两项不够——pos_weight=2.0 消除了全负
       batch 的 FP 抑制梯度，导致 BestThreshF1 从 0.1357 退步至 0.0981，
       FP@0.5 从 100–450 暴涨至 550–1500。rec_35 在 epoch 14 提前终止。
-----------
36. 基于 rec_34/rec_35 对比数据，精准修复两项 Focal Loss 相关的超参设置：
    a. 撤销 pos_weight 改动：将 --full-train-pos-weight-ratio 从 2.0 恢复为 0.0。
       rec_35 vs rec_34 的直接对比（相同架构，仅 pos_weight 不同）显示，
       FP@0.5 从 100–450 暴涨至 550–1500（3–4 倍增幅）。根因：pos_weight=0.0
       时约 49% 的有效 batch 为全负样本，这些 batch 对 Focal Loss 提供了纯 FP
       抑制梯度；pos_weight=2.0 消除了这些 batch，模型失去"不要乱预测"的信号。
    b. 降低 focal_alpha：将 --focal-alpha 从 0.75 改为 0.5。torchvision RetinaNet
       中 alpha 是正类（病灶）的损失权重，1-alpha 是负类（背景）的权重。
       原始 RetinaNet 论文使用 alpha=0.25（背景权重 0.75）；我们此前设 0.75
       导致背景权重仅 0.25，高置信 FP 的背景梯度贡献仅为正常的 1/4。将 alpha
       降至 0.5 使背景权重翻倍（0.25→0.50），对所有高置信 FP 的惩罚强度提升
       2 倍，同时保持正负类梯度平衡，不至于压垮病灶学习。
    保留 anchor_sizes=32,64,128,256,512（覆盖率 97.1% 有效）。
    预期：FP@0.1 降至 50–400 范围，best_thresh 提前稳定在 0.3，
    BestThreshF1 突破 rec_34 的 0.1357 上限，目标 0.16–0.20。
-----------
37. 数据预处理 + 医学预训练 backbone 双重升级（rec_37）：
    a. CLAHE 对比度增强（方向 A）：在 normalize_image() 的灰度图路径中，
       将 CLAHE（clipLimit=2.0, tileGridSize=8×8）应用于转换为 RGB 前的
       灰度图，增强钼靶图像的局部对比度，使病灶与周围腺体的灰度差异更清晰，
       改善特征提取质量。CLAHE 作用于训练集和验证集全部图像（非仅增强）。
    b. 医学预训练 backbone（方向 C）：在 build_model() 的 Step 3 中，
       新增 medical_backbone_path 参数；若提供，则用 RadImageNet ResNet50
       权重覆盖 COCO backbone，大幅缩短迁移链
       （ImageNet/CXR → 多模态放射图像 → 乳腺钼靶），使特征空间更接近
       乳腺影像语义。加载时自动剥离 module./encoder./backbone./body. 等
       前缀，以 strict=False 方式匹配 model.backbone.body；若加载失败则
       自动回退至 COCO 权重。
    背景：rec_34/35/36 共 63 个 epoch 中，best_thresh 几乎始终锁在 0.1，
    最大 recall 仅 26.6%，BestThreshF1 增幅仅 4.7%（0.1357→0.1421）。
    超参数调整空间已接近耗尽，问题本质在于 COCO 预训练特征对乳腺病灶
    的语义表达能力不足。
    ── rec_37 三项 bug 修复（训练过程中发现）──
    c. RadImageNet key 映射修复（commit e201432）：
       RadImageNet ResNet50.pt 使用数字索引键（backbone.0.weight 等），
       与 torchvision 的命名键（conv1.weight、layer1.*.weight 等）不匹配，
       导致全部 318 个骨干参数被当作 unexpected 跳过，权重实际未加载。
       修复：剥离 backbone. 前缀后，额外映射
       0→conv1, 1→bn1, 4→layer1, 5→layer2, 6→layer3, 7→layer4。
       验证：Missing=0, Unexpected=0。
    d. anchor_generator 关键字参数冲突修复（commit 6c63ca0）：
       当前 torchvision 已在工厂函数内部以位置参数传入 anchor_generator，
       我们再以 kwarg 方式传入时触发 TypeError，except 块静默吞掉了
       anchor_sizes、detections_per_img、fg/bg_iou_thresh 等全部参数，
       回退至 torchvision 默认值（detections_per_img=300 等）。
       rec_33–rec_36 及 rec_37 前两次均受此影响，所有自定义检测参数均未生效。
       修复：不再向工厂函数传 anchor_generator，改为构建后赋值
       model.anchor_generator = anchor_generator。
    e. RetinaNet head 重建修复：
       工厂函数按默认 9 anchors/location（3 scales×3 aspects）建造 head；
       替换 anchor_generator 后实际产生 3 anchors/location，
       head 输出通道数不匹配，在 classification_head.compute_loss 时触发
       IndexError: shape [67200] != [201600, 2]。
       修复：替换 anchor_generator 后立即用正确的 num_anchors 重建
       model.head（RetinaNetHead + GroupNorm32）。
-----------
38. COCO head 保留 + RadImageNet backbone + copy-paste 数据增强（rec_38）：
    a. 根因分析：rec_37 在 head 重建（e 项修复）时将 3 anchors/location 的新
       head 完全随机初始化，同时丢失了 COCO 回归 head 中 8.8M 参数的预训练先验；
       加之从未有过 COCO 类级别的分类先验，导致 BestThreshF1 从 rec_36 的 0.1421
       退步至 0.0700。
    b. 保留 COCO head 策略：构建 weights=None、num_classes=2 的模型（9 anchors/
       location，torchvision 默认），然后以 strict=False 加载完整 COCO state_dict；
       分类 head 因 91→2 类形状不匹配自动跳过（随机初始化），回归 head 和 FPN
       完整加载。不再手动重建 head，不再传 anchor_generator。
    c. RadImageNet backbone 覆盖（保留）：COCO weights 加载后，以 strict=False
       用 RadImageNet ResNet50 权重覆盖 backbone.body，使特征提取器贴近放射影像
       语义。conv1 仍做 3 通道均值适配（灰度图输入）。
    d. Copy-paste 数据增强（CopyPasteWrapper 类，--copy-paste-prob 0.4）：
       - 触发条件：仅对负样本（无标注框）以 paste_prob 概率触发。
       - Donor 采样：从正样本（有框）中随机抽取，对每个 bbox 扩展 30% 后裁出
         包含病灶的 crop。
       - Tissue 约束：在均值图（mean_img > 0.05）的乳腺组织区域内随机选取
         粘贴位置；最多尝试 5 次，要求粘贴窗口内 70% 以上像素为组织。
       - LCC mask（去除黑色背景）：对 crop 做 Otsu 阈值 + 3×3 dilate +
         connectedComponents，取最大前景区域，消除矩形 crop 内的黑色残留。
       - Feathering（alpha 渐变）：对 LCC mask 做 distanceTransform，
         alpha_map = clip(dist / 5, 0, 1)，使病灶边缘平滑融合。
       - 亮度对齐（P95 基准）：取 crop 和目标区域内病灶像素各自的第 95 百分位
         亮度，按比值缩放 crop（clamp 0.5~2.0），使粘贴病灶亮度匹配背景组织。
         使用 P95 而非均值，避免黑色像素将均值拉低导致亮度估计偏差。
       - 中心压暗（高斯 gamma map）：对亮度对齐后的 crop 施加中心 gamma=0.75、
         边缘 gamma→1.0 的高斯 gamma 映射，模拟病灶中心密度较高的自然外观。
       - Feathered blend：crop_adj * alpha_t + background * (1 - alpha_t)。
       - 每张负样本最多粘贴 max_pastes（默认 2）个病灶，粘贴结果同时写入 targets。
       - 重叠检测：每次粘贴前检查候选框与已粘贴框的 IoU，若 > 0.3 则跳过
         本次粘贴，避免像素覆盖和重复监督信号。
    e. 新增 CLI 参数：
       - --copy-paste-prob（默认 0.0）：copy-paste 触发概率。
       - --copy-paste-max-pastes（默认 2）：每张负样本最多粘贴数量。
-----------
39. 路线 3 改进 + 召回率优先策略（针对 rec_38 F1 停滞、并重新定位任务目标）：
    背景：bbox 检测器的最终用途是为"是否患病"分类器提供病灶位置特征，
    检测器本身不需要精确分类，只需要高召回率（不漏病灶），FP 增多可以接受。
    a. 新增 --input-min-size 参数（默认 800，推荐 1200）：
       控制 RetinaNet 内置 resize transform 的短边目标尺寸（model.transform.min_size）。
       原始图像 912×1520，resize 后 P10 最小病灶仅 28px，低于最小 anchor（32px）。
       提升至 1200 后，resize scale 从 0.877 变为 1.316，病灶 P10 增至约 42px，
       覆盖 P3 anchor（32–50px），降低约 14% 的病灶因分辨率不足而被漏检的概率。
    b. 将 --focal-alpha 从 0.5 提升至 0.8：正样本（病灶）loss 权重是负样本的 4 倍，
       漏检（FN）的惩罚远大于误报（FP），驱动模型输出更高召回率。
    c. 将 --box-nms-thresh 从 0.3 放宽至 0.5：减少同一病灶的重复框被 NMS 压制，
       避免因 NMS 过激进而丢失真实病灶框。
    d. 将 --box-score-thresh 从 0.05 降至 0.001，--box-detections-per-img 从 10 增至 20：
       几乎输出所有候选框，召回率上限不再受推理端阈值约束；由后续分类器决定置信度。
    e. 关闭 copy-paste（--copy-paste-prob 0.0）：rec_38 对比 rec_37 F1 下降
       （0.052 < 0.070），copy-paste 合成样本与真实病灶分布差异导致验证集性能下降。
    f. train/test 双侧同步：bbox-test.py build_model() 和 meta 读取均已更新，
       从 checkpoint meta["input_min_size"] 恢复训练时的 resize 设置。
    g. 新增 --recall-stop 标志（默认关闭）：开启后以 Recall@0.1（TP@0.1 / GT框总数）
       代替 BestThreshF1 作为早停和最佳 checkpoint 的判断指标，与召回率优先目标对齐。
    h. 日志新增 [Recall] 行：每 epoch 打印 GT_boxes 总数、Recall@0.1/0.3/0.5 及绝对
       TP/GT 数字，方便追踪召回率改善趋势；早停日志同步显示当前监控指标名称。
-----------
40. 针对 rec_39 全量阶段 Recall@0.1 停滞在 0.03–0.15 的问题，进行两项修复（rec_40）：
    根因：全量训练阶段（epoch 6+）正负比 1:10.49，--full-train-pos-weight-ratio=0.0
    导致 effective batch（8 张）中平均仅 0.76 张正样本图，每轮梯度更新中来自病灶的
    分类信号极度稀疏；同时 classification loss 和 bbox_regression loss 等权叠加，
    bbox_regression loss（~0.030）在全量阶段几乎不变，classification loss 的有效
    波动被稀释。
    a. 恢复全量阶段正样本过采样：将 --full-train-pos-weight-ratio 从 0.0 改为 3.0。
       WeightedRandomSampler 使正样本被采到的概率是负样本的 3 倍，effective batch
       中正样本图从平均 0.76 张提升至约 2.3 张（正负比 ~1:2.5），梯度更新频次提升
       约 3 倍。注：rec_35 曾出现 pos_weight=2.0 导致 FP 爆炸，但彼时用的是
       focal_alpha=0.5（正负类等权），现在 focal_alpha=0.8（病灶权重是背景的 4 倍）
       可以提供足够的 FP 抑制梯度，两者协同不会再出现 FP 爆炸问题。
    b. 新增 --classification-loss-scale 参数（默认 1.0，rec_40 使用 2.0）：
       在 train_one_epoch 中将 loss_dict["classification"] 乘以该系数后再求和。
       RetinaNet 只有 classification 和 bbox_regression 两个 loss，bbox_regression
       在全量阶段已趋于稳定（~0.030），乘以 2.0 使分类信号占总 loss 的比例从
       ~64% 提升至 ~78%，强化模型区分病灶与背景的训练压力。
-----------
41. 算法层面修复：验证评估无截断 + 参数重平衡（rec_41）：
    根因分析（rec_40 失败，更深层退化解）：
    (1) focal_alpha=0.8 使背景惩罚权重仅 0.2，模型找到最优策略：对每张图输出满额
        20 个框（消灭 FN cost=0.8），FP 代价（cost=0.2）可忽略。FP@0.1≈43,540，
        几乎每张图输出满 20 个框（score≥0.1），验证为退化解。
    (2) validate_one_epoch 中 model(images) 输出受 detections_per_img=20 硬截断，
        TP@0.1 指标的真实上限被人为压低——即便模型能产生更多正确框，也无法反映在
        Recall@0.1 上，导致早停和 checkpoint 选择均基于失真的指标。
    (3) WeightedRandomSampler pos_weight=3.0 在 13596 个 sampled 样本的竞争中被
        摊薄，有效正样本期望仅约 2.3 张/batch，远未达到 3 倍过采效果。
    a. 新增 --val-detections-per-img 参数（默认 300）：在 validate_one_epoch 中，
       进入验证前临时将 model.detections_per_img 设为该值（默认 300），验证结束后
       恢复训练推理值。这使 TP@0.1 的计算从"20 框里有几个对的"变成"300 框里有
       几个对的"，TP@0.1 指标不再被推理截断，真实反映模型召回能力，早停和最佳
       checkpoint 选择基于准确指标。
    b. focal_alpha 0.8 → 0.5：背景惩罚权重从 0.2 提升至 0.5，迫使模型真正区分
       病灶与背景，而不是选择"全部输出高分"的退化策略。正负类梯度重新接近平衡
       （1:1），FP 的 loss 代价显著提升，退化解不再是全局最优。
    c. pos-weight-ratio 3.0 → 10.0（warmup 和全量阶段均调整）：在 13596 个训练
       样本中，权重 10.0 使正样本期望提升至约 4.1 张/batch
       （10×1183 / (10×1183+12413) × 8 ≈ 4.1），更充分地保证每个梯度更新步
       包含足够的病灶监督信号。

-----------
42. 三方向算法改进（rec_42，分支文件 bbox-train-A/B/C.py，原文件不变）：

    根因分析（rec_41 失败，val_precision≈0.0000 退化解）：
    91% anchor score≥0.5（8 TP / 599,039 predictions），根本原因：67,500 背景
    anchor / 图 vs 3-5 个病灶 anchor，梯度被背景淹没，模型退化为"全部高置信度"。

    方案 A（bbox-train-A.py）— Anchor-level Hard Negative Mining（HNM）：
        在原 RetinaNet 框架内 monkey-patch classification_head.compute_loss；
        对每张图的背景 anchor 计算 per-anchor focal loss，只保留 top-K 最难负样本
        K = max(n_fg × neg_pos_ratio, min_neg)，彻底解决梯度淹没问题。
        新增参数：--hnm-neg-pos-ratio 10 --hnm-min-negatives 512
        调参变化：--classification-loss-scale 1.0，--warmup-pos-weight-ratio 3.0，
                  --full-train-pos-weight-ratio 3.0

    方案 B（bbox-train-B.py）— FCOS（anchor-free）：
        将模型从 RetinaNet 替换为 fcos_resnet50_fpn；anchor-free，无 anchor 比例失
        衡问题；中心度分支（centerness）天然抑制非中心位置的高置信度预测。
        sub-losses：classification / bbox_regression / bbox_ctrness
        调参变化：去掉 --box-fg/bg-iou-thresh（FCOS 不使用 anchor matching）

    方案 C（bbox-train-C.py）— U-Net 分割 → 伪框：
        放弃检测框架，改用 ResNet50 + 4 级解码器（跳跃连接）输出全分辨率热图；
        使用 BCEWithLogitsLoss（pos_weight=100），验证时通过 cv2.connectedComponents
        将热图 blob 转换为伪框进行 recall 评估；不再依赖 anchor / NMS。
        新增参数：--seg-pos-weight 100.0 --seg-val-threshold 0.5
        去掉参数：所有 RetinaNet 专有参数（focal-*, anchor-sizes, box-*-thresh 等）

-----------
43. 图像级二分类器前置过滤（rec_43）：
    根因分析：训练集中 91% 为负样本图像（12413/13596），RetinaNet 在约 91% 的
    batch 中接收"无病灶"图像的 anchor 级梯度，分类头同时学习"找到病灶"和"抑制
    整张正常图像的所有 anchor"两个竞争任务，梯度方向持续对消。从图像层面切断负
    样本流，将检测器有效训练集的图像级负正比从 10.5:1 降低至约 0.25:1。

    两阶段训练流程：
    阶段一（图像级分类器，5–10 轮）：在全部 13,596 张训练图像上用 ResNet50 +
    全局平均池化 + Dropout(0.5) + FC(2048→1) 训练图像级二分类器（has_lesion /
    no_lesion）；BCEWithLogitsLoss，pos_weight 与训练集负正比匹配（默认 10.0）；
    Adam 优化器，LR=1e-3，CosineAnnealingLR 衰减。阶段一 checkpoint 保存至
    models/image_clf.pth，可通过 --clf-checkpoint-path 跳过重新训练复用旧权重。
    阶段二（RetinaNet 检测器，原始流程）：将阶段一分类器应用于全部训练图像，仅
    保留预测为"有病灶"（sigmoid > --clf-threshold）的图像进入 RetinaNet 训练集；
    同时强制保留全部原始正样本图像，防止分类器假阴性导致正样本丢失；验证集保持
    完整真实分布（不过滤）。

    新增参数：
    - --clf-epochs（默认 10）：阶段一图像分类器训练轮数。
    - --clf-lr（默认 1e-3）：阶段一 Adam 学习率。
    - --clf-pos-weight（默认 10.0）：BCEWithLogitsLoss 正样本权重。
    - --clf-threshold（默认 0.5）：分类器过滤阈值；sigmoid 高于此值的图像进入
      检测训练集。
    - --clf-save-path（默认 models/image_clf.pth）：阶段一 checkpoint 保存路径。
    - --clf-checkpoint-path（默认 None）：若提供则跳过阶段一，直接加载已有分类
      器权重，方便复用已训练好的分类器。
    - --skip-clf-stage（默认关闭）：跳过阶段一，使用完整训练集进行检测，等价于
      原始 bbox-train.py 的行为。

-----------
44. 架构彻底切换：将检测问题重构为 patch 级滑窗二分类（rec_44）
    【此文件未作修改，完整实现在分支文件 bbox-train-D.py 中】（方案已舍弃）

-----------
45. 方向 E：U-Net 全图分割检测
    【此文件未作修改，完整实现在分支文件 bbox-train-E.py 中】（已删除）

-----------
46. 方向 F：U-Net Patch 训练检测（解决保守性坍缩根因）
    【此文件未作修改，完整实现在分支文件 bbox-train-F.py 中】（已删除）
    根因分析：方向 E 中全图训练的像素正负比约 500:1，pos_weight=50 仍不足以抵抗
    背景梯度（背景:病灶梯度 ≈ 10:1），模型总能找到"少预测"捷径降低 loss，导致
    保守性坍缩。方向 F 改为 Patch 训练：以 GT bbox 为中心裁取 256×256 patch，
    正负各 50%，正样本像素比提升至 5–20%，梯度不平衡根因被消除。
    验证/推理仍用全图（U-Net 全卷积支持任意尺寸）。
    最终结果（rec_46_upd_3）：F2@0.9=0.4652，FP@0.9=599。
    已训练模型：models/bbox_resnet50.F.pth

-----------
47. 方向 G：两阶段检测 — Stage 2 ROI 分类器过滤 FP（rec_47）
    【此文件未作修改，完整实现在分支文件 bbox-train-G.py 中】（已删除）
    根因分析：方向 F 在 256–384 感受野下无法利用全局上下文区分"病灶"与
    "病灶样乳腺实质"，导致高置信度 FP 无法从 heatmap 层面消除。
    方向 G 在 Stage 1（U-Net heatmap → NMS → 候选框）之后添加 Stage 2：
    对每个候选框 crop（224×224，1.5 倍上下文填充）运行 ResNet50 二分类器，
    过滤 Stage 2 打分低的 FP 候选，目标将 FP@0.9 从 ~600 降至 <200。
    Stage 2 编码器复用 Stage 1（方向 F）的 ResNet50 权重（encoder-lr-multiplier=0.01）。
    训练样本：GT box crop（正）+ 正样本图随机 crop（硬负）+ 负样本图随机 crop（易负）。
    根本瓶颈（进一步追因）：
    - Stage 1 F 在 score_thresh=0.5 时 FP=1068，recall≈71%，约 17% GT 框热图无响应。
    - 降低 stage1_threshold=0.2（upd_5）：候选暴增至 5396，Stage 2 F2 从 0.376 降至 0.289。
    - 热图后处理管线（连通域→NMS）是 FP 的主要来源，无法从 Stage 2 端彻底消除。
    - F+G 两阶段误差累积：Stage 2 精度上限被 Stage 1 recall 锁死。

-----------
48. 方向 H：RetinaNet 全图直接检测（替代 F+G 两阶段流水线，rec_48）
    【此文件未作修改，完整实现在分支文件 bbox-train-H.py 中】
    切换动机：
    - F+G 根本缺陷是热图后处理管线（连通域→NMS）产生大量 FP，且 Stage 1 召回率天花板约 83%。
    - H 直接用 RetinaNet + ResNet50-FPN 全图回归 box，消除热图后处理 FP 来源。
    - Focal Loss 内置极端不平衡处理（γ=2.0，α=0.25），无需手动调 pos_weight/Tversky。
    - FPN P3–P7 五层特征图同时覆盖 32–512px 尺度，直接输出 (box, score)。
    关键实现：
    - 输入尺寸：1024×512；设 min_size=512, max_size=1024 后 torchvision 内部不再 resize。
    - anchor sizes=(32,64,128,256,512)，aspect_ratios=(0.5,1.0,2.0)×5，3 anchors/cell。
    - RadImageNet ResNet50 权重加载（同 F 脚本 key 映射）；conv1 权重通道均值适配。
    - 正样本图像过采样（pos_oversample_factor=4.0）补偿 7% 正样本比例。
    - 差异 LR：backbone.body（encoder_lr=lr×encoder_lr_multiplier）vs FPN+head（full lr）。
    - val 多阈值报告（score@0.1–0.9）输出 TP/FP/FN/Recall/F1/F2。
    - patient_level_split 使用 sorted() 确保确定性（与 G 一致）。
    目标：超越 F+G 最佳 F2=0.4652，争取 F2 > 0.50。
    最终结果（rec_48 upd_6）：best F2@0.3（固定阈值）=0.2788，ep9，ep9 之后无法突破。
      val GT boxes=251，Recall@0.1≈53%，TP@0.3=60，TP@0.7=23。
      根因：监控目标 fbeta2_ref（固定 @0.3）导致模型学会低置信度输出；
      更深层根因：Asymmetry / Architectural_Distortion 需双侧对比才能识别，单视图无学习信号。
    已训练模型：models/bbox_resnet50.H.pth

-----------
49. 方向 I：双侧对比检测（Bilateral Contralateral Comparison，rec_49）
    【此文件未作修改，完整实现在分支文件 bbox-train-I.py 中】
    切换动机：
    - 方向 H (upd_6) 最终结果：best F2@0.3=0.2788，ep9，之后无法突破。
    - 根因：单视图缺少双侧不对称语义信号；Asymmetry / Architectural_Distortion 类病变
      需对侧对比才能识别，单图 RetinaNet 对此类病变几乎无学习信号。
    - 方向 I 引入对侧同视角图作为参照，将三通道 [primary, contra, |primary-contra|] 送入
      RetinaNet，差分通道显式提供"哪一侧不对称"的语义信号。
    关键实现：
    - BilateralSample：新增 laterality / view / contra_path 字段；
      每张图恰好作为一次 primary，训练样本数 ≈16000（同 H）。
    - 标准坐标系（canonical left-facing space）：R 侧水平翻转后与 L 侧解剖对齐；
      翻转后差分通道反映真实解剖不对称性。
    - conv1：primary / contra 通道继承 RadImageNet 均值权重；diff 通道 0.1× 权重
      （让模型缓慢学习如何利用差分信号）。
    - 增强：对比度 / 缩放，禁用 hflip（会破坏双侧坐标系语义）；
      对比度因子同时作用于 primary 和 contra，保持差分通道比例。
    - 数据验证：4999/5000 患者有完整四视角；881/924 阳性患者（95.3%）为单侧病变，
      对侧可作为健康参照。
    目标：突破 H 的 F2@0.3=0.2788，期望 >0.30。
    实验结果（2026-05-24，50 ep，早停 ep14）：
      best F2@0.3=0.1901，ep3；早停于 ep14（patience=10）
      Recall@0.3=15.9%（ep3），F2@0.1=0.4207（ep3）
      训练极不稳定（ep2 cliff 至 0.0149，ep3 反弹后持续震荡）
    失败根因分析：
      - 像素级差分 |primary−contra| 无法产生有效解剖不对称信号：
        两侧乳腺形态/亮度差异无配准，逐像素相减主要是结构噪声。
      - diff 通道（0.1× 初始权重）在初期不参与学习，ep3 峰值恰为其尚未"搅局"的窗口；
        后期激活后反而干扰 primary 通道判断。
      - 方向 I 判定：失败；不保留模型，文件已删除（后续改为方向 J）。

50. 方向 J：Global-Local 特征融合 + ROI 重打分（rec_50）
    【此文件未作修改，完整实现在分支文件 bbox-train-J.py 中】
    切换动机：方向 I 失败（F2@0.3=0.1901），像素级差分无有效解剖对称信号。
    根本改进方向：用特征层融合代替像素层融合，让模型自己学习如何利用全局上下文。
    同时，针对 H 暴露的"找到框但置信度偏低"问题（F2@0.1≈0.40 vs F2@0.3≈0.28），
    增加 ROI 重打分头，在训练和推理中端到端校正置信度。
    关键实现：
      1. GlobalContextEncoder（新增类）
         - 将原始图像下采样至 1/4（256×128），经 3 个 stride-2 BN-ReLU conv 块
           得到 128 通道全局特征图 @ 32×16（与全图 stride-32 对齐）
         - 捕获乳腺密度分布、整体轮廓等宏观上下文信息
      2. GlobalAwareBackbone（新增类）
         - 包装 ResNet50-FPN（即方向 H 的 backbone）
         - 在每个 FPN 层（P3~P7）将全局特征图上/下采样到对应空间尺寸，
           与局部特征 concat 后经 1×1 conv 融合（通道数保持 256）
         - 初始化策略：1×1 conv 前 256ch identity，后 128ch 全零 → 训练初期
           全局路径无影响，逐步学习利用全局信息，不破坏预训练特征
         - 每次 forward 后缓存融合后的 FPN 特征（供 ROI 头使用）
      3. RoiRefinementHead（新增类）
         - 从 P3（stride=8，最高空间分辨率）做 ROI Align 7×7
         - FC(12544→256)→ReLU→Dropout(0.5)→FC(256→1)，输出置信度校正 logit
      4. 训练时 ROI 损失：
         - 正例：GT box ± 20% jitter（标签 1）
         - 负例：背景随机 crop，IoU<0.1（3:1 比例，每图最多 9 个负例）
         - BCEWithLogitsLoss；与检测损失联合反向传播（λ_roi=0.5 可调）
      5. 推理时 ROI 重打分：
         - 对 stage-1 分数 > 0.05 的候选框做 ROI Align → roi_head
         - 最终分数 = √(det_score × roi_score)（几何均值融合）
    目标：突破基线 H 的 F2@0.3=0.2788
    实验结果（2026-05-26，30 epoch，接近早停 ep33）：
      best F2@0.3=0.1571，ep23；
      F2@0.1 峰值 0.4499（ep12），之后持续下滑。
      TP@0.3 序列极端震荡：0,0,4,13,0,5,21,15,16,9,18,32,25,26,16,16,26,8,24,20,17,18,33...
    失败根因分析：
      1. ROI 头训练信号极不稳定：每批正例 ROI 极少（仅 GT box），BCE loss 噪声大，
         导致 roi_score 逐 epoch 大幅抖动，直接传导至 F2@0.3 的极端震荡。
      2. 几何均值公式过激进：√(det×roi) 在 roi_score 较低时把大量候选分数压至 0.3 以下，
         导致 TP@0.3 大量流失（基线 H 平均 TP@0.3≈40，K 仅 22）。
      3. GlobalContextEncoder 效果无法独立评估：无法判断失败是"全局融合本身无效"
         还是"ROI 头破坏了 det_score"。
    结论：方向 J 判定失败；下一步方向 K 单独验证全局融合（去掉 ROI 头）。

51. 方向 K：GlobalContextEncoder + RetinaNet（无 ROI 重打分）（rec_51）
    【此文件未作修改，完整实现在分支文件 bbox-train-K.py 中】
    切换动机：方向 J 失败后，无法确定失败根因是"全局融合无效"还是"ROI 头破坏了 det_score"。
    方向 K 通过移除 ROI 重打分头，单独验证全局上下文融合的效果。
    单变量对比：K vs H（唯一变量：有无 GlobalContextEncoder + GlobalAwareBackbone 融合）。
    关键实现：
      1. GlobalContextEncoder（与方向 J 相同）
         - 将原始图像下采样至 1/4（256×128），经 3 个 stride-2 BN-ReLU conv 块
           得到 128 通道全局特征图 @ 32×16
      2. GlobalAwareBackbone（与方向 J 相同，但无 _cached_features 使用方）
         - 包装 ResNet50-FPN，每个 FPN 层注入全局上下文（concat + 1×1 fusion）
         - identity init（全局路径初始贡献为零），保留预训练特征
      3. 完全移除 RoiRefinementHead 及所有 ROI loss 辅助函数
      4. 推理分数直接使用 det_score（同方向 H），无任何后处理重打分
      5. 差分 LR：model.backbone.base.body（低 LR）vs FPN/头/global_enc/fusion（高 LR）
    实验解读预期：
      K > H（F2@0.3 ≥ 0.28） → 全局融合有效，J 的失败主因是 ROI 头
      K ≈ H                    → 全局融合本身无效，需要换思路
    目标：K F2@0.3 ≥ 0.2788（不低于 H 基线）
    实验结果（rec_51）：best F2@0.3=0.2213 @ ep4，early stop ep16，Val GT=251。
    失败根因：GlobalEncoder 随机初始化，backward 梯度扰动 FPN fusion conv；
    TP 存活率（@0.1→@0.3）从 H 的 45% 降至 30%，分数分布更压缩。
    结论：K ≈ H → 全局融合本身无效，方向 K 失败。

52. 方向 H upd_7：focal gamma 降低（γ=1.0）+ 1024×1024 分辨率（rec_52）
    【实现在 bbox-train-H.py upd_7，非本文件】
    切换动机：对 H 的两个失败模式进行精准干预：
      1. 低置信（H ep9：73 个 @0.1 TP 打分落在 [0.1, 0.3) 区间，存活率 45%）：
         focal_loss_gamma=2.0 过度压制中等置信预测。降低到 γ=1.0，直接针对
         score 校准瓶颈。理论天花板：若全部 @0.1 TP 均能 ≥0.3，F2@0.3 ≈ 0.58。
      2. 检测缺失（118 FN @0.1）：1024×512 下 31% box 在 P3 仅 4~8px（min_side
         P5=35px, P50=86px）。升至 1024×1024（scale 0.5614→0.6737），小病灶
         feature map 表示增大 20%。
    关键变更（相比 H upd_6）：
      - --focal-gamma 1.0（新增参数，默认 2.0 向后兼容）
      - --input-h 1024 --input-w 1024（方形输入）
      - --batch-size 4（4090 24GB 可装下，与 upd_6 有效 batch 相同）
      - [fix] load_samples scale 计算考虑 AR-preserving padding 方向，
              避免 1024×1024 下 min_box_side 阈值计算错误
    单变量原则：相比 H upd_6，同时改了 γ 和分辨率（组合实验）。
    目标：F2@0.3 ≥ 0.35

"""

from __future__ import annotations

import os
# Prevent libgomp warning when OMP_NUM_THREADS is set to "" or "0"
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import atexit
import datetime
import math
import random
import re
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn.functional as F
try:
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
except Exception:  # pragma: no cover
    retinanet_resnet50_fpn_v2 = None  # type: ignore
    RetinaNet_ResNet50_FPN_V2_Weights = None
    RetinaNetClassificationHead = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]

try:
    from torchvision.models import resnet50 as _resnet50, ResNet50_Weights as _ResNet50Weights
except Exception:  # pragma: no cover
    _resnet50 = None  # type: ignore
    _ResNet50Weights = None  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    """Read an image from a path that may contain unicode characters."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8 with 3 channels.

    For grayscale (2-D) inputs, CLAHE contrast enhancement
    (clipLimit=2.0, tileGridSize=8×8) is applied before RGB conversion.
    """
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] > 4:
            img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 image to a float tensor in [0, 1]."""
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _rotate_box_centers_preserve(
    boxes: torch.Tensor,
    angle_deg: float,
    img_h: int,
    img_w: int,
    pivot_x: float,
    pivot_y: float,
) -> torch.Tensor:
    """Rotate bbox centers around a given pivot; keep original box size.

    Each box center is rotated around (pivot_x, pivot_y).  The box is then
    reconstructed with its original width and height centered on the new
    rotated position.  Coordinates are clamped to [0, W] x [0, H].
    """
    if boxes.numel() == 0:
        return boxes.clone()

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    orig_w = boxes[:, 2] - boxes[:, 0]
    orig_h = boxes[:, 3] - boxes[:, 1]
    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    dx = box_cx - pivot_x
    dy = box_cy - pivot_y
    new_cx = cos_a * dx - sin_a * dy + pivot_x
    new_cy = sin_a * dx + cos_a * dy + pivot_y

    new_x1 = (new_cx - orig_w / 2.0).clamp(0.0, float(img_w))
    new_y1 = (new_cy - orig_h / 2.0).clamp(0.0, float(img_h))
    new_x2 = (new_cx + orig_w / 2.0).clamp(0.0, float(img_w))
    new_y2 = (new_cy + orig_h / 2.0).clamp(0.0, float(img_h))

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


def random_augment_fn(
    img: torch.Tensor,
    target: Dict[str, torch.Tensor],
    hflip_prob: float = 0.5,
    brightness_delta: float = 0.2,
    rotation_max_deg: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply random augmentation to a (image tensor, target) pair.

    img: float tensor in [0, 1] of shape [C, H, W].
    target: dict with 'boxes' in xyxy format and other Faster R-CNN keys.

    Augmentations applied:
    - Random horizontal flip (probability = hflip_prob)
    - Random brightness jitter (uniform in [-brightness_delta, +brightness_delta])
    - Random small-angle rotation using strategy-A (keep original image size, fill
      empty corners with 0; update boxes via rotated-corner AABB)
    """
    _, h, w = img.shape

    # Random horizontal flip
    if random.random() < hflip_prob:
        img = torch.flip(img, [2])
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            flipped_boxes = boxes.clone()
            flipped_boxes[:, 0] = float(w) - boxes[:, 2]
            flipped_boxes[:, 2] = float(w) - boxes[:, 0]
            target = {**target, "boxes": flipped_boxes}

    # Random small-angle rotation (strategy A: keep output size, zero-fill corners)
    # Only applied to single-bbox images to avoid pivot ambiguity for multi-lesion cases.
    if rotation_max_deg > 0.0:
        boxes = target.get("boxes")
        n_boxes = boxes.shape[0] if boxes is not None else 0
        if n_boxes == 1:
            angle = random.uniform(-rotation_max_deg, rotation_max_deg)
            if abs(angle) > 0.1:  # skip near-zero rotations for efficiency
                # Rotation pivot = bbox center
                bx1, by1, bx2, by2 = boxes[0].tolist()
                pivot_x = (bx1 + bx2) / 2.0
                pivot_y = (by1 + by2) / 2.0

                # Rotate image tensor via numpy/cv2 (keeps same H x W)
                img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                M = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle, 1.0)
                rotated_np = cv2.warpAffine(
                    img_np, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                img = torch.from_numpy(rotated_np.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

                # Crop away black corners introduced by rotation.
                # detect_breast_region works on non-zero pixels, so the zero-filled
                # rotation corners are naturally excluded.
                crop_box = detect_breast_region(rotated_np, margin_ratio=0.0)
                rotated_np_cropped, _ = crop_image_and_boxes(rotated_np, np.zeros((0, 4), dtype=np.float32), crop_box)
                img = torch.from_numpy(rotated_np_cropped.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                # Remap h, w to the cropped size for subsequent box clamping
                _, h, w = img.shape
                cx1, cy1, _cx2, _cy2 = crop_box

                # Update bbox: rotate center around pivot, keep original size
                rotated_boxes = _rotate_box_centers_preserve(boxes, angle, h + cy1, w + cx1, pivot_x, pivot_y)
                # Remap rotated bbox to crop-local coordinates
                rotated_boxes[:, 0] -= cx1
                rotated_boxes[:, 2] -= cx1
                rotated_boxes[:, 1] -= cy1
                rotated_boxes[:, 3] -= cy1
                rotated_boxes[:, 0] = rotated_boxes[:, 0].clamp(0.0, float(w))
                rotated_boxes[:, 2] = rotated_boxes[:, 2].clamp(0.0, float(w))
                rotated_boxes[:, 1] = rotated_boxes[:, 1].clamp(0.0, float(h))
                rotated_boxes[:, 3] = rotated_boxes[:, 3].clamp(0.0, float(h))
                # Drop degenerate boxes that collapsed after clamp
                keep = (rotated_boxes[:, 2] > rotated_boxes[:, 0] + 1) & (
                    rotated_boxes[:, 3] > rotated_boxes[:, 1] + 1
                )
                rotated_boxes = rotated_boxes[keep]
                target_labels = target.get("labels")
                target_area = target.get("area")
                target_iscrowd = target.get("iscrowd")
                new_target: Dict[str, torch.Tensor] = {
                    **target,
                    "boxes": rotated_boxes,
                }
                if target_labels is not None:
                    new_target["labels"] = target_labels[keep]
                if target_area is not None:
                    area = (rotated_boxes[:, 2] - rotated_boxes[:, 0]) * (
                        rotated_boxes[:, 3] - rotated_boxes[:, 1]
                    )
                    new_target["area"] = area
                if target_iscrowd is not None:
                    new_target["iscrowd"] = target_iscrowd[keep]
                target = new_target

    # Random brightness/contrast jitter
    if brightness_delta > 0.0:
        factor = 1.0 + random.uniform(-brightness_delta, brightness_delta)
        img = torch.clamp(img * factor, 0.0, 1.0)

    return img, target


def detect_breast_region(
    img: np.ndarray,
    margin_ratio: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Detect a breast-region crop on a processed mammogram.

    The processed images already suppress most background, so a largest-contour
    heuristic is sufficient and keeps the detector focused on coarse lesion
    localization inside the breast region.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0, 0, 0, 0
    if np.max(gray) <= 0:
        return 0, 0, w, h

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(gray > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, w, h
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
    else:
        contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(contour)
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + cw)
        y2 = int(y + ch)

    margin_ratio = max(0.0, float(margin_ratio))
    margin_x = int(round((x2 - x1) * margin_ratio))
    margin_y = int(round((y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def crop_image_and_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    crop_box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and remap boxes to the crop-local coordinate system."""
    x1, y1, x2, y2 = crop_box
    cropped_img = img[y1:y2, x1:x2]
    if boxes.size == 0:
        return cropped_img, np.zeros((0, 4), dtype=np.float32)

    cropped_boxes = boxes.astype(np.float32).copy()
    cropped_boxes[:, [0, 2]] -= float(x1)
    cropped_boxes[:, [1, 3]] -= float(y1)

    crop_h, crop_w = cropped_img.shape[:2]
    cropped_boxes[:, 0] = np.clip(cropped_boxes[:, 0], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 2] = np.clip(cropped_boxes[:, 2], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 1] = np.clip(cropped_boxes[:, 1], 0, max(crop_h - 1, 0))
    cropped_boxes[:, 3] = np.clip(cropped_boxes[:, 3], 0, max(crop_h - 1, 0))

    keep = (cropped_boxes[:, 2] > cropped_boxes[:, 0] + 1) & (cropped_boxes[:, 3] > cropped_boxes[:, 1] + 1)
    return cropped_img, cropped_boxes[keep]


@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray
    orig_size: Tuple[float, float]


class VinDrBboxDataset(Dataset):
    """Dataset grouped at the image level.

    Each item contains one image and all lesion boxes found for that image.
    Images with no lesion boxes are kept as negative samples.
    """

    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        split_name: str,
        positive_only: bool = False,
        crop_breast_region: bool = True,
        breast_crop_margin: float = 0.05,
    ) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name
        self.positive_only = positive_only
        self.crop_breast_region = crop_breast_region
        self.breast_crop_margin = breast_crop_margin

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()

        if df.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        self.samples: List[Sample] = []
        grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=True)

        for (patient_id, _series_id, image_id), group in grouped:
            # Keep all boxes for this image (some images have multiple lesions).
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            boxes: np.ndarray
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)

            image_path = images_root / str(patient_id) / f"{image_id}"

            if boxes.size > 0:
                invalid = np.sum((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1]))
                if invalid > 0:
                    print(f"[Warning] Found {invalid} invalid boxes in {image_path}")

            if positive_only and len(boxes) == 0:
                continue

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            self.samples.append(
                Sample(
                    patient_id=str(patient_id),
                    image_id=str(image_id),
                    image_path=image_path,
                    boxes=boxes,
                    orig_size=(orig_h, orig_w),
                )
            )

        if not self.samples:
            raise ValueError(
                f"No image samples could be built from {csv_path} with split={split_name!r}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")

        img = normalize_image(read_image_unicode(sample.image_path))
        h, w = img.shape[:2]

        boxes = sample.boxes.astype(np.float32)

        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
            # 过滤非法框，左值大于右值，或像素值小于 1。
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        if self.crop_breast_region:
            crop_box = detect_breast_region(img, margin_ratio=self.breast_crop_margin)
            img, boxes = crop_image_and_boxes(img, boxes, crop_box)

        if boxes.size == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.from_numpy(boxes)
            labels_tensor = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                (boxes_tensor[:, 2] - boxes_tensor[:, 0]) *
                (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                if boxes_tensor.numel() > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }
        return image_to_tensor(img), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class TrainAugmentWrapper(torch.utils.data.Dataset):
    """Wraps a Dataset/Subset and applies random augmentation during training.

    This allows the same underlying dataset object to be shared between
    train (with augmentation) and val (without augmentation) without creating
    two separate dataset instances.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        hflip_prob: float = 0.5,
        brightness_delta: float = 0.2,
        rotation_max_deg: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.hflip_prob = float(hflip_prob)
        self.brightness_delta = float(brightness_delta)
        self.rotation_max_deg = float(rotation_max_deg)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, target = self.dataset[index]
        img, target = random_augment_fn(
            img, target,
            hflip_prob=self.hflip_prob,
            brightness_delta=self.brightness_delta,
            rotation_max_deg=self.rotation_max_deg,
        )
        return img, target


class CopyPasteWrapper(torch.utils.data.Dataset):
    """Copy-paste augmentation: paste lesion crops from positive samples onto negative images.

    With probability ``paste_prob``, a negative sample is selected and one
    randomly-chosen positive sample's lesion crop is pasted onto it.  All other
    samples are returned unchanged.  The wrapper is applied *after*
    TrainAugmentWrapper so the pasted crops have already been flipped/rotated.

    Args:
        dataset: The underlying dataset (may already be wrapped by TrainAugmentWrapper).
        positive_indices: Indices in *dataset* that correspond to positive images.
            These are sampled when choosing a crop donor.
        paste_prob: Probability that a negative sample is chosen as the paste target.
        max_pastes: Maximum number of crops to paste per target image (chosen uniformly
            from [1, max_pastes]).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        positive_indices: List[int],
        paste_prob: float = 0.4,
        max_pastes: int = 2,
    ) -> None:
        self.dataset = dataset
        self.positive_indices = list(positive_indices)
        self.paste_prob = float(paste_prob)
        self.max_pastes = int(max_pastes)
        if not self.positive_indices:
            print("[Warning] CopyPasteWrapper: no positive indices supplied; augmentation disabled.")

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, target = self.dataset[index]

        # Only augment negative images (no GT boxes) and only with paste_prob probability.
        if (
            not self.positive_indices
            or target["boxes"].shape[0] > 0
            or random.random() > self.paste_prob
        ):
            return img, target

        # img: [C, H, W] float32 tensor
        _, H, W = img.shape
        img = img.clone()
        new_boxes: List[torch.Tensor] = []

        # Compute tissue bounding box on the target image.
        # Pixels with mean value > 0.05 across channels are considered tissue.
        # Paste positions are restricted to this bounding box to avoid placing
        # lesion crops onto black background regions (would teach wrong prior).
        mean_img = img.mean(dim=0)  # [H, W]
        tissue_mask = mean_img > 0.05
        tissue_rows = tissue_mask.any(dim=1).nonzero(as_tuple=False)
        tissue_cols = tissue_mask.any(dim=0).nonzero(as_tuple=False)
        if tissue_rows.numel() == 0 or tissue_cols.numel() == 0:
            # Fully black image; fall back to full image bounds.
            t_y1, t_x1, t_y2, t_x2 = 0, 0, H, W
        else:
            t_y1 = int(tissue_rows[0].item())
            t_y2 = int(tissue_rows[-1].item()) + 1
            t_x1 = int(tissue_cols[0].item())
            t_x2 = int(tissue_cols[-1].item()) + 1

        n_paste = random.randint(1, self.max_pastes)
        for _ in range(n_paste):
            donor_idx = random.choice(self.positive_indices)
            donor_img, donor_target = self.dataset[donor_idx]
            if donor_target["boxes"].shape[0] == 0:
                continue
            # Pick one random box from the donor.
            bi = random.randrange(donor_target["boxes"].shape[0])
            x1, y1, x2, y2 = donor_target["boxes"][bi].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            # Clamp crop to donor image bounds.
            _, dH, dW = donor_img.shape
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, dW), min(y2, dH)
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = donor_img[:, y1c:y2c, x1c:x2c]  # [C, ch, cw]
            cH, cW = crop.shape[1], crop.shape[2]

            # Build a lesion mask for the crop via largest-connected-component.
            # This removes the rectangular frame of black/near-black pixels
            # surrounding the actual tissue, so only genuine lesion pixels are
            # pasted (no hard rectangular boundary visible in the result).
            crop_gray = (crop.mean(dim=0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Light morphological dilation to avoid over-eroding lesion edges.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            n_labels, labels_map = cv2.connectedComponents(thresh, connectivity=8)
            if n_labels > 1:
                # Label 0 is background; find the largest foreground label.
                label_sizes = np.bincount(labels_map.ravel())
                label_sizes[0] = 0  # exclude background
                largest_label = int(label_sizes.argmax())
                lesion_mask = (labels_map == largest_label).astype(np.uint8)  # HxW uint8
            else:
                # Fallback: use all non-zero pixels.
                lesion_mask = (thresh > 0).astype(np.uint8)
            if lesion_mask.sum() == 0:
                continue

            # Feathered alpha: distance transform gives each mask pixel its
            # distance to the nearest edge, clamped to [0, feather_width].
            # This creates a smooth gradient at the lesion boundary so there
            # is no hard cut and no residual black fringe.
            _feather_w = 5  # pixels; wider = softer edge
            dist_map = cv2.distanceTransform(lesion_mask, cv2.DIST_L2, 3)
            alpha_map = np.clip(dist_map / float(_feather_w), 0.0, 1.0).astype(np.float32)  # [cH,cW]
            alpha_t = torch.from_numpy(alpha_map).unsqueeze(0)  # [1, cH, cW]

            # Paste within the tissue bounding box only.
            avail_w = (t_x2 - t_x1) - cW
            avail_h = (t_y2 - t_y1) - cH
            if avail_w < 0 or avail_h < 0:
                # Tissue region smaller than crop; skip.
                continue
            # Try up to 5 random positions; accept only if ≥70% of the paste
            # region contains tissue (mean pixel > 0.05), preventing lesion
            # crops from being placed on black background areas outside the
            # breast contour (the tissue bbox is a rectangle, not the actual shape).
            placed = False
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

            # Step A: Brightness alignment — scale crop P95 to match target region P95
            # (mask-weighted pixels only).  Using the 95th-percentile focuses on the
            # brightest tissue pixels and avoids pulling the estimate down via background.
            target_region = img[:, py:py + cH, px:px + cW]  # [C, cH, cW]
            _lesion_px_crop   = crop[:, lesion_mask.astype(bool)].reshape(-1)   # 1-D
            _lesion_px_target = target_region[:, lesion_mask.astype(bool)].reshape(-1)
            crop_p95   = float(torch.quantile(_lesion_px_crop,   0.95)) if _lesion_px_crop.numel() > 0 else 1e-6
            target_p95 = float(torch.quantile(_lesion_px_target, 0.95)) if _lesion_px_target.numel() > 0 else 1e-6
            brightness_scale = float(np.clip(target_p95 / max(crop_p95, 1e-6), 0.5, 2.0))
            crop_adj = (crop * brightness_scale).clamp(0.0, 1.0)

            # Step B: Center darkening — gaussian-shaped gamma map applied to the
            # brightness-adjusted crop. Center gamma < 1 (darken), edge gamma → 1.
            # Gaussian sigma covers ~half the crop so the effect fades to edges.
            # gamma_center = 0.75 means the brightest center pixel becomes pixel^0.75.
            _gamma_center = 0.75
            cy, cx = cH / 2.0, cW / 2.0
            sigma = max(cy, cx) * 0.5
            ys = torch.arange(cH, dtype=torch.float32)
            xs = torch.arange(cW, dtype=torch.float32)
            dist2 = ((ys.unsqueeze(1) - cy) ** 2 + (xs.unsqueeze(0) - cx) ** 2)
            gaussian = torch.exp(-dist2 / (2 * sigma ** 2))  # [cH, cW], max=1 at center
            # gamma_map: center → _gamma_center, edges → 1.0
            gamma_map = 1.0 - gaussian * (1.0 - _gamma_center)  # [cH, cW]
            gamma_map = gamma_map.unsqueeze(0)  # [1, cH, cW]
            # Apply gamma: pixel^gamma_map (only on alpha-weighted region to avoid edge artefacts)
            crop_adj = torch.pow(crop_adj.clamp(1e-6, 1.0), gamma_map)

            # Step C: Feathered paste: result = crop_adj * alpha + background * (1 - alpha)
            # Guard against overlap: check IoU of candidate box with already-pasted boxes.
            # If IoU > 0.3 with any existing box, skip this paste to avoid pixel corruption
            # and duplicate/overlapping supervision signals.
            candidate_box = torch.tensor(
                [float(px), float(py), float(px + cW), float(py + cH)]
            )
            overlap = False
            for prev_box in new_boxes:
                ix1 = max(candidate_box[0], prev_box[0])
                iy1 = max(candidate_box[1], prev_box[1])
                ix2 = min(candidate_box[2], prev_box[2])
                iy2 = min(candidate_box[3], prev_box[3])
                inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
                if inter > 0:
                    area_a = float((candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1]))
                    area_b = float((prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1]))
                    iou = inter / max(area_a + area_b - inter, 1e-6)
                    if iou > 0.3:
                        overlap = True
                        break
            if overlap:
                continue

            blended = crop_adj * alpha_t + target_region * (1.0 - alpha_t)
            img[:, py:py + cH, px:px + cW] = blended
            new_boxes.append(candidate_box)

        if new_boxes:
            new_boxes_t = torch.stack(new_boxes, dim=0)  # [N, 4]
            new_labels = torch.ones((new_boxes_t.shape[0],), dtype=torch.int64)
            new_area = (new_boxes_t[:, 2] - new_boxes_t[:, 0]) * (new_boxes_t[:, 3] - new_boxes_t[:, 1])
            new_crowd = torch.zeros((new_boxes_t.shape[0],), dtype=torch.int64)
            target = {
                "boxes": new_boxes_t,
                "labels": new_labels,
                "image_id": target["image_id"],
                "area": new_area,
                "iscrowd": new_crowd,
            }

        return img, target


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 helpers: image-level binary classifier for training-set pre-filtering
# ─────────────────────────────────────────────────────────────────────────────

class ImageClassificationDataset(torch.utils.data.Dataset):
    """Wraps VinDrBboxDataset to return (image_tensor, has_lesion_label) pairs.

    Used for Stage-1 image-level binary classifier training.
    The underlying dataset already handles breast-crop and CLAHE preprocessing.
    """

    def __init__(self, base_dataset: torch.utils.data.Dataset, indices: List[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img_tensor, target = self.base_dataset[real_idx]
        # Resize to a fixed smaller size so images can be stacked into a batch.
        # Full resolution (1520×912) causes CUDA OOM in ResNet50 layer2 due to large
        # intermediate feature maps.  (512, 320) is 1/3 scale and sufficient for
        # image-level binary classification; ResNet50 uses GlobalAveragePooling so
        # spatial size does not affect model accuracy.
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), size=(512, 320), mode="bilinear", align_corners=False
        ).squeeze(0)
        label = 1.0 if target["boxes"].shape[0] > 0 else 0.0
        return img_tensor, torch.tensor(label, dtype=torch.float32)


def build_image_classifier(medical_backbone_path: Optional[str] = None) -> torch.nn.Module:
    """Build a ResNet50 image-level binary classifier (has_lesion / no_lesion).

    Architecture: ResNet50 (ImageNet pretrained) + AdaptiveAvgPool2d (inside ResNet50) +
    Dropout(0.5) + FC(2048→1).  conv1 is adapted for grayscale-as-3channel input.
    Optionally overwrites backbone body with a medical-domain pretrained checkpoint.
    """
    if _resnet50 is None:
        raise RuntimeError("torchvision resnet50 not found; please upgrade torchvision.")
    try:
        backbone = _resnet50(weights=_ResNet50Weights.DEFAULT)
    except Exception:
        backbone = _resnet50(pretrained=True)  # type: ignore[call-arg]

    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features, 1),
    )

    if medical_backbone_path is not None:
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
            _idx_to_resnet = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            missing, unexpected = backbone.load_state_dict(stripped, strict=False)
            print(f"[Stage-1] Loaded medical backbone: {medical_backbone_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as exc:
            print(f"[Stage-1 Warning] Could not load medical backbone ({exc}). Using ImageNet weights.")

    try:
        with torch.no_grad():
            mean_w = backbone.conv1.weight.mean(dim=1, keepdim=True)
            backbone.conv1.weight.copy_(mean_w.expand_as(backbone.conv1.weight))
        print("[Stage-1] conv1 weights averaged for grayscale-as-3channel input.")
    except AttributeError:
        pass

    return backbone


def train_clf_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    pos_weight: float = 10.0,
    disable_tqdm: bool = False,
) -> float:
    """Train one epoch of the image-level binary classifier (BCEWithLogitsLoss)."""
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    running_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"clf {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item())
        count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(count, 1)


def run_clf_filter(
    classifier: torch.nn.Module,
    base_dataset: torch.utils.data.Dataset,
    train_indices: List[int],
    samples: List[Sample],
    device: torch.device,
    threshold: float = 0.5,
) -> List[int]:
    """Apply image-level classifier to filter negative training images.

    All original positive images are kept unconditionally to prevent false
    negatives from removing lesion-containing images.  Negative images are
    kept only when classifier sigmoid output > threshold.

    Returns the filtered (and sorted) list of train_indices.
    """
    classifier.eval()
    orig_positives = {i for i in train_indices if samples[i].boxes.size > 0}
    neg_indices = [i for i in train_indices if i not in orig_positives]
    print(f"[Stage-1 Filter] Classifying {len(neg_indices)} negative training images (threshold={threshold})…")
    clf_keep = set(orig_positives)
    with torch.no_grad():
        for real_idx in neg_indices:
            img_tensor, _ = base_dataset[real_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            logit = classifier(img_tensor).squeeze()
            prob = float(torch.sigmoid(logit).item())
            if prob > threshold:
                clf_keep.add(real_idx)
    filtered = sorted(clf_keep)
    n_pos = sum(1 for i in filtered if samples[i].boxes.size > 0)
    n_neg = len(filtered) - n_pos
    print(
        f"[Stage-1 Filter] Filtered train set: {len(filtered)} images "
        f"(pos={n_pos}, neg={n_neg}, neg/pos={n_neg / max(n_pos, 1):.2f}; "
        f"removed {len(train_indices) - len(filtered)} negative images)"
    )
    return filtered


def summarize_subset(samples: List[Sample], indices: List[int]) -> Dict[str, Any]:
    """Summarize a dataset subset without changing its distribution."""
    if not indices:
        return {
            "images": 0,
            "patients": 0,
            "positive_images": 0,
            "negative_images": 0,
            "positive_ratio": 0.0,
        }

    patient_ids = {samples[i].patient_id for i in indices}
    positive_images = sum(1 for i in indices if samples[i].boxes.size > 0)
    negative_images = len(indices) - positive_images
    positive_ratio = float(positive_images / max(len(indices), 1))

    return {
        "images": int(len(indices)),
        "patients": int(len(patient_ids)),
        "positive_images": int(positive_images),
        "negative_images": int(negative_images),
        "positive_ratio": float(positive_ratio),
    }


def compute_neg_pos_ratio(summary: Dict[str, Any]) -> float:
    """Return negative-to-positive image ratio for a summarized subset."""
    positive_images = int(summary.get("positive_images", 0))
    negative_images = int(summary.get("negative_images", 0))
    if positive_images <= 0:
        return float("inf") if negative_images > 0 else 0.0
    return float(negative_images / positive_images)


def warn_on_small_epoch_positive_pool(train_summary: Dict[str, Any], only_use: float) -> None:
    """Warn when epoch subsampling leaves too few positive images for stable training."""
    if only_use >= 1.0:
        return

    positive_images = int(train_summary.get("positive_images", 0))
    estimated_positive_images = math.ceil(positive_images * only_use)
    if 0 < estimated_positive_images < 256:
        print(
            f"[Warning] --only-use={only_use:.3f} leaves about {estimated_positive_images} positive images per epoch "
            f"under the current train split. This often weakens lesion learning and can trigger FP rebound; "
            f"prefer --only-use 1.0 for final detector training."
        )


def select_epoch_subset(
    train_indices: List[int],
    samples: List[Sample],
    epoch: int,
    only_use: float,
    seed: int,
) -> List[int]:
    """Select a rotating subset of training indices for one epoch.

    Ensures all images are visited across epochs by cycling through
    positive and negative samples independently.  Positive samples are
    protected to never fall below the original positive ratio of the subset.
    """
    if only_use >= 1.0:
        return train_indices

    pos_all = [i for i in train_indices if samples[i].boxes.size > 0]
    neg_all = [i for i in train_indices if samples[i].boxes.size == 0]

    total_target = max(1, math.ceil(len(train_indices) * only_use))
    original_pos_ratio = len(pos_all) / max(len(train_indices), 1)

    # Protect positive samples: at least the original ratio worth of positives
    n_pos = max(1, math.ceil(total_target * original_pos_ratio)) if pos_all else 0
    n_pos = min(n_pos, len(pos_all))
    n_neg = min(total_target - n_pos, len(neg_all))
    n_neg = max(0, n_neg)

    selected: List[int] = []

    if pos_all and n_pos > 0:
        pos_cycles = max(1, math.ceil(len(pos_all) / n_pos))
        pos_cycle_group = epoch // pos_cycles
        pos_epoch_in_cycle = epoch % pos_cycles
        rng_pos = random.Random(seed + 500 + pos_cycle_group * 31)
        pos_shuffled = pos_all.copy()
        rng_pos.shuffle(pos_shuffled)
        start_p = pos_epoch_in_cycle * n_pos
        selected += [pos_shuffled[j % len(pos_all)] for j in range(start_p, start_p + n_pos)]

    if neg_all and n_neg > 0:
        neg_cycles = max(1, math.ceil(len(neg_all) / n_neg))
        neg_cycle_group = epoch // neg_cycles
        neg_epoch_in_cycle = epoch % neg_cycles
        rng_neg = random.Random(seed + 700 + neg_cycle_group * 37)
        neg_shuffled = neg_all.copy()
        rng_neg.shuffle(neg_shuffled)
        start_n = neg_epoch_in_cycle * n_neg
        selected += [neg_shuffled[j % len(neg_all)] for j in range(start_n, start_n + n_neg)]

    return selected


def split_train_val_by_patient(
    samples: List[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Split training data into train/val at the patient level.

    The split is patient-level to prevent leakage, and it tries to keep the
    original positive/negative distribution approximately stable by selecting
    roughly the same proportion of positive-patient and negative-patient images.
    """
    usable_indices: List[int] = list(range(len(samples)))

    patient_to_records: Dict[str, Dict[str, Any]] = {}
    for idx in usable_indices:
        sample = samples[idx]
        record = patient_to_records.setdefault(
            sample.patient_id,
            {
                "patient_id": sample.patient_id,
                "indices": [],
                "num_images": 0,
                "pos_images": 0,
                "neg_images": 0,
            },
        )
        record["indices"].append(idx)
        record["num_images"] += 1
        if sample.boxes.size > 0:
            record["pos_images"] += 1
        else:
            record["neg_images"] += 1

    records = list(patient_to_records.values())
    if not records:
        raise ValueError("No usable samples remain after filtering bad data.")

    positive_records = [r for r in records if r["pos_images"] > 0]
    negative_records = [r for r in records if r["pos_images"] == 0]

    def choose_val_patients(group_records: List[Dict[str, Any]], ratio: float, seed_offset: int) -> set[str]:
        if not group_records:
            return set()

        group_total_images = sum(r["num_images"] for r in group_records)
        target_images = int(round(group_total_images * ratio))
        if target_images <= 0:
            return set()

        rng = random.Random(seed + seed_offset)
        remaining = group_records.copy()
        rng.shuffle(remaining)

        selected: List[Dict[str, Any]] = []
        current = 0

        while remaining and current < target_images:
            current_diff = abs(current - target_images)
            best_idx = None
            best_key = None

            for i, rec in enumerate(remaining):
                new_current = current + rec["num_images"]
                key = (abs(new_current - target_images), -rec["num_images"])
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = i

            if best_idx is None:
                break

            candidate = remaining[best_idx]
            new_diff = abs((current + candidate["num_images"]) - target_images)

            # Accept the candidate if it improves the target distance,
            # or if we still have very little validation data selected.
            if (not selected) or (new_diff <= current_diff) or (current < target_images * 0.85):
                selected.append(candidate)
                current += candidate["num_images"]
                remaining.pop(best_idx)
            else:
                break

        if not selected:
            selected = [max(group_records, key=lambda r: r["num_images"])]

        return {r["patient_id"] for r in selected}

    val_patient_ids = set()
    val_patient_ids |= choose_val_patients(positive_records, val_ratio, 101)
    val_patient_ids |= choose_val_patients(negative_records, val_ratio, 202)

    train_indices = [idx for idx in usable_indices if samples[idx].patient_id not in val_patient_ids]
    val_indices = [idx for idx in usable_indices if samples[idx].patient_id in val_patient_ids]

    # Fallback: if the patient-level split is empty on one side, keep training usable.
    if not train_indices and usable_indices:
        print("[Warning] Patient-level split produced an empty training split; falling back to using all usable samples for training.")
        train_indices = usable_indices.copy()
        val_indices = []

    if not val_indices and usable_indices:
        print("[Warning] Patient-level split produced an empty validation split; moving one whole patient to validation.")
        fallback_patient = max(records, key=lambda r: r["num_images"])
        val_patient_ids = {fallback_patient["patient_id"]}
        train_indices = [idx for idx in usable_indices if samples[idx].patient_id not in val_patient_ids]
        val_indices = [idx for idx in usable_indices if samples[idx].patient_id in val_patient_ids]

    train_indices.sort()
    val_indices.sort()

    summary = {
        "val_ratio": float(val_ratio),
        "usable_images": int(len(usable_indices)),
        "usable_patients": int(len(records)),
        "train": summarize_subset(samples, train_indices),
        "val": summarize_subset(samples, val_indices),
        "train_patients": int(len({samples[i].patient_id for i in train_indices})),
        "val_patients": int(len({samples[i].patient_id for i in val_indices})),
    }
    return train_indices, val_indices, summary


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix for two box sets in xyxy format."""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)

    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], a_min=0.0, a_max=None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], a_min=0.0, a_max=None
    )

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Greedy one-to-one matching to compute TP / FP / FN for one image."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0

    keep = pred_scores >= float(score_threshold)
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    ious = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = np.zeros((gt_boxes.shape[0],), dtype=bool)

    tp = 0
    for pred_idx in range(pred_boxes.shape[0]):
        unmatched = np.where(~matched_gt)[0]
        if unmatched.size == 0:
            break

        best_rel = int(unmatched[int(np.argmax(ious[pred_idx, unmatched]))])
        best_iou = float(ious[pred_idx, best_rel])

        if best_iou >= float(iou_threshold):
            matched_gt[best_rel] = True
            tp += 1

    fp = int(pred_boxes.shape[0] - tp)
    fn = int(gt_boxes.shape[0] - tp)
    return tp, fp, fn


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    epoch: int,
    epochs: int,
    disable_tqdm: bool = False,
    val_detections_per_img: Optional[int] = None,
) -> Dict[str, float]:
    """Validate one epoch without gradient computation."""
    model.eval()
    # Optionally override detections_per_img so evaluation is not truncated at the
    # training inference cap (e.g. 20 boxes/image).  When val_detections_per_img is set
    # (e.g. 300), the model can return up to that many boxes per image, allowing TP@0.1
    # to be measured without the hard ceiling that distorts recall statistics and
    # early-stopping.  When None, no override is applied (backward-compatible).
    _orig_det_per_img = getattr(model, "detections_per_img", None)
    if val_detections_per_img is not None and _orig_det_per_img is not None and _orig_det_per_img != val_detections_per_img:
        model.detections_per_img = val_detections_per_img

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_raw_preds = 0

    # Multi-threshold tracking
    multi_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    multi_stats: Dict[float, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in multi_thresholds}

    pbar = tqdm(loader, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                total_images += 1

                pred_boxes = output.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu().numpy()
                pred_scores = output.get("scores", torch.zeros((0,), device=device)).detach().cpu().numpy()
                gt_boxes = target["boxes"].detach().cpu().numpy()

                total_gt_boxes += int(gt_boxes.shape[0])
                total_raw_preds += int(pred_boxes.shape[0])

                # Apply the same score threshold used during validation statistics.
                keep = pred_scores >= float(score_threshold)
                total_pred_boxes += int(np.sum(keep))

                tp, fp, fn = match_predictions_to_gt(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Multi-threshold evaluation
                for t in multi_thresholds:
                    tp_t, fp_t, fn_t = match_predictions_to_gt(
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        gt_boxes=gt_boxes,
                        score_threshold=t,
                        iou_threshold=iou_threshold,
                    )
                    multi_stats[t]["tp"] += tp_t
                    multi_stats[t]["fp"] += fp_t
                    multi_stats[t]["fn"] += fn_t

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_tp + total_fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))

    avg_raw_preds = float(total_raw_preds / max(total_images, 1))
    if avg_raw_preds > 200:
        print(f"[Warning] avg_raw_preds_per_image={avg_raw_preds:.1f} > 200, RetinaNet may not be suppressing background anchors effectively")

    # Print multi-threshold summary and compute best-threshold F1
    parts = []
    best_thresh_f1 = -1.0
    best_thresh = float(score_threshold)
    for t in multi_thresholds:
        tp_t = multi_stats[t]["tp"]
        fp_t = multi_stats[t]["fp"]
        fn_t = multi_stats[t]["fn"]
        p_t = float(tp_t / max(tp_t + fp_t, 1))
        r_t = float(tp_t / max(tp_t + fn_t, 1))
        f1_t = float(2.0 * p_t * r_t / max(p_t + r_t, 1e-12))
        parts.append(f"Val@{t}: TP={tp_t}, FP={fp_t} F1={f1_t:.4f}")
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = t
    print(f"  {' | '.join(parts)}")
    print(f"  [BestThresh] F1={best_thresh_f1:.4f} @ threshold={best_thresh}")

    result: Dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_thresh_f1": best_thresh_f1,
        "best_thresh": best_thresh,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "images": float(total_images),
        "gt_boxes": float(total_gt_boxes),
        "pred_boxes": float(total_pred_boxes),
        "raw_preds": float(total_raw_preds),
        "avg_raw_preds_per_image": avg_raw_preds,
    }
    for t in multi_thresholds:
        result[f"tp@{t}"] = float(multi_stats[t]["tp"])
        result[f"fp@{t}"] = float(multi_stats[t]["fp"])
        result[f"fn@{t}"] = float(multi_stats[t]["fn"])
    # Restore original detections_per_img so training inference is unaffected.
    if val_detections_per_img is not None and _orig_det_per_img is not None and _orig_det_per_img != val_detections_per_img:
        model.detections_per_img = _orig_det_per_img
    return result


def build_model(
    num_classes: int = 2,
    anchor_sizes: List[Tuple[int, ...]] | None = None,
    fg_iou_thresh: float = 0.5,
    bg_iou_thresh: float = 0.4,
    score_thresh: float = 0.05,
    detections_per_img: int = 100,
    nms_thresh: float = 0.5,
    focal_loss_alpha: float = 0.75,
    focal_loss_gamma: float = 2.0,
    medical_backbone_path: Optional[str] = None,
    input_min_size: int = 800,
) -> torch.nn.Module:
    """Build RetinaNet (retinanet_resnet50_fpn_v2) with ResNet50-FPN backbone.

    Steps:
      1. Fetch full COCO-pretrained state_dict; strip cls_logits keys (shape
         mismatch: 91-class vs num_classes).
      2. Build model with weights=None, num_classes=num_classes (default
         9 anchors/location when anchor_sizes is None, so COCO head loads cleanly).
      3. Load stripped COCO weights strict=False (backbone + FPN + regression head).
      4. Optionally overwrite backbone.body with RadImageNet weights
         (medical_backbone_path).
      5. Average conv1 channel weights for grayscale-as-3channel input.

    Args:
        anchor_sizes: Per-FPN-level size tuples.  ``None`` keeps torchvision
            default (3 scales x 3 aspects = 9 anchors/location), required for
            COCO head weight compatibility.
        fg_iou_thresh: Anchor IoU threshold to be assigned as positive.
        bg_iou_thresh: Anchor IoU threshold to be assigned as negative.
        score_thresh: Model-internal score threshold for post-NMS filtering.
        detections_per_img: Max detections kept per image after NMS.
        nms_thresh: IoU threshold for NMS.
        focal_loss_alpha: Positive class weight in Focal Loss.
        focal_loss_gamma: Focusing exponent in Focal Loss.
        medical_backbone_path: Path to RadImageNet ResNet50 checkpoint; if
            None, COCO backbone weights are kept.
        input_min_size: Shorter side of the image after RetinaNet's built-in
            resize transform.  Default 800 matches torchvision default.  Raise
            to 1200 to keep small lesions larger relative to anchors.
    """
    if retinanet_resnet50_fpn_v2 is None:
        raise RuntimeError(
            "torchvision RetinaNet v2 not found. Please upgrade torchvision (>=0.14)."
        )

    # When anchor_sizes is None, we let torchvision use its default AnchorGenerator
    # (3 scales × 3 aspect ratios = 9 anchors/location).  This is required for the
    # COCO-pretrained head to load cleanly (its output channels match 9 anchors).
    # Only build a custom AnchorGenerator when the caller explicitly passes sizes.
    if anchor_sizes is None:
        anchor_generator = None  # will be left as-is after factory construction
    else:
        aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))])
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Step 1: Load full COCO-pretrained RetinaNet (backbone + FPN + head).
    # We keep COCO head weights intact so the detection prior is preserved.
    # num_classes=91 is the COCO default; we will NOT rebuild the head here.
    # The anchor_sizes argument is ignored at this step — we patch it below.
    coco_state_dict = None
    if RetinaNet_ResNet50_FPN_V2_Weights is not None:
        try:
            _pretrained = retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            )
            coco_state_dict = _pretrained.state_dict()
            del _pretrained
        except Exception as exc:
            print(f"[Warning] Could not load RetinaNet COCO weights: {exc}")

    # Step 2: Build model with 2 classes and no weights.
    # Note: do NOT pass anchor_generator to retinanet_resnet50_fpn_v2() —
    # current torchvision passes it positionally inside the factory function,
    # so passing it again as a kwarg raises "got multiple values" TypeError.
    # We replace model.anchor_generator after construction instead.
    try:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
        )
    except TypeError:
        # Very old torchvision: retry with minimal args, then patch below.
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        try:
            model.score_thresh = score_thresh
            model.nms_thresh = nms_thresh
            model.detections_per_img = detections_per_img
            model.fg_iou_thresh = fg_iou_thresh
            model.bg_iou_thresh = bg_iou_thresh
        except AttributeError:
            print("[Warning] Could not patch score_thresh/detections_per_img on old torchvision model.")
    # Replace anchor generator only when caller explicitly passed custom sizes.
    # When anchor_generator is None, the torchvision default (9 anchors/location) is kept,
    # which allows the COCO-pretrained head to load with matching dimensions.
    if anchor_generator is not None:
        model.anchor_generator = anchor_generator
    # Override the built-in resize transform min_size so that small lesions
    # stay larger after rescaling and better match the smallest anchors.
    if input_min_size != 800:
        model.transform.min_size = (input_min_size,)
        model.transform.max_size = int(input_min_size * 1333 / 800)
    print(
        f"[Info] Model params: score_thresh={model.score_thresh}, "
        f"nms_thresh={model.nms_thresh}, "
        f"detections_per_img={model.detections_per_img}, "
        f"input_min_size={model.transform.min_size[0]}"
    )

    # Load COCO weights into the 2-class model with strict=False.
    # backbone + FPN + head regression branch will load cleanly.
    # head classification branch (91-class → 2-class) has a shape mismatch:
    # strict=False cannot skip shape mismatches (only missing/unexpected keys).
    # So we manually remove cls_logits keys from coco_state_dict first,
    # then load the rest — classification head stays randomly initialised.
    if coco_state_dict is not None:
        cls_logits_keys = [k for k in coco_state_dict if "cls_logits" in k]
        for k in cls_logits_keys:
            del coco_state_dict[k]
        missing, unexpected = model.load_state_dict(coco_state_dict, strict=False)
        print(
            f"[Info] Loaded COCO weights (strict=False): "
            f"{len(cls_logits_keys)} cls_logits keys removed (shape mismatch, expected), "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    else:
        print("[Warning] No COCO weights loaded; model starts from random initialisation.")


    # Step 3: Overwrite backbone.body with RadImageNet weights (if provided).
    # COCO FPN + head weights loaded in Step 2 are NOT touched here.
    # Only backbone.body (ResNet50 stem + layers) is overwritten.
    if medical_backbone_path is not None:
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            # Accept checkpoint files that store weights under common keys
            if isinstance(ckpt, dict):
                raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                raw_sd = ckpt
            # Strip common prefixes: module., encoder., backbone., body.
            # Then remap RadImageNet-style numeric Sequential indices to
            # torchvision ResNet named layers:
            #   0 -> conv1,  1 -> bn1,  4 -> layer1,  5 -> layer2,
            #   6 -> layer3, 7 -> layer4
            _idx_to_resnet = {
                "0": "conv1", "1": "bn1",
                "4": "layer1", "5": "layer2",
                "6": "layer3", "7": "layer4",
            }
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            missing, unexpected = model.backbone.body.load_state_dict(stripped, strict=False)
            print(f"[Info] Loaded medical backbone from: {medical_backbone_path}")
            if missing:
                print(f"[Info]   Missing keys ({len(missing)}): {missing[:3]}")
            if unexpected:
                print(f"[Info]   Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        except Exception as exc:
            print(f"[Warning] Could not load medical backbone ({exc}). COCO backbone is retained.")

    # Step 3.5: Adapt conv1 for grayscale-as-3channel input.
    # Mammogram images are grayscale loaded as 3 identical channels (R=G=B). ImageNet's
    # conv1 has 3 distinct channel weights optimized for RGB. Averaging them ensures all
    # three input channels start with the same feature detector, which is the correct
    # inductive bias for our single-modality input.
    try:
        conv1 = model.backbone.body.conv1
        with torch.no_grad():
            mean_w = conv1.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            conv1.weight.copy_(mean_w.expand_as(conv1.weight))
        print("[Info] conv1 weights averaged across channels for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1 (unexpected backbone structure).")

    # Step 4: Set Focal Loss parameters on the classification head.
    try:
        model.head.classification_head.focal_loss_alpha = focal_loss_alpha
        model.head.classification_head.focal_loss_gamma = focal_loss_gamma
        print(f"[Info] RetinaNet Focal Loss: alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")
    except AttributeError:
        print("[Warning] Could not set focal_loss_alpha/gamma on classification head; using defaults.")

    return model


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace, base_lr: Optional[float] = None) -> torch.optim.Optimizer:
    """Create SGD optimizer with separate weight decay for biases/BN and others."""
    decay = float(args.weight_decay)
    base_lr = float(base_lr) if base_lr is not None else float(args.lr)

    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith(".bias") or "bn" in lname or "norm" in lname:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    groups = []
    if params_with_decay:
        groups.append({"params": params_with_decay, "weight_decay": decay})
    if params_without_decay:
        groups.append({"params": params_without_decay, "weight_decay": 0.0})

    opt = torch.optim.SGD(groups, lr=base_lr, momentum=float(args.momentum))
    return opt


def freeze_backbone_layers(model: torch.nn.Module) -> None:
    """Freeze ResNet backbone layer1 and layer2 parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = False


def unfreeze_backbone_layers(model: torch.nn.Module) -> None:
    """Unfreeze previously frozen ResNet backbone layers."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = True


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    disable_tqdm: bool = False,
    classification_loss_scale: float = 1.0,
) -> Tuple[float, int, Dict[str, float]]:
    """Train one epoch with gradient accumulation and optional iter-level LinearLR warmup.

    Returns:
        avg_loss: average loss for the epoch
        optimizer_steps: number of times `optimizer.step()` was actually called
    """
    model.train()
    running_loss = 0.0
    count = 0
    optimizer_steps = 0
    bad_keys_count = 0
    # track RetinaNet sub-losses
    subloss_keys = ("classification", "bbox_regression")
    subloss_sums: Dict[str, float] = defaultdict(float)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        bad_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
        if bad_keys:
            bad_keys_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # Apply classification loss scale to strengthen lesion vs background discrimination.
        # bbox_regression loss keeps weight 1.0.
        if classification_loss_scale != 1.0 and "classification" in loss_dict:
            loss_dict = dict(loss_dict)
            loss_dict["classification"] = loss_dict["classification"] * classification_loss_scale
        loss = sum(loss for loss in loss_dict.values())

        # accumulate named sub-losses for reporting
        for k in subloss_keys:
            v = loss_dict.get(k)
            if v is not None and torch.isfinite(v):
                try:
                    subloss_sums[k] += float(v.item())
                except Exception:
                    # fallback if value cannot be .item()'d
                    subloss_sums[k] += float(v)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        # gradient accumulation (scale before backward)
        scaled_loss = loss / float(accumulation_steps)
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            # Only advance the warmup scheduler at the same time as optimizer.step()
            if warmup_scheduler is not None:
                warmup_scheduler.step()

        batch_loss = float(loss.item())
        running_loss += batch_loss
        count += 1
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    print(f"[Sum] count = {count}")
    print(f"[Sum] bad data count = {bad_keys_count}")

    avg_sublosses: Dict[str, float] = {k: (subloss_sums[k] / max(count, 1)) for k in subloss_keys}
    return running_loss / max(count, 1), optimizer_steps, avg_sublosses


def save_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    meta: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(payload, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bbox detector for VinDr.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to vindr_detection_folds.csv",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Root folder containing processed images_png/<patient_id>/<image_id>",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Best checkpoint path (default: models/bbox_resnet50.pth)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-score-threshold", type=float, default=0.5)
    parser.add_argument("--val-iou-threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.005, help="Base learning rate (after warmup)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Drop negative images (images without any bbox).",
    )
    # Gradient accumulation vs direct large-batch mode
    parser.add_argument("--fuck-running", action="store_true", help="When set use provided batch-size directly; when absent use gradient accumulation")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps when not using --fuck-running")

    # Freeze / unfreeze backbone
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Number of epochs to freeze backbone layer1/2 before unfreezing (0 = no freeze)")

    # Anchor tuning
    parser.add_argument("--anchor-sizes", type=str, default="16,32,64,128,256", help="Comma-separated anchor sizes (one per FPN level)")

    # LR scheduling
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-step-size", type=int, default=0, help="StepLR step size; 0 to use CosineAnnealingLR")

    # IoU thresholds for anchor assignment (RetinaNet matcher)
    parser.add_argument("--box-fg-iou-thresh", type=float, default=0.5, help="Foreground IoU threshold for anchor-to-GT matching")
    parser.add_argument("--box-bg-iou-thresh", type=float, default=0.4, help="Background IoU threshold; anchors below this are negatives")

    # Focal Loss parameters (built in to RetinaNet)
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal loss alpha for foreground (lesion) class; set higher than 0.25 default for imbalanced medical data")

    # Positive-only warmup
    parser.add_argument("--warmup-positive-epochs", type=int, default=0, help="Epochs to train with positive-only images before full training (0=disabled)")

    # Balanced sampling warmup (replaces positive-only warmup)
    parser.add_argument("--warmup-balanced-epochs", type=int, default=0, help="Epochs to use balanced (weighted) sampling before full training (0=disabled)")
    parser.add_argument("--warmup-pos-weight-ratio", type=float, default=10.0, help="Positive sample weight relative to negative in balanced warmup")
    parser.add_argument("--post-warmup-lr", type=float, default=None, help="LR used when rebuilding optimizer after balanced warmup ends (defaults to --lr if not set)")

    # Epoch subsampling
    parser.add_argument("--only-use", type=float, default=1.0, help="Fraction of training data to use per epoch (0.0-1.0); rotates across epochs to cover all data")

    # Inference-time box filtering
    parser.add_argument("--box-score-thresh", type=float, default=0.05, help="Score threshold for inference-time box filtering")
    parser.add_argument("--input-min-size", type=int, default=800, help="Shorter side of the image after RetinaNet resize transform (default 800). Raise to 1200 to make small lesions larger relative to anchors.")
    parser.add_argument("--recall-stop", action="store_true", help="Use Recall@0.1 (TP@0.1 / GT boxes) instead of BestThreshF1 as the early-stopping metric. Recommended for recall-first training.")
    parser.add_argument("--box-detections-per-img", type=int, default=100, help="Max detections per image at inference time")
    parser.add_argument("--val-detections-per-img", type=int, default=None, help="Max detections per image during validation (temporarily overrides --box-detections-per-img). When set (e.g. 300), prevents TP@threshold statistics from being truncated at the training inference cap, giving an accurate recall estimate for early-stopping and checkpoint selection. When omitted (default), uses the same value as --box-detections-per-img (backward-compatible).")
    parser.add_argument("--box-nms-thresh", type=float, default=0.5, help="NMS IoU threshold for post-prediction duplicate suppression; lower (e.g. 0.3) removes more overlapping FP boxes")
    parser.add_argument("--classification-loss-scale", type=float, default=1.0, help="Multiply the RetinaNet classification loss by this factor before summing with bbox_regression loss. Values > 1.0 (e.g. 2.0) strengthen the signal for distinguishing lesions from background, counteracting extreme class imbalance.")
    parser.add_argument("--disable-breast-crop", action="store_true", help="Disable breast-region cropping and train on the full processed image")
    parser.add_argument("--breast-crop-margin", type=float, default=0.05, help="Relative padding added around the detected breast crop")
    parser.add_argument("--hide-progress-bar", action="store_true", help="Suppress tqdm progress bars during training and validation")

    # Data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable random data augmentation (hflip + brightness jitter + optional rotation) on the training set")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Probability of random horizontal flip when --augment is set")
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2, help="Magnitude of random brightness jitter (±delta) when --augment is set")
    parser.add_argument(
        "--aug-rotation-max-deg",
        type=float,
        default=0.0,
        help=(
            "Max absolute rotation angle (degrees) for random small-angle rotation augmentation "
            "when --augment is set. 0 disables rotation. Recommended: 5.0-10.0. "
            "Image size is kept identical (strategy-A: zero-fill corners). "
            "Bboxes are updated via rotated-corner AABB and degenerate boxes are dropped."
        ),
    )

    # Copy-paste augmentation
    parser.add_argument("--copy-paste-prob", type=float, default=0.0, help="Probability of applying copy-paste augmentation to a negative training image (0 = disabled, recommended 0.3-0.5)")
    parser.add_argument("--copy-paste-max-pastes", type=int, default=2, help="Max lesion crops to paste per negative image when copy-paste is enabled")

    # Full-training positive sample weight (prevents post-warmup collapse)
    parser.add_argument(
        "--full-train-pos-weight-ratio",
        type=float,
        default=0.0,
        help=(
            "When > 0, keep a mild positive-oversampling via WeightedRandomSampler throughout the "
            "full training phase (after warmup). Positive samples are given this weight relative "
            "to negative samples (e.g. 3.0 means positives are 3x as likely to be sampled). "
            "Helps prevent the model from collapsing to predict-nothing on heavily imbalanced data. "
            "0.0 (default) disables this and uses plain shuffle."
        ),
    )

    # Medical pretrained backbone (rec_37+)
    parser.add_argument(
        "--medical-backbone-path",
        type=str,
        default=None,
        help=(
            "Path to a medical-domain pretrained ResNet50 checkpoint (e.g. RadImageNet). "
            "If provided, backbone weights are loaded from this file instead of COCO pretrained, "
            "shortening the ImageNet→COCO→mammography transfer chain. "
            "Supports checkpoints with state_dict/model keys and common prefixes "
            "(module., encoder., backbone., body.) which are stripped automatically."
        ),
    )

    # ── Stage-1: image-level binary classifier pre-filtering ──────────────────
    parser.add_argument("--clf-epochs", type=int, default=10,
        help="Number of training epochs for the Stage-1 image-level binary classifier.")
    parser.add_argument("--clf-lr", type=float, default=1e-3,
        help="Adam learning rate for the Stage-1 image classifier.")
    parser.add_argument("--clf-pos-weight", type=float, default=10.0,
        help="BCEWithLogitsLoss pos_weight for Stage-1 classifier (match training neg/pos ratio).")
    parser.add_argument("--clf-threshold", type=float, default=0.5,
        help="Sigmoid threshold: negative training images with predicted probability above this "
             "value are kept for Stage-2 RetinaNet training; all original positives are always kept.")
    parser.add_argument("--clf-save-path", type=str, default=None,
        help="Path to save the best Stage-1 classifier checkpoint (default: models/image_clf.pth).")
    parser.add_argument("--clf-checkpoint-path", type=str, default=None,
        help="If provided, skip Stage-1 training and load an existing classifier from this path.")
    parser.add_argument("--skip-clf-stage", action="store_true",
        help="Skip Stage-1 entirely and use the full unfiltered training set for detection, "
             "equivalent to running the original bbox-train.py.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox_resnet50.pth")
    crop_breast_region = not bool(args.disable_breast_crop)
    breast_crop_margin = max(0.0, float(args.breast_crop_margin))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    set_seed(args.seed)

    # freeze_epochs / warmup conflict check
    _freeze = int(args.freeze_epochs)
    _wbal = int(args.warmup_balanced_epochs)
    if 0 < _freeze < _wbal:
        print(
            f"[Warning] --freeze-epochs={_freeze} falls inside balanced warmup range (0~{_wbal}). "
            f"This will rebuild the optimizer mid-warmup and may cause FP instability. "
            f"Recommended: set --freeze-epochs=0 or --freeze-epochs>={_wbal}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the full training split first, then split it into train/validation
    # at the patient level so that the same patient never appears on both sides.
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=False,
        crop_breast_region=crop_breast_region,
        breast_crop_margin=breast_crop_margin,
    )

    usable_indices = list(range(len(train_dataset.samples)))
    pos_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size > 0]
    neg_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size == 0]

    train_indices, val_indices, split_summary = split_train_val_by_patient(
        samples=train_dataset.samples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    # Keep the existing positive-only behavior for training only.
    # Validation must remain untouched so that it keeps the real distribution.
    if args.positive_only:
        positive_train_indices = [i for i in train_indices if train_dataset.samples[i].boxes.size > 0]
        if positive_train_indices:
            train_indices = positive_train_indices
        else:
            print("Warning: positive_only enabled but no positive samples remain in training split; falling back to mixed training split.")

    train_indices.sort()
    val_indices.sort()

    # ── Stage 1: Image-level binary classifier training and pre-filtering ─────
    if not args.skip_clf_stage:
        print(f"\n{'=' * 60} Stage 1: Image-level binary classifier {'=' * 60}")
        clf_save_path = Path(args.clf_save_path) if args.clf_save_path else (root / "models" / "image_clf.pth")
        classifier = build_image_classifier(
            medical_backbone_path=args.medical_backbone_path if args.medical_backbone_path else None
        )
        classifier.to(device)

        if args.clf_checkpoint_path and Path(args.clf_checkpoint_path).exists():
            print(f"[Stage-1] Loading existing classifier from: {args.clf_checkpoint_path}")
            ckpt = torch.load(args.clf_checkpoint_path, map_location="cpu")
            classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
        else:
            clf_dataset = ImageClassificationDataset(train_dataset, train_indices)
            clf_loader = DataLoader(
                clf_dataset,
                batch_size=min(int(args.batch_size) * 8, 32),
                shuffle=True,
                num_workers=int(args.num_workers),
                pin_memory=torch.cuda.is_available(),
            )
            clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=float(args.clf_lr), weight_decay=1e-4)
            _eff_clf = max(1, int(args.clf_epochs))
            clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_optimizer, T_max=_eff_clf, eta_min=1e-6)
            best_clf_loss = float("inf")
            for clf_epoch in range(_eff_clf):
                clf_loss = train_clf_one_epoch(
                    classifier, clf_loader, clf_optimizer, device,
                    clf_epoch, _eff_clf,
                    pos_weight=float(args.clf_pos_weight),
                    disable_tqdm=args.hide_progress_bar,
                )
                clf_scheduler.step()
                print(f"[Stage-1] Epoch {clf_epoch + 1}/{_eff_clf} | clf_loss={clf_loss:.4f}")
                if clf_loss < best_clf_loss:
                    best_clf_loss = clf_loss
                    clf_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model_state_dict": classifier.state_dict()}, clf_save_path)
            print(f"[Stage-1] Best classifier saved to: {clf_save_path}")
            ckpt = torch.load(clf_save_path, map_location="cpu")
            classifier.load_state_dict(ckpt["model_state_dict"])

        train_indices = run_clf_filter(
            classifier, train_dataset, train_indices,
            train_dataset.samples, device,
            threshold=float(args.clf_threshold),
        )
        del classifier
        torch.cuda.empty_cache()
        print(f"{'=' * 60} Stage 2: RetinaNet detector training {'=' * 60}\n")
    # ── Stage 1 end ────────────────────────────────────────────────────────────

    train_summary = summarize_subset(train_dataset.samples, train_indices)
    val_summary = summarize_subset(train_dataset.samples, val_indices)
    train_neg_to_pos_ratio = compute_neg_pos_ratio(train_summary)
    val_neg_to_pos_ratio = compute_neg_pos_ratio(val_summary)

    split_summary["train"] = train_summary
    split_summary["val"] = val_summary
    split_summary["train_patients"] = int(len({train_dataset.samples[i].patient_id for i in train_indices}))
    split_summary["val_patients"] = int(len({train_dataset.samples[i].patient_id for i in val_indices}))

    _base_train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Optionally wrap the training subset with augmentation.
    if args.augment:
        train_subset = TrainAugmentWrapper(
            _base_train_subset,
            hflip_prob=float(args.aug_hflip_prob),
            brightness_delta=float(args.aug_brightness_delta),
            rotation_max_deg=float(args.aug_rotation_max_deg),
        )
        print(
            f"[Info] Training augmentation enabled | hflip_prob={args.aug_hflip_prob} | "
            f"brightness_delta=±{args.aug_brightness_delta} | "
            f"rotation_max_deg=±{args.aug_rotation_max_deg}"
        )
    else:
        train_subset = _base_train_subset

    # Optionally wrap with copy-paste augmentation.
    copy_paste_prob = float(args.copy_paste_prob)
    if copy_paste_prob > 0.0:
        # Find positive indices within train_subset (indices relative to train_subset).
        # train_indices[i] is the index into train_dataset; we need indices into train_subset.
        _pos_in_subset = [
            i for i, idx in enumerate(train_indices)
            if train_dataset.samples[idx].boxes.size > 0
        ]
        train_subset = CopyPasteWrapper(
            train_subset,
            positive_indices=_pos_in_subset,
            paste_prob=copy_paste_prob,
            max_pastes=int(args.copy_paste_max_pastes),
        )
        print(
            f"[Info] Copy-paste augmentation enabled | "
            f"paste_prob={copy_paste_prob} | max_pastes={args.copy_paste_max_pastes} | "
            f"donor_pool={len(_pos_in_subset)} positive images"
        )

    if len(train_subset) == 0:
        raise ValueError("Training split is empty after patient-level split and filtering.")
    if len(val_subset) == 0:
        raise ValueError("Validation split is empty. Please check the CSV and splitting logic.")

    # Parse anchor sizes from args.
    # When the user does not pass --anchor-sizes (default "16,32,64,128,256" keeps legacy CLI),
    # we detect that case and pass None to build_model so the COCO-default 9-anchor generator
    # is retained and the COCO head weights load without shape mismatch.
    _raw_anchor_arg = str(args.anchor_sizes).strip()
    _default_anchor_str = "16,32,64,128,256"
    if _raw_anchor_arg == _default_anchor_str:
        anchor_sizes = None   # use torchvision default (9 anchors/location)
    else:
        anchor_sizes = tuple((int(s.strip()),) for s in _raw_anchor_arg.split(",") if s.strip())

    model = build_model(
        num_classes=2,
        anchor_sizes=anchor_sizes,
        fg_iou_thresh=float(args.box_fg_iou_thresh),
        bg_iou_thresh=float(args.box_bg_iou_thresh),
        score_thresh=float(args.box_score_thresh),
        detections_per_img=int(args.box_detections_per_img),
        nms_thresh=float(args.box_nms_thresh),
        focal_loss_alpha=float(args.focal_alpha),
        focal_loss_gamma=float(args.focal_gamma),
        medical_backbone_path=args.medical_backbone_path if args.medical_backbone_path else None,
        input_min_size=int(args.input_min_size),
    )
    model.to(device)

    # Freeze low-level backbone layers initially if requested
    if int(args.freeze_epochs) > 0:
        freeze_backbone_layers(model)

    optimizer = create_optimizer(model, args, base_lr=float(args.lr))

    # Choose LR scheduler: StepLR if step size provided, else CosineAnnealingLR
    # T_max excludes balanced warmup epochs so cosine decay aligns with actual training phase
    warmup_balanced_epochs = int(args.warmup_balanced_epochs)
    only_use = float(args.only_use)
    _effective_epochs = max(1, int(args.epochs) - warmup_balanced_epochs)
    if int(args.lr_step_size) > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_effective_epochs, eta_min=1e-6)

    history: List[Dict[str, float]] = []

    print(f"Total usable images: {len(usable_indices)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(
        f"Split summary | train images: {train_summary['images']} (pos={train_summary['positive_images']}, neg={train_summary['negative_images']}) "
        f"| val images: {val_summary['images']} (pos={val_summary['positive_images']}, neg={val_summary['negative_images']})"
    )
    print(
        f"Split summary | train patients: {split_summary['train_patients']} | val patients: {split_summary['val_patients']} | "
        f"val_ratio≈{split_summary['val_ratio']}"
    )
    print(
        f"[Info] Image imbalance | train neg/pos={train_neg_to_pos_ratio:.2f} | "
        f"val neg/pos={val_neg_to_pos_ratio:.2f}"
    )
    print(
        f"[Info] Coarse localization mode | breast_crop_region={crop_breast_region} | "
        f"breast_crop_margin={breast_crop_margin:.3f}"
    )
    print(f"Device: {device}")
    warn_on_small_epoch_positive_pool(train_summary, only_use)

    # Prepare balanced sampling warmup (replaces positive-only warmup)
    warmup_pos_epochs = int(args.warmup_positive_epochs)
    _warmup_weights: List[float] = []
    if warmup_balanced_epochs > 0:
        _train_pos_count = sum(1 for i in train_indices if train_dataset.samples[i].boxes.size > 0)
        if _train_pos_count > 0:
            _pos_weight = float(args.warmup_pos_weight_ratio)
            for j in range(len(train_indices)):
                is_pos = train_dataset.samples[train_indices[j]].boxes.size > 0
                _warmup_weights.append(_pos_weight if is_pos else 1.0)
            print(f"[Info] Balanced warmup: {warmup_balanced_epochs} epochs, pos_weight_ratio={_pos_weight}, train_size={len(train_indices)}")
        else:
            warmup_balanced_epochs = 0
            print("[Warning] No positive training samples; disabling balanced warmup")
        _cls_loss_scale = float(args.classification_loss_scale)
        if _cls_loss_scale != 1.0:
            print(f"[Info] classification_loss_scale={_cls_loss_scale} (classification loss will be multiplied by this factor)")
    # Legacy positive-only warmup subset (kept for backward compat but not recommended)
    warmup_train_subset = None
    if warmup_pos_epochs > 0 and warmup_balanced_epochs == 0 and not args.positive_only:
        _pos_train_idx = [i for i in train_indices if train_dataset.samples[i].boxes.size > 0]
        if _pos_train_idx:
            warmup_train_subset = Subset(train_dataset, _pos_train_idx)
            print(f"[Info] Positive-only warmup: {len(_pos_train_idx)} images for first {warmup_pos_epochs} epochs")
        else:
            warmup_pos_epochs = 0
            print("[Warning] No positive training samples; disabling positive-only warmup")

    best_val_f1 = -float("inf")
    best_recall_at_low_thresh = -float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(int(args.epochs)):
        print(f"\n{'=' * 60} Epoch {epoch + 1}/{int(args.epochs)} start {'=' * 60}")

        # --- Phase detection ---
        is_warmup_balanced = warmup_balanced_epochs > 0 and epoch < warmup_balanced_epochs
        _is_legacy_warmup = (not is_warmup_balanced) and warmup_train_subset is not None and epoch < warmup_pos_epochs

        if is_warmup_balanced and epoch == 0:
            print(f"[Warmup] Starting balanced sampling warmup phase ({warmup_balanced_epochs} epochs)")
        elif _is_legacy_warmup and epoch == 0:
            print(f"[Warmup] Starting positive-only warmup phase ({warmup_pos_epochs} epochs)")

        # --- Reset optimizer/scheduler when transitioning from balanced warmup to full training ---
        rebuilt_warmup = False
        if not is_warmup_balanced and epoch == warmup_balanced_epochs and warmup_balanced_epochs > 0:
            print(f"[Info] Balanced warmup complete, resetting optimizer and switching to full training ({len(train_subset)} images)")
            _post_warmup_lr = float(args.post_warmup_lr) if args.post_warmup_lr is not None else float(args.lr)
            if args.post_warmup_lr is not None:
                print(f"[Info] post-warmup LR set to {_post_warmup_lr} (from --post-warmup-lr)")
            optimizer = create_optimizer(model, args, base_lr=_post_warmup_lr)
            remaining = max(1, int(args.epochs) - epoch)
            if int(args.lr_step_size) > 0:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
            else:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
            rebuilt_warmup = True
        elif not _is_legacy_warmup and epoch == warmup_pos_epochs and warmup_train_subset is not None:
            print(f"[Info] Warmup complete, switching to full training ({len(train_subset)} images)")

        # --- Data selection ---
        if is_warmup_balanced:
            # Balanced warmup: use WeightedRandomSampler on the full training set
            sampler = WeightedRandomSampler(_warmup_weights, num_samples=len(train_indices), replacement=True)
            train_loader = DataLoader(
                train_subset,
                batch_size=int(args.batch_size),
                sampler=sampler,
                num_workers=int(args.num_workers),
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        elif _is_legacy_warmup:
            # Legacy positive-only warmup
            train_loader = DataLoader(
                warmup_train_subset,
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            # Normal training, possibly with --only-use subsampling
            if only_use < 1.0:
                _epoch_offset = epoch - max(warmup_balanced_epochs, warmup_pos_epochs if warmup_train_subset else 0)
                epoch_indices = select_epoch_subset(
                    train_indices, train_dataset.samples, _epoch_offset, only_use, int(args.seed))
                epoch_subset = Subset(train_dataset, epoch_indices)
                if _epoch_offset == 0 or (_epoch_offset % 5 == 0):
                    _ep_pos = sum(1 for i in epoch_indices if train_dataset.samples[i].boxes.size > 0)
                    _ep_neg = len(epoch_indices) - _ep_pos
                    print(f"[Info] Epoch {epoch+1} subset: {len(epoch_indices)} images (pos={_ep_pos}, neg={_ep_neg})")
            else:
                epoch_subset = train_subset

            # Full-training weighted sampler: keep a mild positive-sample bias
            # to prevent the model from collapsing to "predict nothing".
            _full_train_pos_ratio = float(args.full_train_pos_weight_ratio)
            if _full_train_pos_ratio > 0.0:
                # Build per-sample weights based on the current epoch_subset.
                # epoch_subset may be a TrainAugmentWrapper, a Subset, or another Subset;
                # walk down to the underlying train_dataset.samples for the box check.
                _epoch_indices: List[int]
                if hasattr(epoch_subset, "dataset") and hasattr(epoch_subset.dataset, "indices"):
                    # TrainAugmentWrapper -> Subset -> train_dataset
                    _epoch_indices = list(epoch_subset.dataset.indices)  # type: ignore[union-attr]
                elif hasattr(epoch_subset, "indices"):
                    # plain Subset
                    _epoch_indices = list(epoch_subset.indices)  # type: ignore[union-attr]
                else:
                    _epoch_indices = list(range(len(epoch_subset)))
                _ft_weights = [
                    _full_train_pos_ratio if train_dataset.samples[i].boxes.size > 0 else 1.0
                    for i in _epoch_indices
                ]
                _ft_sampler = WeightedRandomSampler(
                    _ft_weights,
                    num_samples=len(_epoch_indices),
                    replacement=True,
                )
                train_loader = DataLoader(
                    epoch_subset,
                    batch_size=int(args.batch_size),
                    sampler=_ft_sampler,
                    num_workers=int(args.num_workers),
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                train_loader = DataLoader(
                    epoch_subset,
                    batch_size=int(args.batch_size),
                    shuffle=True,
                    num_workers=int(args.num_workers),
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )

        # Validation loader must not shuffle and must not use any sampling tricks.
        val_loader = DataLoader(
            val_subset,
            batch_size=max(1, int(args.val_batch_size)),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        # decide accumulation steps (honor --fuck-running / --accumulation-steps)
        accumulation_steps = 1 if args.fuck_running else max(1, int(args.accumulation_steps))

        # iter-level linear warmup only for first epoch (use same heuristic as train-abc.py)
        warmup_scheduler = None
        if epoch == 0:
            warmup_iters = min(1000, len(train_loader) - 1)
            if warmup_iters > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)

        avg_loss, optimizer_steps, avg_sublosses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            int(args.epochs),
            accumulation_steps,
            warmup_scheduler,
            disable_tqdm=args.hide_progress_bar,
            classification_loss_scale=float(args.classification_loss_scale),
        )

        # Unfreeze backbone after configured freeze epochs
        rebuilt_this_epoch = rebuilt_warmup

        if int(args.freeze_epochs) > 0 and (epoch + 1) == int(args.freeze_epochs):
            print(f"[Info] Unfreezing backbone layers after {args.freeze_epochs} epochs and rebuilding optimizer")
            unfreeze_backbone_layers(model)
            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.lr)
            optimizer = create_optimizer(model, args, base_lr=current_lr)
            if int(args.lr_step_size) > 0:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
            else:
                # After unfreezing, schedule for remaining epochs (warmup handled at iter-level)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) - epoch - 1), eta_min=1e-6)
            rebuilt_this_epoch = True

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=float(args.val_score_threshold),
            iou_threshold=float(args.val_iou_threshold),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=args.hide_progress_bar,
            val_detections_per_img=args.val_detections_per_img,  # None means no override (use model's current value)
        )

        # Only step the epoch-level scheduler if we actually performed any optimizer.step()
        # Skip scheduler stepping during balanced warmup phase
        if is_warmup_balanced:
            print(f"[Info] Warmup epoch {epoch + 1}: skipping lr_scheduler.step()")
        elif rebuilt_this_epoch:
            # optimizer/scheduler was just rebuilt this epoch — do not step (this is normal)
            print(f"[Info] Epoch {epoch + 1}: optimizer/scheduler rebuilt this epoch, skipping lr_scheduler.step()")
        elif optimizer_steps > 0:
            print(f"[Info] lr_scheduler.step() at epoch {epoch + 1}.")
            lr_scheduler.step()
        else:
            print(f"[Warning] No optimizer.step() executed in epoch {epoch + 1}; skipping lr_scheduler.step() to avoid PyTorch warning.")

        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "classification": float(avg_sublosses.get("classification", 0.0)),
            "bbox_regression": float(avg_sublosses.get("bbox_regression", 0.0)),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_best_thresh_f1": float(val_metrics.get("best_thresh_f1", val_metrics["f1"])),
            "val_best_thresh": float(val_metrics.get("best_thresh", float(args.val_score_threshold))),
            "val_tp": float(val_metrics["tp"]),
            "val_fp": float(val_metrics["fp"]),
            "val_fn": float(val_metrics["fn"]),
            "val_raw_preds": float(val_metrics.get("raw_preds", 0)),
            "val_avg_raw_preds_per_image": float(val_metrics.get("avg_raw_preds_per_image", 0)),
            "val_recall_at_01": float(val_metrics.get("tp@0.1", val_metrics["tp"])) / max(float(val_metrics.get("gt_boxes", 1)), 1),
            "val_recall_at_03": float(val_metrics.get("tp@0.3", 0)) / max(float(val_metrics.get("gt_boxes", 1)), 1),
            "val_recall_at_05": float(val_metrics.get("tp@0.5", 0)) / max(float(val_metrics.get("gt_boxes", 1)), 1),
        }
        history.append(record)

        # Use best-threshold F1 for checkpoint selection so we don't miss epochs
        # where the model is better at a threshold other than val_score_threshold.
        current_f1 = float(val_metrics.get("best_thresh_f1", val_metrics["f1"]))
        # Recall-first: also track recall@0.1 (TP@0.1 / total GT boxes).
        # When --recall-stop is set, early stopping is driven by recall@0.1 instead of F1.
        gt_boxes_total = float(val_metrics.get("gt_boxes", max(val_metrics["tp"] + val_metrics["fn"], 1)))
        recall_at_01 = float(val_metrics.get("tp@0.1", val_metrics["tp"])) / max(gt_boxes_total, 1)
        current_stop_metric = recall_at_01 if args.recall_stop else current_f1
        best_stop_metric = best_recall_at_low_thresh if args.recall_stop else best_val_f1
        improved = current_stop_metric > (best_stop_metric + float(args.min_delta))

        if improved:
            best_val_f1 = current_f1
            best_recall_at_low_thresh = recall_at_01
            best_epoch = epoch + 1
            no_improve_epochs = 0

            meta = {
                "task": "bbox_detection",
                "num_classes": 2,
                "class_names": ["background", "lesion"],
                "csv_path": str(csv_path),
                "images_root": str(images_root),
                "positive_only": bool(args.positive_only),
                "history": history,
                "torchvision_model": "retinanet_resnet50_fpn_v2",
                "anchor_sizes": str(args.anchor_sizes),
                "focal_loss_alpha": float(args.focal_alpha),
                "focal_loss_gamma": float(args.focal_gamma),
                "val_ratio": float(args.val_ratio),
                "val_score_threshold": float(args.val_score_threshold),
                "val_iou_threshold": float(args.val_iou_threshold),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "best_epoch": int(best_epoch),
                "best_val_precision": float(val_metrics["precision"]),
                "best_val_recall": float(val_metrics["recall"]),
                "best_val_f1": float(val_metrics["f1"]),
                "best_val_recall_at_01": float(recall_at_01),
                "crop_breast_region": bool(crop_breast_region),
                "breast_crop_margin": float(breast_crop_margin),
                "box_fg_iou_thresh": float(args.box_fg_iou_thresh),
                "box_bg_iou_thresh": float(args.box_bg_iou_thresh),
                "warmup_positive_epochs": int(args.warmup_positive_epochs),
                "warmup_balanced_epochs": int(warmup_balanced_epochs),
                "only_use": float(only_use),
                "train_neg_to_pos_ratio": float(train_neg_to_pos_ratio),
                "val_neg_to_pos_ratio": float(val_neg_to_pos_ratio),
                "box_score_thresh": float(args.box_score_thresh),
                "box_nms_thresh": float(args.box_nms_thresh),
                "box_detections_per_img": int(args.box_detections_per_img),
                "input_min_size": int(args.input_min_size),
                "split_summary": split_summary,
            }
            # Save the best checkpoint, not the last one.
            save_checkpoint(save_path, model, meta)
            print(f"[Info] Saved best checkpoint to: {save_path}")
        else:
            no_improve_epochs += 1

        print(
            f"[Milestone] Epoch {epoch + 1:03d}/{int(args.epochs):03d} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_F1={val_metrics['f1']:.4f} | "
            f"val_BestThreshF1={val_metrics.get('best_thresh_f1', val_metrics['f1']):.4f}@{val_metrics.get('best_thresh', args.val_score_threshold)} | "
            f"lr={record['lr']:.6f}"
        )
        print(
            (
                f"  Train sub-losses: classification={record.get('classification', 0.0):.6f}, "
                f"bbox_regression={record.get('bbox_regression', 0.0):.6f}"
            )
        )
        # Recall-at-threshold summary: shows how many GT lesions are found at each threshold.
        # TP / GT boxes = recall. Goal: maximise TP@0.1 (recall with lenient threshold).
        gt_total = int(val_metrics.get("gt_boxes", 0))
        tp_01 = int(val_metrics.get("tp@0.1", 0))
        tp_03 = int(val_metrics.get("tp@0.3", 0))
        tp_05 = int(val_metrics.get("tp@0.5", 0))
        recall_01 = tp_01 / max(gt_total, 1)
        recall_03 = tp_03 / max(gt_total, 1)
        recall_05 = tp_05 / max(gt_total, 1)
        stop_label = "recall@0.1" if args.recall_stop else "BestThreshF1"
        best_stop_val = best_recall_at_low_thresh if args.recall_stop else best_val_f1
        print(
            f"  [Recall] GT_boxes={gt_total} | "
            f"Recall@0.1={recall_01:.3f}({tp_01}/{gt_total}) | "
            f"Recall@0.3={recall_03:.3f}({tp_03}/{gt_total}) | "
            f"Recall@0.5={recall_05:.3f}({tp_05}/{gt_total})"
        )
        print(
            f"  Val counts: TP={int(val_metrics['tp'])}, FP={int(val_metrics['fp'])}, FN={int(val_metrics['fn'])} | "
            f"best_{stop_label}={best_stop_val:.4f} (epoch {best_epoch})"
        )

        if int(args.patience) > 0 and no_improve_epochs >= int(args.patience):
            print(
                f"[EarlyStopping] Stop metric ({stop_label}) has not improved for "
                f"{int(args.patience)} consecutive epochs. Stopping at epoch {epoch + 1}."
            )
            break

    final_meta = {
        "task": "bbox_detection",
        "num_classes": 2,
        "class_names": ["background", "lesion"],
        "csv_path": str(csv_path),
        "images_root": str(images_root),
        "positive_only": bool(args.positive_only),
        "history": history,
        "torchvision_model": "retinanet_resnet50_fpn_v2",
        "anchor_sizes": str(args.anchor_sizes),
        "focal_loss_alpha": float(args.focal_alpha),
        "focal_loss_gamma": float(args.focal_gamma),
        "val_ratio": float(args.val_ratio),
        "val_score_threshold": float(args.val_score_threshold),
        "val_iou_threshold": float(args.val_iou_threshold),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1 if best_val_f1 != -float("inf") else 0.0),
        "best_val_recall_at_01": float(best_recall_at_low_thresh if best_recall_at_low_thresh != -float("inf") else 0.0),
        "recall_stop": bool(args.recall_stop),
        "crop_breast_region": bool(crop_breast_region),
        "breast_crop_margin": float(breast_crop_margin),
        "box_fg_iou_thresh": float(args.box_fg_iou_thresh),
        "box_bg_iou_thresh": float(args.box_bg_iou_thresh),
        "warmup_positive_epochs": int(args.warmup_positive_epochs),
        "warmup_balanced_epochs": int(warmup_balanced_epochs),
        "only_use": float(only_use),
        "train_neg_to_pos_ratio": float(train_neg_to_pos_ratio),
        "val_neg_to_pos_ratio": float(val_neg_to_pos_ratio),
        "box_score_thresh": float(args.box_score_thresh),
        "box_nms_thresh": float(args.box_nms_thresh),
        "box_detections_per_img": int(args.box_detections_per_img),
        "input_min_size": int(args.input_min_size),
        "split_summary": split_summary,
    }

    print(f"Best checkpoint saved at: {save_path}")
    print(f"Best epoch: {best_epoch}, best val_F1: {final_meta['best_val_f1']:.4f}, best Recall@0.1: {final_meta['best_val_recall_at_01']:.4f}")
    print(json.dumps(final_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start time:  {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None):
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
        end_time = time.time()
        if reason:
            print(reason)
        print(f"Exit time:   {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running time: {end_time - start_time:.2f} s.")

    def _handle_signal(signum, _frame):
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"Received {signame}")

    atexit.register(_on_exit)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        main()
    except KeyboardInterrupt as exc:
        reason = "[Info] Training interrupted by user (Ctrl+C)."
        if exc.args and exc.args[0]:
            reason = f"[Info] {exc.args[0]}. Training interrupted."
        _on_exit(reason)
        raise SystemExit(130)
