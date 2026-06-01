r"""
RetinaNet 全图直接检测（torchvision）

前身：bbox-train.py（rec_1–rec_47，Faster R-CNN → RetinaNet_V2 演进）
本文件：方向 H，从 rec_48 起重写，RetinaNet + ResNet50-FPN 直接回归 box。
硬件环境：NVIDIA GeForce RTX 4090，24 GB VRAM

运行命令：

需要下载 RadImageNet ResNet50 预训练权重，即 models/raw/ResNet50.pt 文件：
bash src/init/download_backbone.sh

正式训练：
python src/data/bounding-box/bbox-train.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 1024 \
    --input-w 1024 \
    --patience 10 \
    --monitor-metric fbeta2_ref \
    --ref-score 0.3 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --save-path models/bbox_resnet50.pth \
    --augment \
    --aug-contrast-range 0.8 1.2 \
    --aug-scale-min 0.85 \
    --focal-alpha 0.25 \
    --focal-gamma 1.0 \
    --min-box-side 24.0 \
    --max-box-ar 3.0 \
    --cliff-patience-ratio 0.6 \
    --amp \
    --hide-progress-bar


─────────────────────────────────────────────────────────────────────────────
前身 bbox-train.py 改版历史（rec_1–rec_47）
─────────────────────────────────────────────────────────────────────────────

1.  针对训练后期 Loss 卡在 0.19 左右无法下降的问题，需从学习率策略、
    模型结构和优化器等方面进行系统性干预，打破局部最优解。
2.  引入学习率 Warmup 机制，防止初始训练时因梯度过大破坏预训练权重，并提供合理的初始学习率设定。
3.  增加权重衰减（Weight Decay）的配置参数，通过正则化手段有效防止模型在较小数据集上过拟合。
4.  新增命令行参数 "--fuck-running" 作为算力切换开关：
    当不含此参数时，代码需在 batch_size=2 的前提下通过累积 4 个 step 再执行 optimizer.step()
    来变相实现 batch_size=8 的梯度累积；当存在该参数时，直接使用配置的较大 batch_size 进行正常训练。
 *  fix: 在算得正样本时，需要按照比例上取整，以防止正样本丢失。
5.  针对 912×1520 的高分辨率医疗影像数据，修改模型的 AnchorGenerator，为其添加 8 和 16 这
    样更小的 scale 尺寸，以强化微小病灶的检测能力。
6.  优化训练策略，通过调整模型内部的 ROI 采样比例等参数，变相实现 Hard Negative Mining（挖掘难例）。
7.  实现渐冻层训练策略：在训练初期主动冻结 ResNet 的 layer1 和 layer2 层，仅训练 FPN 和检测头；
    在设定的几轮 Epoch 之后，全量解冻这些底层网络进行全局微调。
8.  强制采用 torchvision 中的 fasterrcnn_resnet50_fpn_v2 版本模型，以利用其更先进的
    数据增强策略和优化过的 FPN 特征提取结构。
  * feat: 需要使用 FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 权重。
-----------
9.  取消对每一个 epoch、batch 的样本强制比例分配，保留原始的正负样本比例。每个 epoch 使用全量
    样本直接进行训练，通过 shuffle=True 打乱，以防止模型的过拟合。
10. 从 training 中自行划出约 15% 作为验证集；划分优先按 patient_id 进行，避免同一 patient
    同时出现在 train 和 val；验证集完全不参与训练；使用验证集指标保存最佳模型；若连续 N 个
    epoch 没有提升则停止训练，N 由参数控制。
-----------
11. 引入可切换的 ROI 分类损失策略：支持标准 CE、Weighted Cross-Entropy 和 Focal Loss 三种模式。
12. 引入在线难例挖掘（OHEM）机制：先逐样本计算独立 loss，按困难度降序排列后仅保留前 K 个最困难
    样本参与反向传播。
13. 将训练阶段 RPN 和 ROI Head 的 IoU 匹配阈值提取为命令行可配置参数。
14. 新增两阶段训练策略（positive-only warmup，通过 --warmup-positive-epochs 配置）：前 N 个
    epoch 仅使用正样本图像训练，之后恢复全量样本训练。
-----------
15. 废弃 Positive-Only Warmup，改用 Balanced Sampling Warmup：前 N 个 epoch 使用
    WeightedRandomSampler 对正样本进行过采样，warmup 结束后重建 optimizer 和 lr_scheduler 切换
    回全量训练。
16. 禁止 Focal Loss + OHEM 同时启用，新增启动时互斥检查，当同时启用时自动禁用 OHEM 并打印警告。
17. 修复 CosineAnnealingLR 的 T_max 与 warmup 阶段冲突，warmup 阶段跳过 scheduler step，
    防止 LR 在 warmup 阶段被 cosine 过度拉低。
18. 在 validate_one_epoch 中增加多阈值验证报告（threshold=0.1/0.3/0.5/0.7/0.9）。
19. 在 validate_one_epoch 中增加 RPN proposal 数量监控，检测 RPN 是否有效抑制背景。
20. 在 build_model 中暴露 box_score_thresh 和 box_detections_per_img 参数，控制推理时的
    低分框过滤和每图最大检测数。
21. 新增 --only-use 参数（float，默认 1.0）：每个 epoch 仅使用分配数据总量的指定比例，
    并对正样本设最低保护；通过跨 epoch 轮转采样机制，确保所有图像在多个 epoch 中均被训练到。
-----------
22. 修复误导性警告：将 scheduler 更新判断从二分支扩展为四分支，明确区分 warmup/重建/正常/异常。
23. 新增 --post-warmup-lr 参数：balanced warmup 结束后重建 optimizer 时使用该 LR，减少 warmup
    结束后的梯度震荡。
24. 新增 freeze_epochs / warmup 冲突检测，提示用户可能引发 FP 反弹的配置。
25. 在每个 epoch 循环开始处打印分割线，便于在长日志中快速定位各 epoch 的起止边界。
26. 在 weighted_ce 模式下新增背景权重保护：若 --cls-weight-bg < 1.0 默认将背景权重钳制到 1.0。
27. 启动时打印 train/val 的 neg/pos image ratio，并对 --only-use < 1.0 做阳性样本量预警。
28. 为了更符合"病灶区域的初步框选"目标，在训练和验证阶段默认启用乳房主体区域裁剪：先检测乳房
    主体轮廓，再对图像进行裁剪，并将 bbox 同步重映射到裁剪后的局部坐标系。
29. 将乳房主体裁剪的开关和边缘留白提取为参数，并把裁剪配置写入 checkpoint meta。
-----------
30. 新增随机数据增强支持（通过 --augment 启用），仅作用于训练集：随机水平翻转、随机亮度扰动、
    随机小角度旋转（只对单框图像执行）；用 TrainAugmentWrapper 包裹训练 Subset。
31. 修复 RPN 正样本 IoU 阈值：将 --rpn-fg-iou-thresh 默认值从 0.7 降低到 0.5，改善小病灶
    anchor 的匹配（resize 后约 225×167px 与最优 anchor 的最大 IoU 仅约 0.57）。
32. 新增全训练阶段持续加权采样（--full-train-pos-weight-ratio），在全量训练后若该参数 > 0
    则继续对正样本保持轻微过采样。
-----------
33. 针对"FP 极高、置信度普遍 > 0.9、F1 长期低于 0.02"的训练失效问题，进行系统性修复：
    a. 参数调整：缩短 warmup、降低过采压力、全量训练不再强制过采样、加大 FP 惩罚；
    b. 新增 --box-nms-thresh 参数，降低后可更积极地去除空间相近的重复 FP 框；
    c. 新增 --rpn-objectness-loss-scale 参数，强化 RPN 拒绝背景区域的训练信号；
    d. 修复 checkpoint 选择逻辑：改用各阈值遍历后的最优 F1 来判断是否保存 best checkpoint。
-----------
34. 彻底切换检测框架：从 Faster R-CNN 迁移至 RetinaNet_ResNet50_FPN_V2，完成医学影像灰度输入
    适配，从根本上解决两阶段 RPN 瓶颈与 Focal Loss 无法原生作用于 proposal 生成阶段的缺陷：
    a. 架构替换：删除 fasterrcnn_resnet50_fpn_v2 相关代码；新增 retinanet_resnet50_fpn_v2；
    b. 删除自定义 loss 模块（约 120 行），因为 RetinaNet 已原生集成 Focal Loss；
    c. 重写 build_model()：加载 COCO 预训练 backbone 权重 → 构建 2 类 RetinaNet；
    d. 新增灰度 conv1 均值初始化：对 conv1 的 3 通道权重取均值后广播覆盖；
    e. 简化 train_one_epoch()；精简 parse_args()，删除 Faster R-CNN 专有参数。

── rec_34–47 结果摘要（以上 Faster R-CNN 演进均未达到可用水平）────────────

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
-----------
37. 数据预处理 + 医学预训练 backbone 双重升级（rec_37）：
    a. CLAHE 对比度增强：在 normalize_image() 的灰度图路径中，将 CLAHE
       （clipLimit=2.0, tileGridSize=8×8）应用于转换为 RGB 前的灰度图。
    b. 医学预训练 backbone：新增 --medical-backbone-path 参数；若提供，则用
       RadImageNet ResNet50 权重覆盖 COCO backbone，缩短迁移链。
    ── rec_37 三项 bug 修复（训练过程中发现）──
    c. RadImageNet key 映射修复：数字索引键（backbone.0.weight 等）映射到
       torchvision 命名键（0→conv1, 4→layer1 等），修复权重未实际加载问题。
    d. anchor_generator 关键字参数冲突修复：改为构建后赋值
       model.anchor_generator = anchor_generator，修复参数静默吞掉的 bug。
    e. RetinaNet head 重建修复：替换 anchor_generator 后立即用正确的
       num_anchors 重建 model.head（RetinaNetHead + GroupNorm32）。
-----------
38. COCO head 保留 + RadImageNet backbone + copy-paste 数据增强（rec_38）：
    a. 保留 COCO head 策略：构建 weights=None、num_classes=2 的模型后，以
       strict=False 加载完整 COCO state_dict，回归 head 和 FPN 完整加载，
       分类 head 因类数不匹配自动随机初始化。
    b. RadImageNet backbone 覆盖（保留）：COCO weights 加载后，以 strict=False
       用 RadImageNet ResNet50 权重覆盖 backbone.body。conv1 做通道均值适配。
    c. Copy-paste 数据增强（CopyPasteWrapper 类，--copy-paste-prob）：
       仅对负样本以 paste_prob 概率触发；Donor 从正样本中随机采样；
       Tissue 约束在乳腺组织区域内随机选取粘贴位置；
       LCC mask 去除黑色背景；Feathering alpha 渐变使边缘平滑融合；
       亮度对齐（P95 基准）缩放 crop 亮度匹配背景；中心压暗高斯 gamma 映射；
       重叠检测（IoU>0.3 跳过）避免像素覆盖。
-----------
39. 路线 3 改进 + 召回率优先策略（rec_39）：
    a. 新增 --input-min-size 参数（推荐 1200）：提升短边尺寸改善小病灶检测。
    b. 将 --focal-alpha 提升至 0.8：漏检惩罚远大于误报，驱动高召回率。
    c. 放宽 --box-nms-thresh 至 0.5：减少重复框被 NMS 压制导致的病灶丢失。
    d. 降低 --box-score-thresh 至 0.001，--box-detections-per-img 增至 20。
    e. 关闭 copy-paste（rec_38 对比 rec_37 F1 下降，copy-paste 效果为负）。
    f. 新增 --recall-stop 标志：以 Recall@0.1 代替 BestThreshF1 作为早停标准。
    g. 日志新增 [Recall] 行：打印 GT_boxes、Recall@0.1/0.3/0.5 及 TP/GT 数字。
-----------
40. 验证评估无截断 + 参数重平衡（rec_40）：
    a. 恢复全量阶段正样本过采样：--full-train-pos-weight-ratio 改为 3.0，
       提升 effective batch 中正样本期望约 3 倍。
    b. 新增 --classification-loss-scale 参数（rec_40 使用 2.0）：强化分类信号
       占总 loss 的比例，迫使模型更专注于区分病灶与背景。
-----------
41. 算法层面修复（rec_41）：
    a. 新增 --val-detections-per-img 参数（默认 300）：验证时临时将
       model.detections_per_img 提高至 300，使 TP@0.1 真实反映模型召回能力，
       不再受推理截断影响，解决 rec_39/40 早停基于失真指标的问题。
    b. focal_alpha 0.8 → 0.5：背景权重翻倍，消除"全部输出高分"退化解。
    c. pos-weight-ratio 3.0 → 10.0：进一步增加每批梯度更新中正样本数量。
-----------
42. 三方向算法改进（rec_42，分支文件 bbox-train-A/B/C.py，原文件不变）：
    方案 A（bbox-train-A.py）— Anchor-level Hard Negative Mining（HNM）：
        monkey-patch classification_head.compute_loss，仅保留 top-K 最难负样本，
        K = max(n_fg × neg_pos_ratio, min_neg)，消除背景 anchor 梯度淹没问题。
    方案 B（bbox-train-B.py）— FCOS（anchor-free）：
        将模型替换为 fcos_resnet50_fpn；无 anchor 比例失衡问题；centerness 分支
        天然抑制非中心位置高置信度预测。
    方案 C（bbox-train-C.py）— U-Net 分割 → 伪框：
        ResNet50 + 4 级解码器输出全分辨率热图；BCEWithLogitsLoss；
        验证时通过 cv2.connectedComponents 将 blob 转换为伪框进行 recall 评估。
-----------
43. 图像级二分类器前置过滤（rec_43）：
    两阶段训练：阶段一在全部图像上训练 ResNet50 图像级二分类器（has_lesion/
    no_lesion），阶段二仅用分类器预测为"有病灶"的图像训练 RetinaNet，
    将检测训练集的图像级负正比从 10.5:1 降低至约 0.25:1。
    新增参数：--clf-epochs / --clf-lr / --clf-pos-weight / --clf-threshold /
    --clf-save-path / --clf-checkpoint-path / --skip-clf-stage。
-----------
44. 架构彻底切换：patch 级滑窗二分类（rec_44）
    【完整实现在分支文件 bbox-train-D.py，方案已舍弃】
-----------
45. 方向 E：U-Net 全图分割检测（rec_45）
    【完整实现在分支文件 bbox-train-E.py，已删除】
-----------
46. 方向 F：U-Net Patch 训练检测（rec_46）
    【完整实现在分支文件 bbox-train-F.py，已删除】
    以 GT bbox 为中心裁取 256×256 patch，正负各 50%，正样本像素比 5–20%，
    消除全图 500:1 像素正负比导致的保守性坍缩。
    最终结果（rec_46_upd_3）：F2@0.9=0.4652，FP@0.9=599。
    已训练模型：models/bbox_resnet50.F.pth
-----------
47. 方向 G：两阶段检测 — Stage 2 ROI 分类器过滤 FP（rec_47）
    【完整实现在分支文件 bbox-train-G.py，已删除】
    在 Stage 1（U-Net heatmap → NMS → 候选框）之后添加 Stage 2（ResNet50 二分类器
    过滤 FP 候选），目标将 FP@0.9 从 ~600 降至 <200。
    根本瓶颈：Stage 1 在 stage1_threshold=0.2 时候选暴增至 5396，热图后处理管线
    是 FP 主要来源，无法从 Stage 2 端彻底消除；F+G 两阶段误差累积。
-----------
48. 方向 H：RetinaNet 全图直接检测（替代 F+G 两阶段流水线，rec_48）
    【完整实现在分支文件 bbox-train-H.py（现已合并为本文件）】
    切换动机：F+G 根本缺陷是热图后处理管线产生大量 FP，且 Stage 1 召回率天花板
    约 83%；H 直接用 RetinaNet + ResNet50-FPN 全图回归 box，消除 FP 来源；
    Focal Loss 内置极端不平衡处理；FPN P3–P7 覆盖 32–512px 尺度。
    最终结果（rec_48 upd_6）：best F2@0.3=0.2788，ep9，val GT boxes=251。
-----------
49. 方向 I：双侧对比检测（rec_49）
    【完整实现在分支文件 bbox-train-I.py，已删除】
    三通道 [primary, contra, |primary-contra|] 送入 RetinaNet，显式提供双侧不对称信号。
    实验结果：best F2@0.3=0.1901，ep3，early stop ep14。
    失败根因：像素级差分无法产生有效解剖不对称信号；两侧乳腺无配准，逐像素相减
    主要是结构噪声；diff 通道后期激活后反而干扰 primary 通道判断。
-----------
50. 方向 J：GlobalContextEncoder + ROI 重打分头（rec_50）
    【完整实现在分支文件 bbox-train-J.py，已删除】
    GlobalContextEncoder 提取全图上下文注入 FPN 各层；RoiRefinementHead 对候选框
    做 ROI Align → FC 校正置信度；推理时用几何均值 √(det×roi) 融合分数。
    实验结果：best F2@0.3=0.1571，ep23，TP@0.3 极端震荡（0~33）。
    失败根因：ROI 头每批正例极少，BCE loss 噪声大；几何均值公式过激进压低候选分数。
-----------
51. 方向 K：GlobalContextEncoder + RetinaNet（无 ROI 重打分）（rec_51）
    【完整实现在分支文件 bbox-train-K.py，已删除】
    移除 ROI 头，单变量验证全局上下文融合效果（K vs H 唯一变量）。
    实验结果：best F2@0.3=0.2213，ep4，early stop ep16，Val GT=251。
    失败根因：GlobalEncoder 随机初始化梯度扰动 FPN fusion conv；
    TP 存活率（@0.1→@0.3）从 H 的 45% 降至 30%。结论：全局融合本身无效。

─────────────────────────────────────────────────────────────────────────────
本文件改版历史（rec_48 起，RetinaNet + ResNet50-FPN 直接检测）
─────────────────────────────────────────────────────────────────────────────

rec_48（初版）
  - RetinaNet + ResNet50-FPN，全图 1024×512 直接检测
  - anchor sizes=(32,64,128,256,512)，aspect_ratios=(0.5,1.0,2.0)

upd_1（rec_48 修复与日志改进）
  [fix] labels=torch.zeros（前景类索引 0）替换 torch.ones
        → 修复 num_classes=1 时 CUDA index-out-of-bounds 崩溃
  [fix] resnet_fpn_backbone(backbone_name=...) 改为关键字参数
        → 消除 torchvision ≥0.13 的 DeprecationWarning
  [fix] 验证日志 [BestRecall] 标签改为 Recall@0.3 (ref)
        → 原标签误导性（实为固定阈值参考值，并非最大 recall）
  [new] 新增 --monitor-metric fbeta2_ref 选项
        → 使用固定 score=0.3 处的 F2 作为 checkpoint/early-stop 标准
        → 比 fbeta2（best across all thresh，始终落在 score=0.1）噪声更低

upd_2（rec_48_upd_2 数据增强与宽高比修复）
  根因分析（rec_48_upd_1 F2@0.3 在 epoch 11 封顶于 0.2176 的原因）：
    1. 宽高比扭曲：原图 1520×912（AR=1.667）直接 resize 到 1024×512（AR=2.0），
       水平/垂直缩放比不等（0.561 vs 0.674），导致圆形病灶被拉伸 20%，
       模型学到扭曲特征，验证泛化能力受限。历史实验（rec_34–47）均未遇此问题，
       因旧框架用 torchvision 内部等比 resize。本次属新发现盲点。
    2. 数据增强过弱：仅水平翻转 + 亮度抖动，无对比度/缩放多样性，
       过拟合在 epoch 12 后显现（训练 loss 持续下降，验证 F2 不升）。
  [fix] AR-preserving pad：读图后先将 H 方向 pad 到 W×target_AR（加黑边），
        再等比 resize 到 1024×512 → 修复 20% AR 扭曲，作用于训练和验证。
        (VinDr-Mammo 固定 1520×912 → pad H to 1824 → resize 1024×512)
  [new] --aug-contrast-range MIN MAX：对比度抖动（乘性偏移，以图像均值为轴），
        默认 1.0 1.0（禁用，向后兼容）。rec_48_upd_2 推荐 0.8 1.2。
  [new] --aug-scale-min S：随机 zoom-out 增强（等比缩小至 S–1.0 倍后居中 pad），
        默认 1.0（禁用，向后兼容）。rec_48_upd_2 推荐 0.85。

upd_3（epoch-seeded WeightedRandomSampler）
  根因分析（rec_48_upd_1/upd_2 均在 epoch 7-8 出现断崖的原因）：
    WeightedRandomSampler 在训练前创建一次，各 epoch 顺序消耗 PyTorch 全局随机
    状态（seed=42 固定）。epoch 7-8 恰好抽到"困难批次"（小病灶/低对比度正样本
    密集），梯度冲突导致 loss 反升、验证 F2@0.3 断崖。两次训练的断崖完全同步，
    证明是确定性批次组成问题，而非模型本身的收敛问题。
  [fix] 每个 epoch 用独立 generator（manual_seed = base_seed + epoch）重建
        WeightedRandomSampler 和 DataLoader，打破跨 epoch 的批次确定性。
        各 epoch 的批次组成仍然完全可复现（同 seed 多次运行结果一致），
        但不再依赖前序 epoch 的随机状态消耗，消除了 epoch 7 固定谷底。
  [new] --focal-alpha F：Focal Loss 前景权重 α（torchvision 默认 0.25）。
        α=0.25 使背景梯度是正样本的 3 倍，模型学会保守预测（score 集中在
        0.1-0.2），导致 F2@0.3 受限。提高 α 可上移置信度分布。
        默认 0.25（向后兼容）。rec_48_upd_3 推荐 0.4。
  [new] checkpoint meta 新增 best_fbeta2 / best_fbeta2_thresh / focal_alpha：
        训练时遍历阈值得到的最优 F2 及对应阈值一并保存进 .pth。
        部署时直接读取，无需手动调参：
          ckpt = torch.load("models/bbox_resnet50.pth")
          deploy_thresh = ckpt["meta"]["best_fbeta2_thresh"]
          # → 在此阈值处 ckpt["meta"]["best_fbeta2"] 最大

upd_4（纯 Mass 检测 baseline）
  根因分析（rec_48_upd_3 Recall@0.1=0.404 封顶的原因）：
    验证集 267 GT boxes 中，仅 Mass（56.6%）具备清晰可检测的视觉特征。
    其余类型存在结构性漏检问题：
      · Suspicious_Calcification（20.2%）：约 30% AR>2，anchor fg_iou<0.5 被
        标为 ignored；约 20% size<32px 完全超出 anchor 覆盖；其余训练信号不稳定。
      · Asymmetry / Architectural_Distortion（12.8%）：需双侧对比才能识别，
        单视图特征图无法与背景区分，即使 anchor 覆盖也无学习信号。
    理论上限 Recall@0.1 ≈ 55-65%（非 100%），当前 40.4% 已逼近该上限。
  [new] --lesion-types TYPES：逗号分隔的病灶类型名称，只将指定类型的 box 作为
        正样本 GT，其余 box 被忽略（图像若所有 box 均被过滤则视为负样本）。
        默认 None（使用全部 box，向后兼容）。upd_4 推荐 Mass。
  [revert] focal_alpha 从 0.4 退回 0.25：upd_3 的 α=0.4 使 Recall@0.1
        从 0.502 降至 0.404（-20%），虽然 TP@0.7 提升，但 upd_4 目标是
        建立 Mass 检测的 recall 上限，优先覆盖率。upd_4 推荐 α=0.25。

upd_5（按几何可检测性过滤 GT box，全类型训练）
  根因分析（upd_4 对比）：
    upd_4 仅训练 Mass，F2@0.3 从 0.2164 提升至 0.3005（+39%），
    Recall@0.1 从 40.4% 提升至 54.0%（per-box）。
    证明其他类型（尤其是 Suspicious_Calcification 的高 AR / 小尺寸子集）
    向训练注入了噪声信号。但剔除全部非 Mass 类型牺牲了临床覆盖面。
    upd_5 改为按"几何可检测性"精确过滤：保留所有类型中 anchor 实际能
    覆盖的 GT box，丢弃 anchor 物理上无法匹配的 box。
  [new] --min-box-side S：在 resized 图像空间（如 1024×512）中，最短边
        < S px 的 box 被丢弃。anchor 最小尺寸 = 32px，低于此的 box IoU=0，
        对训练无正向贡献，只增加 FN 计数。推荐 24.0（3/4 最小 anchor）。
  [new] --max-box-ar R：长宽比（max/min）> R 的 box 被丢弃。anchor 覆盖
        AR 0.5–2.0，AR>2.5 时最优 anchor fg_iou < 0.5，训练中标记为
        ignored（既不贡献正样本 loss 也不贡献负样本 loss）。推荐 3.0。
  [note] upd_5 不使用 --lesion-types，全类型训练，仅靠几何过滤精确控制
        噪声信号。预期 val GT boxes ≈ 240-250（vs 267 全量，150 Mass-only）。

upd_6（断崖感知 patience，解决 upd_5 提前停止问题）
  根因分析（upd_5 训练观察）：
    训练过程中 F2@0.3 呈现 ~5 epoch 周期振荡，峰值在 ep4/ep9/ep17。
    WeightedRandomSampler 每 epoch 重置采样种子，某些 epoch 开始时
    集中采样困难批次，导致 Focal Loss 梯度峰值 → 当 epoch 指标骤降。
    这些"断崖 epoch"连续触发 patience 计数，在 best ep9 之后 9 个 epoch
    内（ep10–ep18）没能突破，patience=10 在 ep19 触发早停，错过了
    ep22–23 潜在峰值（下一个周期峰）。
  [new] --cliff-patience-ratio R：若当前 epoch 的监控指标 < best×R，
        判定为"断崖 epoch"，不增加 patience 计数器。R=0 禁用（默认）。
        推荐 0.6：从实验数据可见，真正断崖的 metric 为 best 的 34–53%，
        普通徘徊在 60–96%，R=0.6 可以准确分离两类。
  [note] 断崖 epoch 的模型权重仍正常更新（不回滚），依赖下一个 epoch
        的采样顺序自然恢复。仅 patience 计数被跳过，不影响优化本身。
  [result] rec_48_upd_6（rec_to48_upd_6.txt）：best F2@0.3=0.2788 @ ep9，
        early stop ep23，Val GT=251。

─── 期间实验（均未超越基线 H upd_6）───────────────────────────────────────
  方向 I（rec_49）：帧间差分特征融合，best F2@0.3=0.1901 @ ep3。
    根因：像素级差分图引入结构性噪声，特征图中人工边界破坏 FPN 梯度。
  方向 J（rec_50）：在 RetinaNet 上追加 ROI 分类头（两阶段联合训练），
    best F2@0.3=0.1571 @ ep23。
    根因：ROI 头训练信号不稳定（几何均值评分过激进 + 联合训练梯度相互干扰）。
  方向 K（rec_51）：GlobalContextEncoder 注入 FPN 各层（无 ROI 头），
    best F2@0.3=0.2213 @ ep4，early stop ep16，Val GT=251。
    根因：GlobalEncoder 随机初始化，backward 梯度扰动 FPN fusion conv 权重；
    TP 存活率（@0.1→@0.3）从 H 的 45% 降至 K 的 30%，分数分布更压缩。
──────────────────────────────────────────────────────────────────────────────

* 以下序号标记为52，从upd_7开始计算。

upd_7（focal gamma 降低 + 1024×1024 分辨率，rec_52）
* 这就是本文件。
  根因分析（H ep9 vs K ep4 对比 + box 尺寸统计）：
    H 的两个独立失败模式：
      1. 低置信（73 个 TP 打分落在 [0.1, 0.3) 区间，存活率仅 45%）：
         根因是 focal_loss_gamma=2.0 过度压制中等置信预测，公式
         (1-p_t)^2.0 对 p_t∈[0.2,0.5] 的梯度衰减是 γ=1.0 时的 ~2.5 倍，
         导致 score 分布向 0 压缩。理论上限：若 133 个 @0.1 TP 均能 ≥0.3，
         F2@0.3 可达 ≈0.58（vs 当前 0.2788）。
      2. 检测缺失（118 FN @0.1）：当前 1024×512（scale=0.5614）下，
         31% 的 GT box 在 P3 上仅 4~8px（box 统计：min_side P5=35px,
         P50=86px），升至 1024×1024（scale=0.6737）box 增大 20%，
         改善小病灶特征表示。
  [new] --focal-gamma G：Focal Loss 指数 γ（torchvision 默认 2.0）。降低 γ
        减少对中等置信预测的梯度抑制，使 score 分布上移。默认 2.0（向后兼容）。
        rec_52 推荐 1.0。
  [new] 1024×1024 输入（--input-h 1024 --input-w 1024）：FPN 面积翻倍，
        但 4090 24GB 仍可使用 batch=4（激活值约 12~16GB，总显存约 14~17GB）。
  [fix] load_samples box 过滤 scale 计算：增加 input_h 参数，根据 AR 关系
        正确判断 pad-height vs pad-width，避免 1024×1024 下 scale 计算错误
        （旧逻辑在 1024×1024 下 scale=1024/912=1.123 而实际为 1024/1520=0.674）。
  [result] rec_52（2026-05-25，RTX 4090，7h51m，ep25 early stop）：
    best ep=12 | F2@0.3=0.2836 | @0.3: TP=62 FP=15 Rec=0.244 Prec=0.805
    @0.7: TP=21 FP=0 | @0.9: TP=6 FP=0 | Val GT=254（scale fix 多保留 3 个）
    目标 ≥0.35 未达到。分析：
      · γ=1.0 改善高置信区间：@0.7 TP ~2→21（+~10×），@0.9: 0→6，校准成功
      · F2@0.3 仅 +1.7%（0.2788→0.2836），根因是 score floor 未解决：
        ep12 共 113 个 @0.1 TP，其中 51 个（45%）卡在 [0.1, 0.3) 区间，
        γ=1.0 不足以把这段推过 0.3 阈值
      · 独立问题：141 个 GT box 完全未被 @0.1 检出（检测天花板），
        与 score calibration 无关，需架构或分辨率升级
      · 推理阈值改为 0.2 时 ep12 checkpoint 可得 F2@0.2=0.3225（+13.7%,
        TP=73 FP=43），无需重训练
    后续（52_upd_8 拟选方向）：
      γ=0.5 进一步降低 score floor；新增 --ref-score 参数使 ref_fbeta2_thresh
      可配置（当前硬编码第 927 行），用 F2@0.2 作为训练监控指标。

upd_8（--ref-score 参数 + γ=0.5）
  根因分析（rec_52 + 项目架构对齐）：
    rec_52 用 F2@0.3 作为 checkpoint 标准，而 README 明确 Stage 1 目标是
    "高召回率，漏检不可接受，误报可容忍"。F2@0.3（Prec=80%）与此不符：
      · ep12 @0.3 发送给 Stage 2：62 TP + 15 FP = 77 个候选框（漏掉 192 个 GT）
      · ep12 @0.1 发送给 Stage 2：113 TP + 157 FP = 270 个候选框（Stage 2 过滤 FP）
    ep4 @0.1 TP=132 对 Stage 2 比 ep12 @0.3 TP=62 更有价值，但被 F2@0.3 排除。
    同时 rec_52 的 score floor 问题（51 个 TP 卡在 [0.1, 0.3)）说明 γ=1.0 不够低。
  [new] --ref-score R：F2 和 Recall 监控指标使用的参考分数阈值（原 ref_fbeta2_thresh
        和 best_recall_thresh 硬编码 0.3）。必须为 0.1/0.2/0.3/0.5/0.7/0.9 之一。
        默认 0.3（向后兼容）。rec_52_upd_8 推荐 0.2，与 Stage 1 高召回部署阈值对齐。
  [change] --focal-gamma 0.5：γ 从 1.0 进一步降低，对 p_t=0.2 的梯度加权相比
        γ=1.0 多 2.2 倍，预期将更多 stuck TP（[0.1, 0.3) 区间）推过 0.2 阈值。
  [new] --amp：BF16 自动混合精度（torch.autocast），训练和验证均启用。
        RTX 4090 预计减少训练时长 35~50%，对精度无明显影响。默认关闭（向后兼容）。
  [new] --compile：torch.compile() 模型图编译（需 PyTorch ≥ 2.0），
        固定尺寸输入下每 epoch 减少约 15~25%。第一个 epoch 多花约 1 min 编译。
        默认关闭（向后兼容）。
  [result] rec_52_upd_8（rec_to52_upd_8.txt）：best F2@0.2=0.3223 @ ep12，
        val GT=254。
        @0.2: TP=72 FP=29 Rec=0.283 Prec=0.713 | @0.3: TP=55 FP=7 Rec=0.217 Prec=0.887
        @0.7: TP=19 FP=0 | @0.9: TP=8 FP=0
        结论：F2@0.2 与 upd_7（0.3225）持平；F2@0.3=0.2551 低于 upd_7（0.2836）。
        γ=0.5 未能进一步改善 score calibration；upd_7（γ=1.0）仍是 F2@0.3 最佳。

"""

from __future__ import annotations

import os
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import atexit
import datetime
import random
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    _TORCHVISION_DET_OK = True
except ImportError:
    _TORCHVISION_DET_OK = False

try:
    from torchvision.models import resnet50 as _resnet50
    try:
        from torchvision.models import ResNet50_Weights as _ResNet50Weights
        _IMAGENET_WEIGHTS: Any = _ResNet50Weights.DEFAULT
    except ImportError:
        _ResNet50Weights = None  # type: ignore[assignment]
        _IMAGENET_WEIGHTS = True  # pretrained=True fallback
except ImportError:
    _resnet50 = None  # type: ignore[assignment]
    _IMAGENET_WEIGHTS = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8 with CLAHE for grayscale."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while len(order) > 0:
        idx = int(order[0])
        keep.append(idx)
        if len(order) == 1:
            break
        rest = order[1:]
        bx = boxes[idx]
        br = boxes[rest]
        ix1 = np.maximum(bx[0], br[:, 0])
        iy1 = np.maximum(bx[1], br[:, 1])
        ix2 = np.minimum(bx[2], br[:, 2])
        iy2 = np.minimum(bx[3], br[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_bx = (bx[2] - bx[0]) * (bx[3] - bx[1])
        area_br = (br[:, 2] - br[:, 0]) * (br[:, 3] - br[:, 1])
        union = area_bx + area_br - inter
        iou = inter / np.maximum(union, 1e-6)
        order = rest[iou < iou_thresh]
    return keep


def compute_iou_matches(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    matched_gt = [False] * int(gt_boxes.shape[0])
    matched_pred = [False] * int(pred_boxes.shape[0])
    for pi in range(pred_boxes.shape[0]):
        best_iou = 0.0
        best_gi = -1
        px1, py1, px2, py2 = float(pred_boxes[pi, 0]), float(pred_boxes[pi, 1]), \
                              float(pred_boxes[pi, 2]), float(pred_boxes[pi, 3])
        for gi in range(gt_boxes.shape[0]):
            if matched_gt[gi]:
                continue
            gx1, gy1, gx2, gy2 = float(gt_boxes[gi, 0]), float(gt_boxes[gi, 1]), \
                                  float(gt_boxes[gi, 2]), float(gt_boxes[gi, 3])
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt[best_gi] = True
            matched_pred[pi] = True
    tp = sum(matched_pred)
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - sum(matched_gt)
    return int(tp), int(fp), int(fn)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray          # (N, 4) xyxy in original image coords
    orig_size: Tuple[float, float]  # (H, W)


def load_samples(
    csv_path: Path,
    images_root: Path,
    split_name: str,
    lesion_types: Optional[List[str]] = None,
    min_box_side: float = 0.0,
    max_box_ar: float = float("inf"),
    input_w: int = 512,
    input_h: int = 1024,
) -> List[Sample]:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    samples: List[Sample] = []
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        first = group.iloc[0]  # read orig_h/w before filtering
        if lesion_types:
            type_mask = pd.Series(False, index=group.index)
            for lt in lesion_types:
                if lt in group.columns:
                    type_mask = type_mask | (group[lt] == 1)
            group = group[type_mask]
        valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
        boxes: np.ndarray = valid.to_numpy(dtype=np.float32) if not valid.empty else np.zeros((0, 4), dtype=np.float32)
        image_path = images_root / str(patient_id) / f"{image_id}"

        if boxes.size > 0:
            invalid = int(np.sum((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])))
            if invalid > 0:
                print(f"[Warning] Found {invalid} invalid boxes in {image_path}")

        # Detectability filter (in resized-image-space coordinates)
        if boxes.size > 0 and (min_box_side > 0.0 or max_box_ar < float("inf")):
            orig_w_val = float(first["width"]) if pd.notna(first["width"]) else float(input_w)
            orig_h_val = float(first["height"]) if pd.notna(first["height"]) else float(input_h)
            # AR-preserving scale: determine which dimension is padded, use the
            # non-padded dimension to compute scale (matches Dataset preprocessing).
            target_ar = float(input_h) / max(float(input_w), 1.0)
            actual_ar = orig_h_val / max(orig_w_val, 1.0)
            if actual_ar <= target_ar:  # pad height → x-scale = input_w / orig_w
                scale = float(input_w) / max(orig_w_val, 1.0)
            else:                       # pad width  → y-scale = input_h / orig_h
                scale = float(input_h) / max(orig_h_val, 1.0)
            bw = (boxes[:, 2] - boxes[:, 0]) * scale
            bh = (boxes[:, 3] - boxes[:, 1]) * scale
            min_sides = np.minimum(bw, bh)
            ars = np.maximum(bw, bh) / np.maximum(min_sides, 1e-3)
            keep = np.ones(len(boxes), dtype=bool)
            if min_box_side > 0.0:
                keep &= (min_sides >= min_box_side)
            if max_box_ar < float("inf"):
                keep &= (ars <= max_box_ar)
            boxes = boxes[keep]

        orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
        orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

        samples.append(Sample(
            patient_id=str(patient_id),
            image_id=str(image_id),
            image_path=image_path,
            boxes=boxes,
            orig_size=(orig_h, orig_w),
        ))

    return samples


def patient_level_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    patients = sorted({s.patient_id for s in samples})  # sorted → deterministic order
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    train_idx = [i for i, s in enumerate(samples) if s.patient_id not in val_patients]
    val_idx = [i for i, s in enumerate(samples) if s.patient_id in val_patients]
    return train_idx, val_idx


# =============================================================================
# Dataset
# =============================================================================

class DetectionDataset(Dataset):
    """Full-image detection dataset for torchvision RetinaNet.

    Each item is (image_tensor [3, H, W], target_dict) where target_dict
    contains 'boxes' [N, 4] (xyxy) and 'labels' [N] (all 1 = lesion).
    Negative images return empty boxes/labels.
    """

    def __init__(
        self,
        samples: List[Sample],
        indices: List[int],
        input_h: int,
        input_w: int,
        augment: bool = False,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 0.2,
        aug_contrast_min: float = 1.0,
        aug_contrast_max: float = 1.0,
        aug_scale_min: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_brightness_delta = aug_brightness_delta
        self.aug_contrast_min = aug_contrast_min
        self.aug_contrast_max = aug_contrast_max
        self.aug_scale_min = aug_scale_min
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[self.indices[idx]]

        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            target: Dict[str, torch.Tensor] = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            return img_t, target

        orig_h, orig_w = img.shape[:2]
        boxes = sample.boxes.copy()

        # Clip and filter degenerate boxes
        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # AR-preserving pad: make H/W match target AR before resize to eliminate distortion.
        # VinDr-Mammo images are 1520×912 (AR=1.667); target 1024×512 (AR=2.0) → pad H.
        target_ar = self.input_h / max(self.input_w, 1)
        actual_ar = orig_h / max(orig_w, 1)
        if actual_ar < target_ar - 1e-6:  # image too wide → pad height
            padded_h = int(round(orig_w * target_ar))
            pad_top = (padded_h - orig_h) // 2
            pad_bottom = padded_h - orig_h - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 1] += pad_top
                boxes[:, 3] += pad_top
            orig_h = padded_h
        elif actual_ar > target_ar + 1e-6:  # image too tall → pad width
            padded_w = int(round(orig_h / target_ar))
            pad_left = (padded_w - orig_w) // 2
            pad_right = padded_w - orig_w - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 0] += pad_left
                boxes[:, 2] += pad_left
            orig_w = padded_w

        # Resize image
        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Scale boxes to input_h×input_w space
        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        if boxes.size > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            # Re-filter after scaling
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # Augmentation
        if self.augment:
            if self.rng.random() < self.aug_hflip_prob:
                img_resized = img_resized[:, ::-1, :].copy()
                if boxes.size > 0:
                    old_x1 = boxes[:, 0].copy()
                    old_x2 = boxes[:, 2].copy()
                    boxes[:, 0] = self.input_w - old_x2
                    boxes[:, 2] = self.input_w - old_x1
            if self.aug_brightness_delta > 0:
                delta = (self.rng.random() * 2 - 1) * self.aug_brightness_delta * 255
                img_resized = np.clip(img_resized.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            if self.aug_contrast_max > self.aug_contrast_min + 1e-6:
                factor = self.aug_contrast_min + self.rng.random() * (self.aug_contrast_max - self.aug_contrast_min)
                mean_val = float(img_resized.mean())
                img_resized = np.clip(
                    mean_val + (img_resized.astype(np.float32) - mean_val) * factor, 0, 255
                ).astype(np.uint8)
            if self.aug_scale_min < 1.0 - 1e-6:
                scale = self.aug_scale_min + self.rng.random() * (1.0 - self.aug_scale_min)
                if scale < 1.0 - 1e-6:
                    scaled_h = max(int(self.input_h * scale), 1)
                    scaled_w = max(int(self.input_w * scale), 1)
                    img_small = cv2.resize(img_resized, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    pad_top_zo = (self.input_h - scaled_h) // 2
                    pad_left_zo = (self.input_w - scaled_w) // 2
                    img_zoomed = np.zeros_like(img_resized)
                    img_zoomed[pad_top_zo:pad_top_zo + scaled_h, pad_left_zo:pad_left_zo + scaled_w] = img_small
                    img_resized = img_zoomed
                    if boxes.size > 0:
                        boxes[:, 0] = boxes[:, 0] * scale + pad_left_zo
                        boxes[:, 2] = boxes[:, 2] * scale + pad_left_zo
                        boxes[:, 1] = boxes[:, 1] * scale + pad_top_zo
                        boxes[:, 3] = boxes[:, 3] * scale + pad_top_zo
                        keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                        boxes = boxes[keep]

        img_t = image_to_tensor(img_resized)  # [3, H, W] float32 in [0, 1]

        if boxes.size > 0:
            target = {
                "boxes": torch.from_numpy(boxes.astype(np.float32)),
                "labels": torch.zeros(boxes.shape[0], dtype=torch.int64),  # class 0 = foreground (num_classes=1)
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        return img_t, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def make_oversampling_weights(
    samples: List[Sample],
    indices: List[int],
    pos_oversample_factor: float = 4.0,
) -> torch.Tensor:
    """Assign higher weight to positive-sample images for WeightedRandomSampler."""
    weights = []
    for i in indices:
        weights.append(pos_oversample_factor if samples[i].boxes.shape[0] > 0 else 1.0)
    return torch.DoubleTensor(weights)


# =============================================================================
# Model
# =============================================================================

def build_retinanet(
    medical_backbone_path: Optional[str] = None,
    num_classes: int = 1,
    anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 5,
    min_size: int = 512,
    max_size: int = 1024,
    trainable_backbone_layers: int = 5,
    nms_thresh: float = 0.3,
    score_thresh: float = 0.05,
    detections_per_img: int = 500,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> "RetinaNet":
    """Build a RetinaNet with ResNet50-FPN backbone.

    Loads ImageNet weights first, then overrides with RadImageNet if provided.
    conv1 weights are averaged across channels for grayscale-as-3channel input.
    """
    if not _TORCHVISION_DET_OK:
        raise RuntimeError(
            "torchvision detection module not available. "
            "Requires torchvision >= 0.11 with detection support."
        )

    # Build ResNet50-FPN backbone
    try:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=_IMAGENET_WEIGHTS,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone.")
    except TypeError:
        # Older torchvision API
        backbone = resnet_fpn_backbone(  # type: ignore[call-arg]
            backbone_name="resnet50",
            pretrained=True,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone (legacy API).")

    # Override with RadImageNet if provided
    if medical_backbone_path is not None:
        _idx_to_resnet = {
            "0": "conv1", "1": "bn1",
            "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
        }
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            raw_sd = (
                ckpt.get("state_dict", ckpt.get("model", ckpt))
                if isinstance(ckpt, dict)
                else ckpt
            )
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            result = backbone.body.load_state_dict(stripped, strict=False)
            missing = len(result.missing_keys) if result is not None else "?"
            unexpected = len(result.unexpected_keys) if result is not None else "?"
            print(f"[Info] Loaded RadImageNet backbone (missing={missing}, unexpected={unexpected}).")
        except Exception as exc:
            print(f"[Warning] Could not load RadImageNet backbone ({exc}). Keeping ImageNet weights.")

    # Adapt conv1 for grayscale-as-3channel (average over input channels)
    try:
        with torch.no_grad():
            mean_w = backbone.body.conv1.weight.mean(dim=1, keepdim=True)
            backbone.body.conv1.weight.copy_(mean_w.expand_as(backbone.body.conv1.weight))
        print("[Info] conv1 weights averaged for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1.")

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )

    # Override focal loss alpha and gamma (torchvision defaults: alpha=0.25, gamma=2.0)
    try:
        model.head.classification_head.focal_loss_alpha = float(focal_alpha)
        print(f"[Info] Focal loss alpha set to {focal_alpha}.")
    except AttributeError:
        print(f"[Warning] Could not set focal_loss_alpha on this torchvision version.")
    try:
        model.head.classification_head.focal_loss_gamma = float(focal_gamma)
        print(f"[Info] Focal loss gamma set to {focal_gamma}.")
    except AttributeError:
        print(f"[Warning] Could not set focal_loss_gamma on this torchvision version.")

    return model


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: "RetinaNet",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int = 1,
    disable_tqdm: bool = False,
    use_amp: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())  # type: ignore[arg-type]

        if not torch.isfinite(losses):
            optimizer.zero_grad(set_to_none=True)
            continue

        (losses / float(accumulation_steps)).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(losses.item())
        count += 1
        if not disable_tqdm:
            pbar.set_postfix(loss=f"{losses.item():.4f}")  # type: ignore[union-attr]

    return running_loss / max(count, 1)


# =============================================================================
# Validation
# =============================================================================

def validate(
    model: "RetinaNet",
    samples: List[Sample],
    val_indices: List[int],
    device: torch.device,
    input_h: int,
    input_w: int,
    iou_threshold: float,
    collection_score_thresh: float = 0.01,
    nms_thresh: float = 0.3,
    epoch: int = 0,
    epochs: int = 1,
    disable_tqdm: bool = False,
    ref_score: float = 0.3,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Validate with full-image RetinaNet inference at multiple score thresholds."""
    model.eval()

    # Unwrap compiled model for attribute access (torch.compile wraps in OptimizedModule)
    _m = getattr(model, "_orig_mod", model)

    # Temporarily lower score_thresh to collect all candidate boxes
    orig_score_thresh = _m.score_thresh
    orig_nms_thresh = _m.nms_thresh
    orig_det_per_img = _m.detections_per_img
    _m.score_thresh = collection_score_thresh
    _m.nms_thresh = nms_thresh
    _m.detections_per_img = 1000

    score_thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    if ref_score not in score_thresholds:
        raise ValueError(f"--ref-score {ref_score} must be one of {score_thresholds}")
    stats: Dict[float, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in score_thresholds
    }
    total_gt_boxes = 0

    pbar = tqdm(val_indices, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for sample_idx in pbar:
            sample = samples[sample_idx]
            try:
                img = normalize_image(read_image_unicode(sample.image_path))
            except FileNotFoundError:
                continue

            orig_h, orig_w = img.shape[:2]
            gt_boxes = sample.boxes.astype(np.float32).copy()

            # AR-preserving pad (must match training preprocessing)
            target_ar = input_h / max(input_w, 1)
            actual_ar = orig_h / max(orig_w, 1)
            if actual_ar < target_ar - 1e-6:  # pad height
                padded_h = int(round(orig_w * target_ar))
                pad_top = (padded_h - orig_h) // 2
                pad_bottom = padded_h - orig_h - pad_top
                img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
                if gt_boxes.size > 0:
                    gt_boxes[:, 1] += pad_top
                    gt_boxes[:, 3] += pad_top
                orig_h = padded_h
            elif actual_ar > target_ar + 1e-6:  # pad width
                padded_w = int(round(orig_h / target_ar))
                pad_left = (padded_w - orig_w) // 2
                pad_right = padded_w - orig_w - pad_left
                img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
                if gt_boxes.size > 0:
                    gt_boxes[:, 0] += pad_left
                    gt_boxes[:, 2] += pad_left
                orig_w = padded_w

            # Scale GT boxes to input_h×input_w coordinate space
            scale_x = input_w / max(orig_w, 1)
            scale_y = input_h / max(orig_h, 1)
            if gt_boxes.size > 0:
                gt_boxes[:, 0] *= scale_x
                gt_boxes[:, 2] *= scale_x
                gt_boxes[:, 1] *= scale_y
                gt_boxes[:, 3] *= scale_y
                keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
                gt_boxes = gt_boxes[keep]

            img_resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            img_t = image_to_tensor(img_resized).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model([img_t])  # List[Dict] with 'boxes', 'scores', 'labels'
            pred_boxes = outputs[0]["boxes"].cpu().float().numpy()    # (K, 4) xyxy
            pred_scores = outputs[0]["scores"].cpu().float().numpy()  # (K,)

            for thresh in score_thresholds:
                mask = pred_scores >= thresh
                filtered_boxes = pred_boxes[mask]
                tp, fp, fn = compute_iou_matches(filtered_boxes, gt_boxes, iou_threshold)
                stats[thresh]["tp"] += tp
                stats[thresh]["fp"] += fp
                stats[thresh]["fn"] += fn

            total_gt_boxes += int(gt_boxes.shape[0]) if gt_boxes.size > 0 else 0

    # Restore model thresholds
    _m.score_thresh = orig_score_thresh
    _m.nms_thresh = orig_nms_thresh
    _m.detections_per_img = orig_det_per_img

    # Compute metrics
    f1_per_thresh: Dict[float, float] = {}
    recall_per_thresh: Dict[float, float] = {}
    fbeta2_per_thresh: Dict[float, float] = {}
    for thresh in score_thresholds:
        tp = stats[thresh]["tp"]
        fp = stats[thresh]["fp"]
        fn = stats[thresh]["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fbeta2 = (1 + 4) * prec * rec / max(4 * prec + rec, 1e-9)
        f1_per_thresh[thresh] = f1
        recall_per_thresh[thresh] = rec
        fbeta2_per_thresh[thresh] = fbeta2

    best_f1_thresh = max(f1_per_thresh, key=lambda t: f1_per_thresh[t])
    best_f1 = f1_per_thresh[best_f1_thresh]
    best_recall_thresh: float = ref_score
    best_recall = recall_per_thresh[best_recall_thresh]
    best_fbeta2_thresh = max(fbeta2_per_thresh, key=lambda t: fbeta2_per_thresh[t])
    best_fbeta2 = fbeta2_per_thresh[best_fbeta2_thresh]
    ref_fbeta2_thresh: float = ref_score
    ref_fbeta2 = fbeta2_per_thresh[ref_fbeta2_thresh]

    parts = []
    for thresh in score_thresholds:
        tp = stats[thresh]["tp"]
        fp = stats[thresh]["fp"]
        fn = stats[thresh]["fn"]
        parts.append(
            f"@{thresh}: TP={tp} FP={fp} FN={fn} "
            f"Rec={recall_per_thresh[thresh]:.3f} F1={f1_per_thresh[thresh]:.4f} F2={fbeta2_per_thresh[thresh]:.4f}"
        )
    print(f"  [Val] GT_boxes={total_gt_boxes} | {' | '.join(parts)}")
    print(
        f"  [BestF1] F1={best_f1:.4f} @ score={best_f1_thresh} | "
        f"Recall@{best_recall_thresh}={best_recall:.4f} (ref) | "
        f"[BestFbeta2] F2={best_fbeta2:.4f} @ score={best_fbeta2_thresh} | "
        f"F2@{ref_fbeta2_thresh}={ref_fbeta2:.4f} (ref)"
    )

    result: Dict[str, float] = {
        "best_f1": float(best_f1),
        "best_f1_thresh": float(best_f1_thresh),
        "best_recall": float(best_recall),
        "best_recall_thresh": float(best_recall_thresh),
        "best_fbeta2": float(best_fbeta2),
        "best_fbeta2_thresh": float(best_fbeta2_thresh),
        "ref_fbeta2": float(ref_fbeta2),
        "ref_fbeta2_thresh": float(ref_fbeta2_thresh),
        "val_gt_boxes": float(total_gt_boxes),
    }
    for thresh in score_thresholds:
        result[f"tp@{thresh}"] = float(stats[thresh]["tp"])
        result[f"fp@{thresh}"] = float(stats[thresh]["fp"])
        result[f"fn@{thresh}"] = float(stats[thresh]["fn"])
    return result


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    save_path: Path,
    model: "RetinaNet",
    meta: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, save_path)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RetinaNet-ResNet50-FPN detection model for VinDr lesion detection (Direction H)."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size. Default 4 for full 1024×512 images on a 24GB GPU.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1,
                        help="LR multiplier for pretrained ResNet50 body. "
                             "FPN + head use full --lr.")
    parser.add_argument("--input-h", type=int, default=1024,
                        help="Resize height for model input.")
    parser.add_argument("--input-w", type=int, default=512,
                        help="Resize width for model input.")
    parser.add_argument("--val-iou-threshold", type=float, default=0.1,
                        help="IoU threshold for matching predicted boxes to GT during validation.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cliff-patience-ratio", type=float, default=0.0,
                        help="Cliff-aware patience: if the monitored metric for an epoch falls "
                             "below (best × cliff_patience_ratio), classify that epoch as a "
                             "'cliff' and do NOT increment the patience counter. "
                             "0 = disabled (default). Recommended: 0.6 — isolates genuine "
                             "cliff drops (< 60%% of best) from normal plateau oscillation, "
                             "preventing premature early-stopping due to sampler-induced spikes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps. Use 2–4 if GPU memory is tight.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training augmentation (hflip, brightness jitter).")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2)
    parser.add_argument("--aug-contrast-range", type=float, nargs=2, default=[1.0, 1.0],
                        metavar=("MIN", "MAX"),
                        help="Contrast jitter range (multiplicative factor around image mean). "
                             "1.0 1.0 = disabled (default). Recommended: 0.8 1.2.")
    parser.add_argument("--aug-scale-min", type=float, default=1.0,
                        help="Minimum zoom-out scale for random scale augmentation. "
                             "1.0 = disabled (default). Recommended: 0.85.")
    parser.add_argument("--medical-backbone-path", type=Path, default=None,
                        help="Path to RadImageNet ResNet50 checkpoint (.pt). "
                             "If None, uses ImageNet weights only.")
    parser.add_argument("--pos-oversample-factor", type=float, default=4.0,
                        help="Weight multiplier for positive (lesion) images in the training sampler. "
                             "4.0 → positive images sampled ~4× more than negatives. "
                             "Compensates for the 7%% natural positive image ratio.")
    parser.add_argument("--anchor-sizes", type=str, default="32,64,128,256,512",
                        help="Comma-separated anchor sizes (one per FPN level). "
                             "Default: 32,64,128,256,512.")
    parser.add_argument("--nms-thresh", type=float, default=0.3,
                        help="NMS IoU threshold applied during inference. Default 0.3.")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Minimum score threshold for reported detections. Default 0.05.")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal Loss foreground weight alpha. Higher values increase "
                             "positive-sample gradient weight, shifting score distribution "
                             "upward. torchvision default 0.25; try 0.4 to reduce "
                             "confidence suppression. Default 0.25 (backward-compatible).")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal Loss modulating exponent gamma. Lower values reduce "
                             "gradient suppression of medium-confidence predictions, shifting "
                             "score distribution upward. torchvision default 2.0; try 1.0 to "
                             "address score calibration bottleneck. Default 2.0 (backward-compatible).")
    parser.add_argument("--ref-score", type=float, default=0.3,
                        help="Reference score threshold used by fbeta2_ref and recall monitor "
                             "metrics. Must be one of the evaluated thresholds: "
                             "0.1, 0.2, 0.3, 0.5, 0.7, 0.9. Default 0.3 (backward-compatible). "
                             "Use 0.2 to align with Stage-1 high-recall deployment.")
    parser.add_argument("--monitor-metric", type=str, default="fbeta2",
                        choices=["f1", "recall", "fbeta2", "fbeta2_ref"],
                        help="Metric to monitor for checkpoint saving and early stopping. "
                             "fbeta2_ref uses F2 at the fixed reference threshold (--ref-score), "
                             "which is more stable than fbeta2 (best across all thresholds).")
    parser.add_argument("--hide-progress-bar", action="store_true")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (BF16) training and validation. "
                             "Requires CUDA. Reduces training time ~35-50%% on Ampere/Ada GPUs "
                             "(e.g. RTX 4090) with negligible quality impact. Default: disabled "
                             "(backward-compatible).")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile() to the model before training. "
                             "Requires PyTorch >= 2.0. First epoch is slower (compilation). "
                             "Reduces per-epoch time ~15-25%% for fixed-size inputs. "
                             "Default: disabled (backward-compatible).")
    parser.add_argument("--lesion-types", type=str, default=None,
                        help="Comma-separated lesion type names to keep as positive GT boxes "
                             "(e.g. 'Mass' or 'Mass,Focal_Asymmetry'). Images whose boxes are "
                             "all filtered out become negative samples. Default: None (use all "
                             "annotated boxes, backward compatible).")
    parser.add_argument("--min-box-side", type=float, default=0.0,
                        help="Minimum box shortest side in resized-image space (pixels). "
                             "Boxes below this are dropped as GT (no anchor can match them). "
                             "0 = no filter (default). Recommended for VinDr 1024\u00d7512: 24.0 "
                             "(removes boxes smaller than 3/4 of the minimum 32px anchor).")
    parser.add_argument("--max-box-ar", type=float, default=float("inf"),
                        help="Maximum box aspect ratio (max_side / min_side) to keep as positive "
                             "GT. Boxes with AR > this have fg_iou < 0.5 with all anchors and "
                             "produce no useful training signal. inf = no filter (default). "
                             "Recommended: 3.0 (anchors cover AR 0.5-2.0; 3.0 adds 50%% margin).")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    repo_root = repo_root_from_file()
    csv_path = args.csv_path or repo_root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = args.images_root or repo_root / "data" / "processed" / "images_png"
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.pth"

    print(f"Start time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    all_samples = load_samples(csv_path, images_root, split_name="training",
                               lesion_types=[t.strip() for t in args.lesion_types.split(",")]
                               if args.lesion_types else None,
                               min_box_side=float(args.min_box_side),
                               max_box_ar=float(args.max_box_ar),
                               input_w=int(args.input_w),
                               input_h=int(args.input_h))
    print(f"Total samples: {len(all_samples)}")
    if args.lesion_types:
        print(f"Lesion type filter: {args.lesion_types}")
    if float(args.min_box_side) > 0.0 or float(args.max_box_ar) < float("inf"):
        print(f"Box detectability filter: min_side≥{args.min_box_side:.1f}px, max_AR≤{args.max_box_ar:.1f}")

    train_idx, val_idx = patient_level_split(all_samples, val_ratio=0.15, seed=int(args.seed))
    val_pos_idx = [i for i in val_idx if all_samples[i].boxes.shape[0] > 0]

    n_train_pos = sum(1 for i in train_idx if all_samples[i].boxes.shape[0] > 0)
    n_train_neg = len(train_idx) - n_train_pos
    val_gt_total = sum(all_samples[i].boxes.shape[0] for i in val_pos_idx)
    print(f"Train: {len(train_idx)} images (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val: {len(val_idx)} images | Val positive images: {len(val_pos_idx)} | Val GT boxes: {val_gt_total}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input size: {args.input_h}×{args.input_w}")

    encoder_lr = float(args.lr) * float(args.encoder_lr_multiplier)
    head_lr = float(args.lr)
    print(
        f"Encoder LR: {encoder_lr:.2e} | Head/FPN LR: {head_lr:.2e} | "
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | Patience: {args.patience}"
    )
    print(f"Pos oversample factor: {args.pos_oversample_factor:.1f}")
    print(f"Monitor metric: {args.monitor_metric}")

    # Parse anchor sizes
    anchor_size_vals = [int(s.strip()) for s in str(args.anchor_sizes).split(",")]
    anchor_sizes = tuple((s,) for s in anchor_size_vals)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size_vals)
    print(f"Anchor sizes: {anchor_sizes} | Aspect ratios: (0.5, 1.0, 2.0) per level")

    # Build model
    model = build_retinanet(
        medical_backbone_path=(
            str(args.medical_backbone_path) if args.medical_backbone_path else None
        ),
        num_classes=1,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        min_size=min(int(args.input_w), int(args.input_h)),
        max_size=max(int(args.input_w), int(args.input_h)),
        trainable_backbone_layers=5,
        nms_thresh=float(args.nms_thresh),
        score_thresh=float(args.score_thresh),
        detections_per_img=500,
        focal_alpha=float(args.focal_alpha),
        focal_gamma=float(args.focal_gamma),
    )
    model.to(device)

    if getattr(args, "compile", False):
        print("[Info] Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model)  # type: ignore[assignment]

    # Differential LR: encoder body vs FPN + head
    body_param_ids = {id(p) for p in model.backbone.body.parameters()}
    body_params = list(model.backbone.body.parameters())
    other_params = [p for p in model.parameters() if id(p) not in body_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": encoder_lr},
            {"params": other_params, "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.01
    )

    # Build training dataset (constant across epochs, augmentation is stochastic)
    train_dataset = DetectionDataset(
        samples=all_samples,
        indices=train_idx,
        input_h=int(args.input_h),
        input_w=int(args.input_w),
        augment=bool(args.augment),
        aug_hflip_prob=float(args.aug_hflip_prob),
        aug_brightness_delta=float(args.aug_brightness_delta),
        aug_contrast_min=float(args.aug_contrast_range[0]),
        aug_contrast_max=float(args.aug_contrast_range[1]),
        aug_scale_min=float(args.aug_scale_min),
        seed=int(args.seed),
    )

    # Weighted sampler: oversample positive images
    sampler_weights = make_oversampling_weights(
        all_samples, train_idx, pos_oversample_factor=float(args.pos_oversample_factor)
    )
    # train_sampler and train_loader are rebuilt each epoch inside the loop
    # (epoch-seeded generator) to break cross-epoch batch determinism.

    # Training state
    best_metric = 0.0
    best_epoch = 0
    no_improve = 0
    monitor_metric_name = str(args.monitor_metric)

    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None) -> None:
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
        print(f"\n[Exit] Reason: {reason or 'normal'}")
        print(f"Best {monitor_metric_name}={best_metric:.4f} at epoch {best_epoch}")

    atexit.register(_on_exit, "atexit")

    def _sig_handler(signum: int, frame: Any) -> None:
        _on_exit(f"signal {signum}")
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except (OSError, ValueError):
            pass

    # Training loop
    for epoch in range(int(args.epochs)):
        print(f"\n{'─' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'─' * 72}")

        # Rebuild sampler with an epoch-specific seed so each epoch draws an
        # independent batch sequence (still reproducible: same seed → same run).
        _epoch_gen = torch.Generator()
        _epoch_gen.manual_seed(int(args.seed) + epoch)
        train_sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_idx),
            replacement=True,
            generator=_epoch_gen,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            sampler=train_sampler,
            num_workers=int(args.num_workers),
            collate_fn=detection_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=int(args.epochs),
            accumulation_steps=int(args.accumulation_steps),
            disable_tqdm=bool(args.hide_progress_bar),
            use_amp=bool(getattr(args, "amp", False)),
        )
        lr_scheduler.step()

        val_metrics = validate(
            model=model,
            samples=all_samples,
            val_indices=val_pos_idx,
            device=device,
            input_h=int(args.input_h),
            input_w=int(args.input_w),
            iou_threshold=float(args.val_iou_threshold),
            collection_score_thresh=0.01,
            nms_thresh=float(args.nms_thresh),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=bool(args.hide_progress_bar),
            ref_score=float(args.ref_score),
            use_amp=bool(getattr(args, "amp", False)),
        )

        if monitor_metric_name == "recall":
            cur_monitor = float(val_metrics.get("best_recall", 0.0))
            monitor_thresh = float(val_metrics.get("best_recall_thresh", 0.3))
        elif monitor_metric_name == "fbeta2":
            cur_monitor = float(val_metrics.get("best_fbeta2", 0.0))
            monitor_thresh = float(val_metrics.get("best_fbeta2_thresh", 0.3))
        elif monitor_metric_name == "fbeta2_ref":
            cur_monitor = float(val_metrics.get("ref_fbeta2", 0.0))
            monitor_thresh = float(val_metrics.get("ref_fbeta2_thresh", 0.3))
        else:
            cur_monitor = float(val_metrics.get("best_f1", 0.0))
            monitor_thresh = float(val_metrics.get("best_f1_thresh", 0.3))

        report_tp = int(val_metrics.get(f"tp@{monitor_thresh}", 0))
        report_fp = int(val_metrics.get(f"fp@{monitor_thresh}", 0))
        report_fn = int(val_metrics.get(f"fn@{monitor_thresh}", 0))
        report_recall = report_tp / max(report_tp + report_fn, 1)
        report_prec = report_tp / max(report_tp + report_fp, 1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"{monitor_metric_name}={cur_monitor:.4f} @ score={monitor_thresh:.1f} "
            f"(TP={report_tp} FP={report_fp} FN={report_fn} "
            f"Recall={report_recall:.3f} Prec={report_prec:.3f}) | "
            f"lr={lr_scheduler.get_last_lr()[-1]:.6f}"
        )

        improved = cur_monitor > best_metric + float(args.min_delta)
        cliff_ratio = float(args.cliff_patience_ratio)
        is_cliff = (
            cliff_ratio > 0.0
            and best_metric > 0.0
            and cur_monitor < best_metric * cliff_ratio
        )
        if improved:
            best_metric = cur_monitor
            no_improve = 0
            best_epoch = epoch + 1
            save_checkpoint(
                save_path=save_path,
                model=model,
                meta={
                    "epoch": epoch + 1,
                    monitor_metric_name: cur_monitor,
                    "monitor_thresh": monitor_thresh,
                    # Also save the sweep-optimal F2 threshold for deployment use.
                    # When monitor_metric is fbeta2_ref (fixed 0.3), monitor_thresh=0.3
                    # but best_fbeta2_thresh gives the threshold that maximises F2.
                    "best_fbeta2": float(val_metrics.get("best_fbeta2", 0.0)),
                    "best_fbeta2_thresh": float(val_metrics.get("best_fbeta2_thresh", 0.3)),
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                    "anchor_sizes": str(args.anchor_sizes),
                    "nms_thresh": float(args.nms_thresh),
                    "focal_alpha": float(args.focal_alpha),
                    "focal_gamma": float(args.focal_gamma),
                },
            )
            print(f"  [Checkpoint] Epoch {epoch + 1} | Saved ({monitor_metric_name}={best_metric:.4f}, patience reset) -> {save_path}")
        elif is_cliff:
            print(
                f"  [Cliff] Epoch {epoch + 1}: metric={cur_monitor:.4f} is "
                f"{cur_monitor / best_metric * 100:.0f}% of best={best_metric:.4f} "
                f"(< {cliff_ratio:.0%} threshold) — patience not incremented "
                f"(no_improve={no_improve}/{args.patience})"
            )
        else:
            no_improve += 1
            _pat = int(args.patience)
            if _pat > 0 and no_improve >= _pat:
                print(f"  [EarlyStop] no_improve={no_improve}/{_pat} — early stopping triggered.")
                break
            elif _pat > 0:
                _remaining = _pat - no_improve
                _warn = "  !!!  close to stopping" if _remaining <= 3 else ""
                print(f"  [Patience] no_improve={no_improve}/{_pat} (remaining={_remaining}){_warn}")

    print(f"\nTraining complete. Best {monitor_metric_name}={best_metric:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {save_path}")
    _exit_state["reported"] = True


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    h, rem = divmod(int(end - start), 3600)
    m, s = divmod(rem, 60)
    print(f"End time:    {datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed:     {h:02d}:{m:02d}:{s:02d}")
