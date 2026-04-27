# MammoPearl-Training

MammoPearl-IBCDS 训练数据集仓库，同时也作为《数字图像处理》课程的作业项目。

本项目旨在实现：基于乳腺 X 光图像的乳腺癌筛查。

该数据集来源于 [VinDr Mammogram 数据集](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)。其中，csv 标注文件经过了部分的修改。

## 项目结构

```plaintext
MammoPearl-Training/
├── data/                   # 数据相关目录
│   ├── raw/
│   │   ├── images_png/                 # 原始数据目录
│   │   │   └── dataset.sha256          # 数据集完整性校验文件
│   │   └── vindr_detection_folds.csv   # 数据集划分文件
│   ├── processed/
│   │   └── images_png/     # 预处理后的数据目录
│   └── segmented/          # 分割后的图像
│       ├── base/           # 基于 processed 的原图
│       └── mask/           # 遮罩
├── src/
│   ├── init/
│   │   └── download-dataset.py         # 原始数据下载脚本
│   └── data/
│       ├── pre-process/    # 数据预处理
│       ├── segment/        # 图像分隔
│       └── bounding-box/   # 标注框预测模型的训练
│
├── tools/                  # 工具文件
├── models/                 # 训练产出的模型
├── README.md
├── build_dataset.sh        # 项目及数据集初始化脚本
└── requirements.txt        # Python 依赖列表
```

## 依赖初始化

使用该脚本下载依赖与数据集：
```bash
bash ./build_dataset.sh
```

## 数据训练

详见：[docs/train-process.md](./docs/train-process.md)。

## 系统架构

本项目采用两阶段级联架构：

```
乳腺 X 光原图（PNG）
    │
    ▼
[预处理]  src/data/pre_process/
    • 乳腺区域提取（Otsu 阈值 + 最大轮廓掩膜，去除背景和人工标记）
    • 双边滤波降噪（保留病灶边缘及钙化点）
    • CLAHE 对比度增强（提升局部微小病灶可见度）
    │
    ▼
[Stage 1 — 病灶框检测]  src/data/bounding-box/
    • 模型：RetinaNet (ResNet-50 FPN v2)
    • 目标：高召回率（漏检不可接受，误报可容忍）
    • 输出：病灶候选框列表 + 各框置信度分数
    │
    ├─── 未检出病灶框 → 低患病风险（权重低）
    │
    └─── 检出病灶框  → 框位置 + 置信度作为附加特征
                              │
                              ▼
                    [Stage 2 — 患病分类]  （下游，待实现）
                        • 输入：原图 + 病灶框位置 + 置信度
                        • 目标：判断是否患病 / 病灶类型
                        • 病灶框的存在及置信度作为先验权重
                              │
                              ▼
                         最终诊断结论
```

**设计逻辑**：Stage 1 只负责"是否有可疑区域"及"在哪里"，Stage 2 利用这些位置信息进行更精确的患病判断。Stage 1 的漏检（FN）会导致 Stage 2 丢失关键信息，因此召回率是 Stage 1 的核心指标。
