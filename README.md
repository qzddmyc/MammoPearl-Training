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
│       └── mask/           # 分割形成的遮罩
├── models/                 # 训练产出的模型
│   └── raw/                # 预训练权重存放目录
├── src/
│   ├── init/
│   │   ├── download-dataset.py         # 原始数据下载脚本
│   │   └── download_backbone.sh        # 病灶框检测预训练权重下载脚本
│   └── data/
│       ├── pre_process/                # 数据预处理
│       ├── segment/                    # 图像分割
│       ├── bounding-box/               # 病灶框区域检测
│       ├── recognition-traditional/    # 传统机器学习
│       └── deep-learning/              # 深度学习
│
├── docs/                   # 项目文档
├── tools/                  # 工具文件
├── README.md
├── build_dataset.sh        # 项目及数据集初始化脚本
└── requirements.txt        # Python 依赖列表
```

## 依赖初始化

使用该脚本下载依赖与数据集：
```bash
bash ./build_dataset.sh
```

## 数据集概况

详见：[docs/dataset-overview.md](./docs/dataset-overview.md)。

## 项目详细描述文档

详见：[docs/train-process.md](./docs/train-process.md)。

## 系统架构

### 传统机器学习路线

独立于深度学习路线，采用 **两阶段级联** 策略完成病灶分类。

```
预处理图像
    │
    ▼
patch 采样 + 手工特征提取
    • 正样本：从标注框中心裁剪 128×128 patch
    • 负样本：积分图法在最致密腺体区域挖掘难负样本
    • 统计特征：灰度均值 / 标准差 / 偏度 / 峰度 / 熵
    • 纹理特征：GLCM 对比度 / 相关性 / 能量 / 同质性
    • 频域特征：Gabor 滤波器组响应
    • 梯度特征：Sobel / Laplacian 能量
    │
    ▼
[Stage-1] SVM 二分类（正常 vs 病灶候选）
    • Pipeline: StandardScaler → PCA(80) → SVC
    • 高召回率优先（阈值 0.700），Recall ≈ 0.984
    │
    └── 正样本候选
              │
              ▼
        [Stage-2] XGBoost 4 类病灶分类
            • 类别：Asymmetry_Distortion / Mass /
                    Skin_Other / Suspicious_Calcification
            • Accuracy ≈ 0.652
```

### 深度学习路线

当前深度学习方案采用两阶段分类流水线：

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
[Stage 1 - 图像级二分类筛查]  src/data/deep-learning/
    • 模型：EfficientNet-B4（二分类，是否存在病变）
    • 目标：高召回率优先，@0.10 阈值时 Recall ≈ 95.5%
    • 输出：stage1_prob（0–1）
    │
    ├─── prob < threshold → 阴性，流程结束
    │
    └─── prob ≥ threshold → 阳性
                  │
                  ▼
        [Stage 2 - 条件病变类型分类]  src/data/deep-learning/
            • 模型：EfficientNet-B4（三分类）
            • 类别：Mass / Calcification / Asymmetry_Distortion（含 Skin_Other）
            • 仅对 Stage 1 判阳性的图像运行
                  │
                  ▼
             最终输出：有无病变 + 病变类型
```

**设计逻辑**：Stage 1 负责高召回筛查，尽量少漏检；Stage 2 只在阳性图像上细分病变类型。
