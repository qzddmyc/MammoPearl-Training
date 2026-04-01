# MammoPearl-Training

MammoPearl-IBCDS 训练数据集仓库，同时也作为《数字图像处理》课程的作业项目。

本项目旨在实现：基于乳腺 X 光图像的乳腺癌筛查。

该数据集来源于 [VinDr Mammogram 数据集](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)。

## 项目结构

```plaintext
MammoPearl-Training
├── data                    # 数据相关目录
│   ├── raw
│   │   ├── images_png                  # 原始数据目录
│   │   │   └── dataset.sha256          # 数据集完整性校验文件
│   │   └── vindr_detection_folds.csv   # 数据集划分文件
│   └── processed
│       └── images_png      # 预处理后的数据目录
├── src
│   ├── init
│   │   └── download-dataset.py   # 原始数据下载脚本
│   └── data
│       ├── pre-process.py        # 数据预处理脚本
│       └── pre-process-test.py   # 数据预处理测试脚本
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
