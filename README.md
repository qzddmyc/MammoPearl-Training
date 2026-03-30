# MammoPearl-Training

Repo for MammoPearl-IBCDS Training Dataset, also serves as the course assignment of Digital Image Processing.

The dataset is obtained from [VinDr Mammogram dataset](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png).

\* 注意：你需要将下载的 `/archive/images_png` 文件夹下的所有内容放至本项目的 `/data/raw/images_png` 中。

## 预处理

Download the Python requirements:
```bash
pip install -r requirements.txt
```

Execute the Python script:
```bash
python ./src/data/pre-process.py
```
