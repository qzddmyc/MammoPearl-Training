# 数据训练

## 预处理

你可以使用以下命令运行预处理程序：
```bash
python src/data/pre_process/pre-process.py
```
如果你需要获得预处理的过程图片，使用如下命令：
```bash
python src/data/pre_process/pre-process-test.py
```
此程序会选取随机 3 个文件夹下的所有图片进行预处理，并按照 `<图片索引>-<处理步骤>-<原始名称>` 的命名规则，将所有的中间产出结果保存至项目的 `/tmp/pre_process_test` 文件夹下。

## 图像分割

你可以使用以下命令运行分割的**测试**程序：
```bash
python src/data/segment/segment-test.py
```
该测试程序可以选取部分**正样本**的**训练**文件，并根据标注文件中的数据对图像进行分割。其保存的目录会在运行之后进行输出。

完整的分割程序并未完善，敬请期待。

但是你可以在 `src/data/segment/segment.py` 中阅读顶部提示词，并查看其中写好的四个函数。这四个函数就是分割的关键函数。

另外，你可以在 `src/data/bounding-box/` 文件夹下查看到 `bbox-*` 文件，这些是用来训练识别 bbox （标注框）的训练与测试文件。由于**模型训练耗时较长且效果不佳**，这些文件正在测试中。`bbox-visualize-breast-crop.py` 文件使用来查看训练与测试文件的标注框位置是否正确的，以防止图像放缩导致异常。

## 传统机器学习分类

基于手工特征（GLCM 纹理、LBP、Wavelet、Gabor 滤波等）和传统机器学习算法（SVM + XGBoost），实现了一套无深度学习依赖的双阶段乳腺癌分类系统。

代码位于 `src/data/recognition-traditional/`，完整训练命令（`--skip-mask` 表示跳过掩码生成，掩码当前未被采样逻辑实际使用）：

```bash
python src/data/recognition-traditional/run_pipeline.py --skip-mask
```

详细说明（特征提取方法、训练成果、算法说明）请参见 [docs/recognition-traditional.md](./recognition-traditional.md)。
