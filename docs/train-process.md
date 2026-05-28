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

你可以在 `src/data/segment/segment.py` 中阅读顶部提示词，并查看其中写好的四个函数。这四个函数就是分割的关键函数。

## 病灶区域识别（利用 bbox）

你可以在 `src/data/bounding-box/` 文件夹下查看到 `bbox-*` 文件，这些是用来训练识别 bbox（标注框）的训练与测试文件。

我们最开始的计划是，先训练一个可以寻找出病灶区域的模型（stage1），之后将结果送入 stage2 进行分类。最终由于“精确定位”的要求、样本自身大小问题等原因，召回率不到 50%（或者说是 Recall 与 Precision 不可兼得），放弃了该思路。但是我们保留了优化历史（changelog）与最终方案的代码，供参考。

以及该文件夹下的 `vis-*` 文件，是数据可视化的一些文件，也不再使用，但可以独立运行：
- `vis-augment.py` 是利用翻转与旋转，增加数据集的可视化代码。
- `vis-copy-paste.py` 是利用将正样本的病灶复制至负样本中，增加正样本的可视化代码。

文件夹下的 `output.txt` 是最终保留轮的训练与测试的日志产出。

## 传统机器学习分类

基于手工特征（GLCM 纹理、LBP、Wavelet、Gabor 滤波等）和传统机器学习算法（SVM + XGBoost），实现了一套无深度学习依赖的双阶段乳腺癌分类系统。

代码位于 `src/data/recognition-traditional/`，完整训练命令：

```bash
python src/data/recognition-traditional/run_pipeline.py
```

详细说明（特征提取方法、训练成果、算法说明）请参见 [docs/recognition-traditional.md](./recognition-traditional.md)。
