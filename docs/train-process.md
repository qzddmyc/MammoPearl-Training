# 数据训练

## 预处理

你可以使用以下命令运行预处理程序：
```bash
python ./src/data/pre_process/pre-process.py
```
如果你需要获得预处理的过程图片，使用如下命令：
```bash
python ./src/data/pre_process/pre-process-test.py
```
此程序会选取随机 3 个文件夹下的所有图片进行预处理，并按照 `<图片索引>-<处理步骤>-<原始名称>` 的命名规则，将所有的中间产出结果保存至项目的 `/tmp/pre_process_test` 文件夹下。

## 图像分割

你可以使用以下命令运行分割的**测试**程序：
```bash
python ./src/data/segment/segment-test.py
```
该测试程序可以选取部分**正样本**的**训练**文件，并根据标注文件中的数据对图像进行分割。其保存的目录会在运行之后进行输出。

完整的分割程序并未完善，敬请期待。

但是你可以在 `src/data/segment/segment.py` 中阅读顶部提示词，并查看其中写好的四个函数。这四个函数就是分割的关键函数。

另外，你可以在 `src/data/segment/` 文件夹下查看到 `bbox-*` 文件，这些是用来训练识别 bbox （标注框）的训练与测试文件。由于存在大量被污染的数据，这些文件正在测试中。`validate_bbox_coords.py` 文件使用来查看训练与测试文件的标注框位置是否正确的，以防止图像放缩导致异常。