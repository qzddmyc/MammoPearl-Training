# 数据训练

## 预处理

你可以使用以下命令运行预处理程序：
```bash
python ./src/data/pre-process.py
```
如果你需要获得预处理的过程图片，使用如下命令：
```bash
python ./src/data/pre-process-test.py
```
此程序会选取随机 3 个文件夹下的所有图片进行预处理，并按照 `<图片索引>-<处理步骤>-<原始名称>` 的命名规则，将所有的中间产出结果保存至项目的 `/tmp` 文件夹下。

## 图像分隔

运行：
```bash
python ./src/data/img-segmentation.py
```
测试：
```bash
python ./src/data/img-segmentation-test.py
```
文档：
[segment.md](./segment.md)

注意，在图像分隔中使用到的所有数据均不依赖于 csv 标注文件，而是依赖预处理步骤的产出文件夹。
