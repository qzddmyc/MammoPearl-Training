**概述**
- **目的**: 本文档说明为乳腺 X 光片实现的图像分割工作、所用方法、运行方式与结果保存位置。
- **相关脚本**: 分割实现位于 [src/data/img-segmentation.py](src/data/img-segmentation.py#L1-L200)，用于快速测试的脚本位于 [src/data/img-segmentation-test.py](src/data/img-segmentation-test.py#L1-L200)。预处理流水线参见 [src/data/pre-process.py](src/data/pre-process.py#L1-L200) 与 [src/data/pre-process-test.py](src/data/pre-process-test.py#L1-L200)。

**实现细节**
- **输入**: 已完成预处理后的灰度图像（来自 `data/processed/images_png/<patient>/*`），预处理包括掩膜提取、双边滤波与 CLAHE 增强，详见 `src/data/pre-process.py`。
- **分割方法**: 默认采用 Otsu 全局阈值法（`otsu`），并提供可选的自适应阈值法（`adaptive`）。
- **预处理 / 去噪**: 在阈值前对灰度图做高斯模糊（`cv2.GaussianBlur`，kernel=(5,5)）以降低噪声对阈值的影响。
- **形态学后处理**: 使用椭圆形结构元素（7x7）执行开运算与闭运算以去除小噪点并填充小孔。
- **连通域筛选**: 对二值结果做连通组件分析，仅保留面积最大的连通域，避免小斑点干扰后续分析。

**输出与文件位置**

- 本实现直接遍历并处理 `data/processed/images_png` 下的所有图片（递归查找）。
- 分割输出被保存到仓库内部固定位置：
  - `data/segmented/base/<patient>/` — 保存处理后的原图（灰度/预处理结果），保持与 `data/processed/images_png/<patient>/` 相同的子文件夹结构与文件名。
  - `data/segmented/mask/<patient>/` — 保存对应的二值 mask（0/255），文件名与原图相同。

**如何运行**

- 直接在仓库根目录运行分割脚本：

  `python src/data/img-segmentation.py`

  - 脚本会自动查找 `data/processed/images_png` 下的图片，处理后把结果放到 `data/segmented/base/<patient>/` 与 `data/segmented/mask/<patient>/`。

**验证与调整建议**

- **目视验证**: 使用文件管理器或图像查看器比较 `data/segmented/base/<patient>/*.png` 与 `data/segmented/mask/<patient>/*.png`。
- **参数调整**: 若发现分割不理想，可在 `src/data/img-segmentation.py` 中修改形态学核大小、模糊参数或尝试 `adaptive` 方法。

**已实现的文件（快速索引）**
- **分割实现**: [src/data/img-segmentation.py](src/data/img-segmentation.py#L1-L200)
- **分割测试**: [src/data/img-segmentation-test.py](src/data/img-segmentation-test.py#L1-L200)
- **预处理脚本**: [src/data/pre-process.py](src/data/pre-process.py#L1-L200)
- **预处理测试**: [src/data/pre-process-test.py](src/data/pre-process-test.py#L1-L200)

**后续改进建议**
- **基于学习的方法**: 若需更鲁棒的分割，可考虑基于 U-Net 的监督学习分割模型，并用预处理后的图像与人工 mask（如有）训练；
- **参数搜索**: 对形态学核、阈值方法与去噪参数做网格搜索或基于少量人工标注的验证集调参；
- **批处理并行化**: 对大量图片可并行化处理以加速（使用 multiprocessing 或 GPU 加速的模型）。

**注意事项**
- 本分割脚本为传统图像处理流水线，适合快速原型与可解释性需求；对复杂病灶或贴合临床级别的分割应使用带标注的深度学习方法。
