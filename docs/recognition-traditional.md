# 乳腺癌传统机器学习分类系统

> **数据集**：VinDr Mammography（经处理版本）
> 
> **方法**：手工特征 + 传统机器学习（无深度学习依赖）
> 
> **代码目录**：`src/data/recognition-traditional/`
> 
> **入口脚本**：`run_pipeline.py`

---

## 一、整体思路

乳腺 X 光片的病灶区域与周围正常腺体在外观上极为相似，难以直接用一个分类器搞定"有没有病"和"是什么病"两个问题。因此系统拆成两个阶段：

```
原始影像
  │
  ▼
[Step 1] 采样：把图像切成一个个 128×128 的小块
  ├─ 正样本：从标注的病灶框中心裁剪
  └─ 难负样本：从同一张图的最致密腺体区域裁剪
  │
  ▼
[Step 2] 预处理（CLAHE 对比度增强 → 归一化）
  │
  ▼
[Step 3] 提取手工特征（描述纹理、形状、频率特性）
  │
  ├─▶ [Stage-1] 二分类：这块区域有没有病变？- SVM，目标高召回率
  │         │ 预测为有病变
  │         ▼
  └─▶ [Stage-2] 多分类：是哪种病变？- XGBoost，目标高准确率
```

> **[Step 0 - 可选]** 使用 `--generate-mask` 可在采样前生成乳腺区域掩码，但当前采样逻辑尚未读取掩码，不影响上述流程。详见：十、注意事项第 2 条。

Stage-1 宁可多报也不漏报（高 recall）；Stage-2 在 Stage-1 的阳性结果中进一步区分病变类型。

---

## 二、数据说明

| 项目 | 内容 |
|------|------|
| 数据来源 | `data/raw/vindr_detection_folds.csv`（约 20,486 行） |
| 图像路径 | `data/processed/images_png/<patient_id>/<image_id>` |
| 训练 / 测试 | 患者级别划分，零重叠：16,391 张训练，4,095 张测试 |
| 病变类别 | 10 种（见下方） |

**10 种原始病变类型**：

| 编号 | 英文名 | 说明 |
|------|--------|------|
| 0 | Architectural_Distortion | 乳腺组织结构扭曲 |
| 1 | Asymmetry | 两侧乳腺不对称 |
| 2 | Focal_Asymmetry | 局灶性不对称（单个区域的密度集中） |
| 3 | Global_Asymmetry | 全局性不对称（整体密度差异） |
| 4 | Mass | 肿块（有明确边界的占位） |
| 5 | Nipple_Retraction | 乳头凹陷 |
| 6 | Skin_Retraction | 皮肤收缩 |
| 7 | Skin_Thickening | 皮肤增厚 |
| 8 | Suspicious_Calcification | 可疑钙化（微小高亮点） |
| 9 | Suspicious_Lymph_Node | 可疑淋巴结 |

---

## 三、手工特征提取

传统机器学习模型无法直接处理原始像素，需要先将图像转换为一组数值向量，再交给分类器学习。这些数值向量就是"特征"：每一维对应图像的某种可量化属性（如纹理粗糙程度、边缘强度、频率分布等）。

特征提取的质量直接决定模型的上限：如果特征无法区分不同病变的视觉差异，再强的分类器也无济于事。本系统从纹理、频率、统计等多个角度提取共 **113 维基础特征**（供 Stage-1 使用）和额外 **6 维扩展特征**（供 Stage-2 使用）。

### 3.1 基础特征（113 维，用于 Stage-1）

#### GLCM 纹理（48 维）
全称是灰度共生矩阵（Gray-Level Co-occurrence Matrix）。它统计的是：在一张图里，某个灰度值的像素旁边出现另一个灰度值像素的频率。从中可以提取 4 个描述符：

- **对比度（Contrast）**：相邻像素的灰度差异大不大。钙化点周围对比度高，腺体纹理对比度低。
- **相关性（Correlation）**：像素灰度值的线性相关程度。
- **能量（Energy）**：纹理的均匀程度。均匀纹理能量高，复杂纹理能量低。
- **同质性（Homogeneity）**：灰度过渡的平滑程度。

计算时考虑 2 种距离（1 像素、3 像素）× 4 个方向（0°、45°、90°、135°），再加上 GLCM 矩阵本身的熵，共 48 维。

#### LBP 纹理（26 维）
全称是局部二值模式（Local Binary Pattern）。对每个像素，比较它和周围 24 个邻居的灰度大小，大的记 1、小的记 0，生成一个二进制编码。然后统计整个图像块里各种编码出现的频率（直方图），共 26 维。

LBP 对旋转不敏感，适合描述乳腺纹理的粗细和方向性。

#### Wavelet 小波特征（18 维）
用 db4 小波对图像做 2 级分解，像给图像拍一张"频率 X 光"：低频是整体轮廓，高频是细节边缘。对每个高频子带提取均值（绝对值）、标准差、能量，共 18 维。

钙化点会在高频子带产生强响应；肿块则主要影响低频分量。

#### Gabor 滤波（24 维）
Gabor 滤波器相当于在图像上找某个方向、某个频率的条纹或纹理。用 3 种频率 × 4 个方向 = 12 个滤波器，每个提取响应的均值和标准差，共 24 维。

用来捕捉乳腺纹理的方向性，比如放射状扭曲（Architectural_Distortion）会有特定的方向分布。

#### 统计量（5 维）
直接统计图像块所有像素的：均值、标准差、偏度（分布是否歪）、峰度（分布是否尖）、四分位距（IQR）。

---

### 3.2 扩展特征（额外 6 维，用于 Stage-2）

在基础 113 维之上，针对两类比较特殊的病变额外加了 6 维：

#### 钙化特征（3 维）
用 Laplacian 算子（一种边缘增强算子）提取图像的高频细节，取响应的均值、标准差和最大值。

钙化是非常细小的高亮点，在 Laplacian 处理后会有很强的响应，其他病变则相对较弱。

#### 肿块形状特征（3 维）
先用 Otsu 阈值把图像二值化（亮的区域设为前景），然后找最大轮廓，计算：

- **圆度（Circularity）**：轮廓有多圆。良性肿块通常边缘光滑偏圆，恶性的通常不规则。
- **实体度（Solidity）**：轮廓面积 / 凸包面积，衡量形状是否有凹陷。
- **内部密度（Density）**：前景区域的平均亮度。

---

## 四、Stage-1：二分类（有没有病变）

### 算法：SVM（支持向量机）

SVM 的核心思想是找一个"最宽的分界线"把两类样本分开。用 RBF 核函数（一种把数据映射到高维空间的技巧）可以处理线性不可分的情况。

**训练流程：**

1. 数据标准化（`StandardScaler`）：让每个特征的均值为 0、方差为 1
2. PCA 降维：把 113 维压缩到 80 维，去掉冗余特征
3. SVM 训练，设置 `class_weight='balanced'` 自动处理正负样本不均衡（正样本 1,795 个 vs 负样本 48,000 个）
4. **5 折网格搜索（GridSearchCV）**：在 `C=[0.1, 1, 10]` 和 `gamma=['scale', 'auto']` 上交叉验证，选出 recall 最高的参数组合
5. **自动调优阈值**：在 `[0.1, 0.7]` 区间扫描概率阈值，选出 recall ≥ 0.90 时 F1 最高的那个点

> 最优参数：`C=1.0, gamma='scale'`，阈值 `0.700`

---

## 五、Stage-2：多分类（是哪种病变）

### 类别合并策略

原始 10 个类别中，部分在训练集里样本极少（如 Suspicious_Lymph_Node 可能只有几个），完全没法学习。为此将 10 类合并为 4 类：

| 合并后的类 | 包含的原始类别 | 合并依据 |
|-----------|-------------|---------|
| Asymmetry_Distortion | Architectural_Distortion, Asymmetry, Focal_Asymmetry, Global_Asymmetry | 都表现为局部或整体的密度/结构不对称，patch 纹理相近 |
| Mass | Mass | 训练集样本量最多（989 条），独立保留 |
| Skin_Other | Nipple_Retraction, Skin_Retraction, Skin_Thickening, Suspicious_Lymph_Node | 均出现在图像边缘区域，视觉特征相近 |
| Suspicious_Calcification | Suspicious_Calcification | 视觉特征独特（微小高亮点），独立保留 |

> 曾尝试将 Focal_Asymmetry 单独作为第 5 类，结果 Macro F1 从 0.568 降至 0.449（两组各约 50 个样本，模型无法有效区分），最终回退到 4 类方案。

### 算法：XGBoost

XGBoost 是梯度提升树的高效实现：先训一棵决策树，再训一棵专门修正前面错误的树，反复叠加 600 棵。比随机森林更"聪明"，擅长处理样本不均衡和特征冗余。

**主要超参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_estimators` | 600 | 树的数量 |
| `max_depth` | 8 | 每棵树最深 8 层 |
| `learning_rate` | 0.03 | 每棵树的贡献权重（越小越稳健） |
| `subsample` | 0.8 | 每棵树随机用 80% 的样本 |
| `colsample_bytree` | 0.7 | 每棵树随机用 70% 的特征 |
| `gamma` | 0.1 | 分裂节点的最小增益门槛（防止过拟合） |
| `reg_alpha` | 0.1 | L1 正则，减少不重要特征的影响 |

**训练流程：**

1. 只取 Stage-1 标签为阳性的样本（1,795 个），忽略负样本
2. `StandardScaler` 特征归一化
3. `LabelEncoder` 将类别索引编码为连续整数
4. 5 折交叉验证，OOF 准确率 **0.670**
5. 在全部 1,795 个样本上重新训练最终模型

---

## 六、训练成果

### 评估方式说明

下面的数字**不是**滑窗推理的评估结果，而是一种"标准化的 patch 级评估"。这里的 **patch** 是指从原始乳腺图像中裁剪出的固定尺寸（128×128 像素）小图块，是模型的基本处理单元。具体评估流程如下：

1. **测试集构建**：和训练集完全一样：对每个已知的病灶标注框，从框的中心裁剪出 128×128 的图像块作为正样本；同时从正常图像上随机裁剪负样本。最终测试集共 **12,447 个 patch**（其中 447 个病变阳性，12,000 个正常）。
2. **模型推理**：把每个测试 patch 直接送给模型分类，统计 precision/recall/F1。
3. **为什么这样做**：它能准确衡量模型"在最理想条件下"的分类能力：正样本的窗口已经对准了病灶中心。这是评估手工特征 + 分类器组合质量的标准做法，训练流程简单快速。

**与滑窗推理的区别**：滑窗推理对整张图像密集扫描，窗口不一定对准病灶，大量窗口只包含边缘或背景区域，模型需要在更多干扰中找到真正的病变。因此，如果在整个测试集上跑滑窗评估，检测级别的 precision 和 recall 会比下面的数字低。

---

### Stage-1（测试集 12,447 个 patch）

| 指标 | 值 |
|------|-----|
| Precision（精确率） | 0.9932 |
| Recall（召回率） | 0.9843 |
| F1 分数 | 0.9888 |
| ROC-AUC | 1.0000 |

> ROC-AUC=1.0 是因为正样本直接来自标注框中心，模型看到的每个正样本都是"病灶摆在中间"的图块，比真实滑窗场景容易得多。真实部署中 ROC-AUC 会有所下降。

### Stage-2（测试集 440 个正样本）

| 病变类型 | 精确率 | 召回率 | F1 | 测试样本数 |
|---------|-------|-------|-----|---------|
| Asymmetry_Distortion | 0.50 | 0.25 | 0.33 | 102 |
| Mass | 0.65 | 0.88 | **0.75** | 232 |
| Skin_Other | 0.80 | 0.40 | 0.53 | 20 |
| Suspicious_Calcification | 0.75 | 0.57 | 0.65 | 86 |

| 整体指标 | 值 |
|---------|-----|
| Accuracy（整体准确率） | 0.6523 |
| Cohen Kappa（一致性系数） | 0.3859 |
| Macro F1（各类 F1 均值） | **0.5652** |

### 迭代改进过程

| 迭代 | 方案 | Accuracy | Kappa | Macro F1 |
|------|------|---------|-------|---------|
| 基准 | 10 类原始，无加权 | 0.618 | 0.326 | 0.252 |
| 迭代 1 | 10 类 + sample_weight 加权 | 0.575 | 0.286 | 0.224 ↓ |
| 迭代 2 | 合并为 4 类，无加权 | 0.645 | 0.375 | **0.568** ↑↑ |
| 迭代 3 | 5 类（拆出 Focal_Asymmetry） | 0.630 | 0.347 | 0.449 ↓ |
| 迭代 4（最终） | 4 类 + 调优超参数 | **0.652** | **0.386** | 0.565 |

### 当前瓶颈

Asymmetry_Distortion 类的 F1 长期停留在 0.33，根本原因：

1. **类内差异大**：这一类合并了结构扭曲、整体不对称、局灶不对称等 4 种外观差异显著的病变，128×128 的 patch 纹理特征难以统一表达
2. **手工特征的天花板**：GLCM/LBP/Wavelet 等统计特征擅长描述局部纹理，但"不对称性"需要全乳图像级别的感受野才能感知，patch 级别的特征无法捕捉
3. **样本量不足**：4 种亚型合并后仍只有 406 个训练样本（Architectural_Distortion 95 + Asymmetry 77 + Focal_Asymmetry 216 + Global_Asymmetry 20）

如需突破此瓶颈，需引入深度学习（CNN 在全乳图像上直接提取特征）。

---

## 七、滑窗推理

上面的评估都是在"已知病灶位置"的情况下做的（直接裁剪框中心）。对于实际部署（给一张新图找出病灶位置和类型），系统实现了对**单张图片**实现滑窗推理的测试方式：

**工作原理：**

1. 以指定步长（stride）在完整图像上滑动 128×128 窗口
2. 对每个窗口提取基础特征，运行 Stage-1 判断是否有病变
3. 对 Stage-1 阳性的窗口提取扩展特征，运行 Stage-2 判断病变类型
4. 用 NMS（非极大值抑制）合并高度重叠的检测框

**使用方法：**

```bash
python src/data/recognition-traditional/run_pipeline.py \
    --infer data/processed/images_png/<patient_id>/<image_id>.png \
    --stride 32 \
    --nms-iou 0.3
```

输出示例：
```
[infer] Image xxx.png: 1520×912, 77 windows (stride=128, patch_size=128)
[infer] Stage-1 positives: 5 / 77 (threshold=0.700)
[infer] After NMS (IoU≤0.3): 5 detections (from 5)

[infer] Detected 5 region(s):
  Rank     x1    y1    x2    y2   Score  Class
  -------------------------------------------------------
  1       512  1152   640  1280  0.9889  Focal_Asymmetry
  ...
```

> `stride` 越小，检测越精细但速度越慢（stride=32 约生成 1,000+ 个窗口）；`nms-iou` 越小，保留的框越多。

---

## 八、文件结构

```
src/data/recognition-traditional/
├── config.py           # 全局配置（路径、超参数、类别合并映射）
├── preprocessing.py    # CLAHE + 归一化 + resize
├── sampling.py         # 正样本 / 难负样本 patch 采样
├── features.py         # 手工特征提取（GLCM/LBP/Wavelet/Gabor/统计量 + 扩展）
├── train_stage1.py     # Stage-1 SVM 训练（GridSearchCV + 阈值调优）
├── train_stage2.py     # Stage-2 XGBoost 训练
├── evaluate.py         # 评估指标计算与报告生成
├── run_pipeline.py     # CLI 入口（训练 / 评估 / 滑窗推理）
└── output/
    ├── models/         # 保存的模型文件（.pkl）
    ├── features/       # 特征缓存（.npy）
    └── reports/        # 评估报告（.txt）与混淆矩阵图（.png）
```

## 九、完整运行命令

```bash
# 完整训练（Stage-1 + Stage-2）
python src/data/recognition-traditional/run_pipeline.py

# 只训练 Stage-1
python src/data/recognition-traditional/run_pipeline.py --stage 1

# 只训练 Stage-2（需要已有 Stage-1 模型）
python src/data/recognition-traditional/run_pipeline.py --stage 2

# 只跑评估（复用已保存的模型）
python src/data/recognition-traditional/run_pipeline.py --eval-only

# 对单张图像做滑窗检测
python src/data/recognition-traditional/run_pipeline.py \
    --infer data/processed/images_png/<patient_id>/<image_id>.png \
    --stride 32 --nms-iou 0.3

# 先生成乳腺掩码再训练（但掩码生成与否与训练本身无关）
python src/data/recognition-traditional/run_pipeline.py --generate-mask
```

---

## 十、注意事项

1. **运行产出与缓存管理**：

   - **特征缓存**：首次运行 `build_features()` 时，提取完成的基础特征矩阵会保存到 `output/features/` 下的三个 `.npy` 文件（`train_features.npy`、`train_labels.npy`、`train_stage2_labels.npy` 及对应的 `test_*` 文件）。后续再次运行会直接加载缓存，跳过耗时的采样和特征提取步骤。
   - **模型文件**：训练完成后，Stage-1 模型（`pipeline + threshold`）保存至 `output/models/`，Stage-2 模型（`xgb model + scaler + label encoder`）同样保存至此目录，供后续 `--eval-only` 和 `--infer` 加载。
   - **评估报告**：每次评估后在 `output/reports/` 下生成文本报告（`.txt`）和混淆矩阵图（`.png`）。

   **何时需要手动删除缓存**：修改了 `STAGE2_MERGE_MAP`（类别合并方案）后，标签文件（`train_stage2_labels.npy` / `test_stage2_labels.npy`）不会自动更新，需手动删除 `output/features/` 下所有 `.npy` 文件，再重新运行以重建缓存。特征矩阵本身（`*_features.npy`）不受类别合并影响，可以保留，但为简单起见删除全部再重建最为稳妥。

2. **掩码生成（当前未接入采样）**：默认不生成掩码；如需生成请加 `--generate-mask`，但掩码当前未被采样逻辑实际使用：`build_dataset()` 的难负样本挖掘用积分图法在全图采样，不依赖掩码文件。如需改进，需在 `build_dataset()` 里加入掩码读取和边界过滤的逻辑。

3. **掩码生成的外部依赖**：掩码生成步骤依赖 `src/data/segment/segment.py`（整个传统 ML 模块的唯一外部依赖），需单独安装其依赖环境；其余模块均自包含，仅使用 `--generate-mask` 时才需要此依赖。
