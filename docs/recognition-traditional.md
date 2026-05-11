# 乳腺癌影像层级诊断系统 — 实现文档

> **数据集**：VinDr Mammography（经处理版本）  
> **方法栈**：传统机器学习（无深度学习依赖）  
> **代码目录**：`src/data/recognition-traditional/`  
> **入口脚本**：`run_pipeline.py`

---

## 一、系统架构

针对乳腺 X 光影像中病灶与正常组织高度相似、肉眼难辨的挑战，系统采用**"先检出，后分类"**的双阶段级联策略：

```
原始影像
  │
  ▼
[Step 0] 乳腺区域分割（segment.py → data/segmented/）
  │
  ▼
[Step 1] 采样（正样本 bbox ROI + 难负样本 Hard Negative Mining）
  │
  ▼
[Step 2] 预处理（CLAHE → 归一化）
  │
  ▼
[Step 3] 手工特征提取
  ├─ 基础特征 → Stage-1
  └─ 扩展特征 → Stage-2
  │
  ├─▶ [Stage-1] 二分类（有无病灶）— SVM/RF，高召回率优先
  │         │ 预测为阳性
  │         ▼
  └─▶ [Stage-2] 多分类（病灶类型）— XGBoost/LightGBM，高准确率优先
```

**阶段一** 核心指标：Recall（减少漏诊）  
**阶段二** 核心指标：Accuracy / Kappa（减少误诊）

---

## 二、数据与路径约定

| 路径 | 说明 |
|------|------|
| `data/raw/vindr_detection_folds.csv` | 标注文件（约 20 486 行）|
| `data/processed/images_png/<patient_id>/<image_id>` | 处理后影像（`image_id` 含 `.png` 后缀）|
| `data/segmented/base/` | 乳腺区域分割基底图 |
| `data/segmented/mask/` | 乳腺掩码（`generate_masks()` 生成）|
| `src/data/recognition-traditional/output/features/` | 特征缓存（`.npy`）|
| `src/data/recognition-traditional/output/models/` | 保存的模型（`.pkl`）|
| `src/data/recognition-traditional/output/reports/` | 评估报告（JSON/PNG/TXT）|

CSV 关键字段：

| 字段 | 含义 |
|------|------|
| `patient_id` | 患者 ID |
| `image_id` | 影像文件名（含 `.png`）|
| `split` | `training` / `test` |
| `No_Finding` | 1 = 正常图像 |
| `xmin, ymin, xmax, ymax` | 病灶边界框 |
| 10 个病灶列 | `Architectural_Distortion`, `Asymmetry`, `Focal_Asymmetry`, `Global_Asymmetry`, `Mass`, `Nipple_Retraction`, `Skin_Retraction`, `Skin_Thickening`, `Suspicious_Calcification`, `Suspicious_Lymph_Node` |

---

## 三、模块说明

### `config.py` — 全局配置

所有路径常量、超参数集中定义，避免硬编码：

- `PATCH_SIZE = 128`：图像块大小
- `HARD_NEG_PER_IMAGE = 3`：每图难负样本数
- `PCA_N_COMPONENTS = 80`：PCA 降维维数
- `STAGE1_DECISION_THRESHOLD = 0.3`：初始决策阈值（训练后自动调优）
- `GLCM_DISTANCES`, `GLCM_ANGLES`, `GABOR_FREQUENCIES`, `GABOR_THETAS`：特征超参数

---

### `preprocessing.py` — 图像预处理

| 函数 | 说明 |
|------|------|
| `to_gray(img)` | BGR/RGB → 灰度 uint8 |
| `apply_clahe(gray)` | CLAHE 对比度增强（clip=2.0，tile 8×8）|
| `normalize_image(gray)` | 归一化至 `[0,1]` float32 |
| `frangi_filter(gray_norm)` | 纯 numpy 多尺度 Hessian blob 增强（scales=1,2,4）|
| `preprocess_patch(patch)` | 完整流水线：to_gray → resize → CLAHE → normalize |

---

### `sampling.py` — 样本集构建

| 函数 | 说明 |
|------|------|
| `generate_masks(force=False)` | 调用 `build_training_masks_from_csv`，生成乳腺掩码 |
| `build_dataset(split)` | 返回 `(patches, records)` |

`build_dataset` 逻辑：

1. 对每张有病灶的图像，按 bbox 裁剪正样本（`stage1_label=1`）
2. 对同一图像的最亮非 ROI 区域（积分图法），裁剪 `HARD_NEG_PER_IMAGE` 个难负样本（`stage1_label=0`）
3. 对 `No_Finding=1` 的图像，随机采样负样本

`records` 字段：`patient_id`, `image_id`, `stage1_label`(0/1), `stage2_label`(-1 或 0–9), `is_hard_neg`

---

### `features.py` — 手工特征提取

#### 基础特征（Stage-1）

| 特征类型 | 方法 | 说明 |
|----------|------|------|
| GLCM | 灰度共生矩阵 | 32 级量化；contrast / correlation / energy / homogeneity / entropy × 距离 × 角度 |
| LBP | 局部二值模式 | `method="uniform"`，`radius=3`，`n_points=24`，26 维直方图 |
| Wavelet | 小波变换 | `db4`，2 级分解；高频子带的 mean(abs) / std / energy |
| Gabor | Gabor 滤波 | 3 频率 × 4 方向；幅值 mean + std |
| Statistical | 统计量 | mean, std, skewness, kurtosis, IQR（5 维）|

#### 扩展特征（Stage-2 额外）

| 特征类型 | 方法 | 说明 |
|----------|------|------|
| 钙化特征 | Laplacian 高频细节 | 3 维，用于区分钙化点 |
| 肿块形状 | circularity, solidity, density | 3 维，用于区分肿块形状 |

```python
extract_features(patch_norm, extended=False)        # → 1D float32 向量
extract_features_batch(patches, extended, verbose)  # → shape (N, D)
```

---

### `train_stage1.py` — Stage-1 训练

**模型**：`StandardScaler → PCA(80) → SVC(rbf, probability=True, class_weight='balanced')`

**流程**：

1. 5 折 GridSearchCV，scoring=recall，param_grid: `C=[0.1,1,10]`, `gamma=['scale','auto']`
2. 在 OOF 预测上调优阈值：在 `[0.1, 0.7]` 区间寻找使 recall ≥ 0.90 且 F1 最大的点

**保存**：`output/models/stage1_pipeline.pkl`, `output/models/stage1_threshold.pkl`

**接口**：

```python
pipeline, threshold = train_stage1(X, y, model_type="svm")
pipeline, threshold = load_stage1()
labels, proba = predict_stage1(pipeline, X, threshold)
```

---

### `train_stage2.py` — Stage-2 训练

**模型**：XGBClassifier（`multi:softmax`，400 棵树，lr=0.05）

**流程**：

1. `LabelEncoder` 将原始类别索引（0–9）映射为连续编码
2. `StandardScaler` 特征归一化
3. 5 折交叉验证计算 OOF 准确率

**保存**：`output/models/stage2_model.pkl`, `output/models/stage2_scaler.pkl`, `output/models/stage2_label_encoder.pkl`

> 注意：调用前需过滤掉 `stage2_label == -1` 的负样本

**接口**：

```python
model, scaler, le = train_stage2(X, y, model_type="xgboost")
model, scaler, le = load_stage2()
label_indices, label_names = predict_stage2(model, scaler, le, X)
```

---

### `evaluate.py` — 评估报告

| 函数 | 输出 |
|------|------|
| `evaluate_stage1(y_true, y_pred, proba, tag)` | precision / recall / F1 / ROC-AUC |
| `evaluate_stage2(y_true, y_pred, class_names, tag)` | accuracy / Cohen Kappa / Macro-F1 + confusion matrix |
| `evaluate_pipeline(y_true_s1, y_pred_s1, y_true_s2, y_pred_s2, proba_s1)` | 端对端联合评估 |

报告保存在 `output/reports/`：`<tag>_metrics.json`, `<tag>_report.txt`, `<tag>_confusion_matrix.png`

---

### `run_pipeline.py` — 主入口

```bash
# 完整流程（含 mask 生成）
python src/data/recognition-traditional/run_pipeline.py

# 跳过 mask 生成（masks 已存在）
python src/data/recognition-traditional/run_pipeline.py --skip-mask

# 仅训练 Stage-1
python src/data/recognition-traditional/run_pipeline.py --stage 1

# 仅训练 Stage-2（需已有 Stage-1 模型）
python src/data/recognition-traditional/run_pipeline.py --stage 2

# 仅评估（加载已保存模型）
python src/data/recognition-traditional/run_pipeline.py --eval-only

# 强制重新生成 masks
python src/data/recognition-traditional/run_pipeline.py --force-mask
```

---

## 四、评估指标体系

| 阶段 | 核心指标 | 说明 |
|------|----------|------|
| Stage-1 | Recall, F1 | 最小化漏诊（假阴性）|
| Stage-2 | Cohen Kappa, Macro-F1 | 多类别分类一致性 |
| 端对端 | ROC-AUC | 全流程判别能力 |

---

## 五、设计要点

1. **难负样本挖掘**：不随机采样，而是从每张图像最亮最致密的正常腺体区域裁剪，强迫模型学习"致密腺体"与"病灶"的微观纹理差异。

2. **阈值自动调优**：Stage-1 在训练后自动扫描概率阈值区间，以满足 recall ≥ 0.90 约束的前提下最大化 F1，无需手动调参。

3. **特征缓存**：基础特征矩阵（用于 Stage-1）在首次提取后保存为 `.npy`，再次运行时直接加载，避免重复计算。

4. **掩码生成**：`data/segmented/` 初始为空目录，`Step 0` 必须调用 `generate_masks()` 完成乳腺区域分割后才能进行后续采样。

5. **外部代码依赖**：`sampling.py` 的 `generate_masks()` 在运行时会动态导入 `src/data/segment/segment.py` 中的 `build_training_masks_from_csv`。这是整个传统 ML 模块**唯一的外部代码依赖**。若使用 `--skip-mask` 跳过掩码生成步骤，则该依赖完全不会被触发，其余所有模块（`preprocessing.py`, `features.py`, `train_stage1.py`, `train_stage2.py`, `evaluate.py`）均自包含，不引用 `src/data/` 下任何其他目录的代码。

6. **路径兼容性**：`image_id` 字段已包含 `.png` 后缀，直接拼接 `IMAGES_ROOT / patient_id / image_id` 即可，无需额外处理。
