# 传统机器学习与深度学习路线对比

> 数据集：VinDr Mammography（经处理版本）
>
> 对比对象：`src/data/recognition-traditional/` 与 `src/data/deep-learning/`

---

## 一、对比范围

本文只保留两条路线的**最终成果对照**。

- 传统机器学习：patch 级两阶段分类流程
- 深度学习：全图级两阶段分类流程

两条路线的输入粒度不同，因此这里重点展示各自最终结果，不展开讨论横向评估细节。

---

## 二、路线概览

| 项目 | 传统机器学习 | 深度学习 |
|------|--------------|----------|
| 输入 | 128×128 patch | 512×512 全图 |
| Stage 1 | SVM 二分类 | EfficientNet-B4 二分类 |
| Stage 2 | XGBoost 4 类分类 | EfficientNet-B4 3 类条件分类 |
| Stage 2 类别 | Asymmetry_Distortion / Mass / Skin_Other / Suspicious_Calcification | Mass / Calcification / Asymmetry_Distortion |

---

## 三、Stage 1 最终成果

### 传统机器学习（测试集 12,447 个 patch）

| 指标 | 值 |
|------|-----|
| Precision | 0.9932 |
| Recall | **0.9843** |
| F1 | 0.9888 |
| ROC-AUC | 1.0000 |

### 深度学习（测试集 4,000 张全图）

| 阈值 | Recall | Precision | F1 | F2 | TP | FP | FN |
|------|--------|-----------|-----|-----|----|----|----|
| 0.10 | **0.9552** | 0.1080 | 0.1941 | 0.3719 | 341 | 2816 | 16 |
| 0.40 | 0.6331 | 0.2430 | 0.3512 | **0.4792** | 226 | 704 | 131 |

说明：

- 传统机器学习 Stage 1 的最终召回率为 **0.9843**。
- 深度学习 Stage 1 在高召回工作点（阈值 0.10）下最终召回率为 **0.9552**。

---

## 四、Stage 2 最终成果

### 传统机器学习（测试集 440 个正样本）

| 病变类型 | Precision | Recall | F1 | 测试样本数 |
|---------|-----------|--------|----|-----------|
| Asymmetry_Distortion | 0.50 | 0.25 | 0.33 | 102 |
| Mass | 0.65 | 0.88 | **0.75** | 232 |
| Skin_Other | 0.80 | 0.40 | 0.53 | 20 |
| Suspicious_Calcification | 0.75 | 0.57 | 0.65 | 86 |

| 整体指标 | 值 |
|---------|-----|
| Accuracy | **0.6523** |
| Cohen Kappa | 0.3859 |
| Macro F1 | **0.5652** |

### 深度学习（测试集，Stage 1 过滤 @0.10，341/357 通过）

| 类别 | N | Recall | Precision | F1 |
|------|---|--------|-----------|-----|
| Mass | 185 | 0.622 | 0.575 | 0.597 |
| Calcification | 72 | 0.417 | 0.297 | 0.347 |
| Asymmetry_Distortion | 84 | 0.119 | 0.250 | 0.161 |
| **Macro** | — | — | — | **0.368** |

端到端正确分类率（Stage 1 通过且 Stage 2 类型正确）：

- Mass：58.4%
- Calcification：40.5%
- Asymmetry_Distortion：11.6%

---

## 五、最终结果摘要

| 结果项 | 传统机器学习 | 深度学习 |
|--------|--------------|----------|
| Stage 1 最终 Recall | **0.9843** | **0.9552**（阈值 0.10） |
| Stage 2 最终 Macro F1 | **0.5652** | **0.368** |
| Stage 2 最优类别 | Mass，F1=0.75 | Mass，F1=0.597 |
| Stage 2 最难类别 | Asymmetry_Distortion，F1=0.33 | Asymmetry_Distortion，F1=0.161 |

一句话总结：传统机器学习在 **patch 级**最终指标上更高；深度学习则给出了当前**全图级**两阶段流程的最终结果基线。