"""
use_example.py — use.py 使用示例

直接执行此脚本即可查看完整推理输出：
    python src/data/deep-learning/use_example.py

示例演示以下三种用法：
  1. 使用 MammoPearlPredictor 类（推荐，模型只加载一次）
  2. 使用 predict() 顶层便捷函数（每次重新加载模型）
  3. 以字节流方式传入图片
"""

from __future__ import annotations

from pathlib import Path

# ── 导入推理接口 ─────────────────────────────────────────────────────────────
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from use import MammoPearlPredictor, predict  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 固定配置（根据实际路径修改）
# ──────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[3]   # MammoPearl-Training/

STAGE1_CKPT = str(_ROOT / "models" / "clf_efficientnet_b4.pth")
STAGE2_CKPT = str(_ROOT / "models" / "clf2_cond_efficientnet_b4.pth")
IMAGES_ROOT = str(_ROOT / "data" / "processed" / "images_png")

# 测试集示例图像（patient_id / image_id.png）
EXAMPLE_MASS = (
    "81885e37a9a42bd351516b83bc3f6d66",
    "6e0cfd8cdae637f4613868dbbc34b453.png",
)
EXAMPLE_CALC = (
    "f9b9d3c214c10fa252823cd4aaab269e",
    "1cbc0ae5d67abccd58a4ba6d657d921e.png",
)
EXAMPLE_ASYM = (
    "115666dcca5a3ecb52368efc61da2578",
    "162908641d0d783855f67d93c64ec900.png",
)
EXAMPLE_NEGATIVE = (
    "522963c771a84cb777c49ba7a4ca69fc",
    "04c9b9305ae9a552975cceab2c15632e.png",
)

SEP = "─" * 66


def _img_path(patient_id: str, image_id: str) -> str:
    return str(Path(IMAGES_ROOT) / patient_id / image_id)


def _label(patient_id: str, image_id: str) -> str:
    """根据测试集固定标签返回期望类别（仅供示例说明）。"""
    mapping = {
        EXAMPLE_MASS[1]:     "Mass（肿块）   — Stage2预测：Mass",
        EXAMPLE_CALC[1]:     "Calcification（钙化）— Stage2预测：Calcification",
        EXAMPLE_ASYM[1]:     "Asymmetry_Distortion（不对称/扭曲）— Stage2预测：Asym",
        EXAMPLE_NEGATIVE[1]: "No Finding（阴性）   — Stage1 拒绝",
    }
    return mapping.get(image_id, "未知")


def _print_result(
    tag: str,
    result,
    *,
    expected: tuple[str, str] | None = None,
) -> None:
    if expected is None:
        expected = tuple(tag.split('/')[-2:])
    print(f"\n[{tag}]  期望：{_label(*expected)}")
    print(f"  has_lesion   : {result.has_lesion}")
    print(f"  stage1_prob  : {result.stage1_prob:.4f}")
    if result.has_lesion:
        print(f"  lesion_type  : {result.lesion_type}  (id={result.lesion_type_id})")
        for name, prob in result.lesion_type_probs.items():
            bar = "█" * int(prob * 30)
            print(f"    {name:<25} {prob:.4f}  {bar}")
    else:
        print("  （Stage 2 未运行，概率均为 0.0）")


# ──────────────────────────────────────────────────────────────────────────────
# 示例 1：MammoPearlPredictor 类（推荐用法）
# ──────────────────────────────────────────────────────────────────────────────

print(SEP)
print("示例 1：MammoPearlPredictor 类（模型加载一次，批量推理）")
print(SEP)
print(f"加载模型中...")
predictor = MammoPearlPredictor(
    stage1_ckpt=STAGE1_CKPT,
    stage2_ckpt=STAGE2_CKPT,
    stage1_threshold=0.1,       # Stage 1 阳性判定阈值，默认 0.1
)
print("模型加载完成。\n")

images = [EXAMPLE_MASS, EXAMPLE_CALC, EXAMPLE_ASYM, EXAMPLE_NEGATIVE]
for pid, iid in images:
    path = _img_path(pid, iid)
    result = predictor.predict(image=path)  # image 必须通过关键字参数传入
    _print_result(f"{pid}/{iid}", result)

# ──────────────────────────────────────────────────────────────────────────────
# 示例 2：字节流传入
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("示例 2：以字节流（bytes）传入图片")
print(SEP)

pid, iid = EXAMPLE_MASS
path = _img_path(pid, iid)
with open(path, "rb") as f:
    image_bytes = f.read()

result = predictor.predict(image=image_bytes)   # 传入 bytes，结果与路径方式完全相同
_print_result(f"{pid}/{iid} (bytes)", result, expected=(pid, iid))

# ──────────────────────────────────────────────────────────────────────────────
# 示例 3：predict() 顶层便捷函数
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("示例 3：predict() 顶层便捷函数（每次重新加载模型，适合单次脚本调用）")
print(SEP)

pid, iid = EXAMPLE_CALC
result = predict(
    stage1_ckpt=STAGE1_CKPT,
    stage2_ckpt=STAGE2_CKPT,
    image=_img_path(pid, iid),  # image 必须通过关键字参数传入
    stage1_threshold=0.1,
)
_print_result(f"{pid}/{iid}", result)

# ──────────────────────────────────────────────────────────────────────────────
# 示例 4：访问结果字段
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("示例 4：PredictionResult 字段访问")
print(SEP)

pid, iid = EXAMPLE_MASS
result = predictor.predict(image=_img_path(pid, iid))

print(f"result.has_lesion        = {result.has_lesion}")
print(f"result.stage1_prob       = {result.stage1_prob:.6f}")
print(f"result.lesion_type       = {result.lesion_type!r}")
print(f"result.lesion_type_id    = {result.lesion_type_id}")
print(f"result.lesion_type_probs = {result.lesion_type_probs}")
print(f"repr(result)             = {result!r}")

print(f"\n{SEP}")
print("完成。")
print(SEP)
