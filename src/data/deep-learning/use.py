r"""
MammoPearl 推理接口（use.py）

两阶段乳腺 X 光病变检测：
  Stage 1：全图二分类（有无病变，BCEWithLogitsLoss 单 logit）
  Stage 2：条件病变类型分类（Mass / Calcification / Asymmetry_Distortion，
           仅在 Stage 1 判断为阳性时运行）

独立性说明：
    本文件是“项目内推理接口”，可脱离训练数据集、CSV 标注和 dataset.py 单独用于推理；
    但它并不是“只保留 use.py 一个文件即可运行”的单文件脚本。
    运行时仍依赖：
        1. Stage 1 / Stage 2 checkpoint（.pth）
        2. 同目录下的 clf-train.py 与 clf2-train.py（用于动态构建模型结构）
        3. torch / numpy / opencv-python 运行环境

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
快速上手
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from src.data.deep-learning.use import MammoPearlPredictor

    # 初始化（模型只加载一次，适合批量处理）
    predictor = MammoPearlPredictor(
        stage1_ckpt="models/clf_efficientnet_b4.pth",
        stage2_ckpt="models/clf2_cond_efficientnet_b4.pth",
    )

    # 图片路径方式
    result = predictor.predict(image="data/processed/images_png/xxx/yyy.png")

    # 图片字节方式
    with open("image.png", "rb") as f:
        result = predictor.predict(image=f.read())

    # 访问结果
    print(result.has_lesion)           # bool：Stage 1 是否检出病变
    print(result.stage1_prob)          # float：Stage 1 病变概率（0~1）
    print(result.lesion_type)          # str|None："Mass"/"Calcification"/"Asymmetry_Distortion"/None
    print(result.lesion_type_id)       # int|None：0=Mass, 1=Calc, 2=Asym, None=无病变
    print(result.lesion_type_probs)    # dict：三类概率（has_lesion=False 时均为 0.0）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
顶层便捷函数（每次调用均重新加载模型，适合单次脚本调用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from src.data.deep-learning.use import predict

    result = predict(
        stage1_ckpt="models/clf_efficientnet_b4.pth",
        stage2_ckpt="models/clf2_cond_efficientnet_b4.pth",
        image="path/to/image.png",        # 或 image=bytes_data
        stage1_threshold=0.1,             # 可选，默认 0.1（高召回）
        device="cuda",                    # 可选，默认自动检测
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MammoPearlPredictor(
    stage1_ckpt        : str        Stage 1 检查点文件路径
    stage2_ckpt        : str        Stage 2 检查点文件路径
    stage1_threshold   : float      Stage 1 阳性判定阈值，默认 0.1
    device             : str|None   "cuda"/"cpu"/None（自动检测）
)

predictor.predict(image=...) / predict(..., image=...) 的 image 参数：
    str   → 图片文件路径（PNG / JPEG / BMP 均可）
    bytes → 图片原始字节数据
    image 必须通过关键字参数传入。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
返回值：PredictionResult
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    has_lesion        bool        Stage 1 判断结果（prob >= threshold）
    stage1_prob       float       Stage 1 输出的病变概率（0~1）
    lesion_type       str|None    预测病变类型；has_lesion=False 时为 None
    lesion_type_id    int|None    0=Mass, 1=Calcification, 2=Asymmetry_Distortion；
                                  has_lesion=False 时为 None
    lesion_type_probs dict        {"Mass": float, "Calcification": float,
                                   "Asymmetry_Distortion": float}；
                                  has_lesion=False 时三项均为 0.0
"""

from __future__ import annotations

import importlib.util as _ilu
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LESION_TYPE_NAMES = {0: "Mass", 1: "Calcification", 2: "Asymmetry_Distortion"}

_DEFAULT_S1_CKPT = str(Path("models") / "clf_efficientnet_b4.pth")
_DEFAULT_S2_CKPT = str(Path("models") / "clf2_cond_efficientnet_b4.pth")


# ──────────────────────────────────────────────────────────────────────────────
# 返回值类型
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """单张图像的检测结果。"""

    has_lesion: bool
    """Stage 1 判断结果：prob >= threshold 为 True。"""

    stage1_prob: float
    """Stage 1 输出的病变概率（0~1）。"""

    lesion_type: str | None = None
    """预测的病变类型名称；has_lesion=False 时为 None。
    可能的值：'Mass' / 'Calcification' / 'Asymmetry_Distortion'"""

    lesion_type_id: int | None = None
    """病变类型 ID（0=Mass, 1=Calcification, 2=Asymmetry_Distortion）；
    has_lesion=False 时为 None。"""

    lesion_type_probs: dict[str, float] = field(default_factory=lambda: {
        "Mass": 0.0,
        "Calcification": 0.0,
        "Asymmetry_Distortion": 0.0,
    })
    """Stage 2 输出的各病变类型概率；has_lesion=False 时三项均为 0.0。"""

    def __repr__(self) -> str:  # noqa: D105
        if not self.has_lesion:
            return f"PredictionResult(has_lesion=False, stage1_prob={self.stage1_prob:.4f})"
        probs = ", ".join(f"{k}={v:.4f}" for k, v in self.lesion_type_probs.items())
        return (
            f"PredictionResult(has_lesion=True, stage1_prob={self.stage1_prob:.4f}, "
            f"lesion_type='{self.lesion_type}', probs=[{probs}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _load_module(name: str, file_name: str):
    """动态导入同目录下的脚本（文件名含连字符）。"""
    spec = _ilu.spec_from_file_location(name, _HERE / file_name)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preprocess_image(
    image: Union[str, bytes],
    input_h: int,
    input_w: int,
) -> torch.Tensor:
    """将图片（路径或字节）预处理为模型输入 Tensor。

    流程与 MammoDataset 完全一致：
      1. 解码为灰度图
      2. 等比缩放 + 中心 letterbox 填充
      3. 归一化到 uint8 [0, 255]
      4. 复制为 3 通道 RGB
      5. ImageNet 归一化 → float32 Tensor (3, H, W)
    """
    if isinstance(image, str):
        raw = np.fromfile(image, dtype=np.uint8)
    elif isinstance(image, bytes):
        raw = np.frombuffer(image, dtype=np.uint8)
    else:
        raise TypeError(f"image 须为 str（路径）或 bytes，实际类型：{type(image).__name__}")

    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("图像解码失败，请确认文件格式正确。")

    # 转灰度
    if img.ndim == 2:
        gray = img
    elif img.shape[2] >= 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 等比缩放 + letterbox 填充（与 _load_gray 完全相同）
    h, w  = gray.shape
    scale = min(input_h / h, input_w / w)
    nh    = int(round(h * scale))
    nw    = int(round(w * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas   = np.zeros((input_h, input_w), dtype=resized.dtype)
    pad_top  = (input_h - nh) // 2
    pad_left = (input_w - nw) // 2
    canvas[pad_top: pad_top + nh, pad_left: pad_left + nw] = resized

    if canvas.dtype != np.uint8:
        canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 灰度 → RGB 3 通道
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)  # (H, W, 3) uint8

    # ImageNet 归一化
    img_f = rgb.astype(np.float32) / 255.0
    img_f = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1))  # (3, H, W) float32
    return tensor


def _load_stage1_model(ckpt_path: str, device: torch.device):
    """加载 Stage 1 模型及其超参数。返回 (model, input_h, input_w)。"""
    clf_train = _load_module("clf_train", "clf-train.py")

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    input_h   = ckpt_args.get("input_h", 512)
    input_w   = ckpt_args.get("input_w", 512)
    in_ch     = ckpt_args.get("in_channels", 3)

    model = clf_train.build_model(pretrained=False, in_channels=in_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, input_h, input_w


def _load_stage2_model(ckpt_path: str, device: torch.device):
    """加载 Stage 2 模型及其超参数。返回 (model, input_h, input_w)。"""
    clf2_train  = _load_module("clf2_train", "clf2-train.py")

    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args   = ckpt.get("args", {})
    input_h     = ckpt_args.get("input_h", 512)
    input_w     = ckpt_args.get("input_w", 512)
    num_classes = ckpt.get("num_classes", 3)

    model = clf2_train.build_stage2_model(pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, input_h, input_w


# ──────────────────────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────────────────────

class MammoPearlPredictor:
    """两阶段乳腺 X 光病变检测器。

    建议在程序启动时初始化一次（模型加载耗时约 1~3 秒），
    之后重复调用 predict() 进行推理。

    Parameters
    ----------
    stage1_ckpt        : Stage 1 检查点路径
    stage2_ckpt        : Stage 2 检查点路径
    stage1_threshold   : Stage 1 阳性判定阈值（默认 0.1）
    device             : "cuda" / "cpu" / None（自动检测）
    """

    def __init__(
        self,
        stage1_ckpt: str = _DEFAULT_S1_CKPT,
        stage2_ckpt: str = _DEFAULT_S2_CKPT,
        stage1_threshold: float = 0.1,
        device: str | None = None,
    ) -> None:
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._threshold = stage1_threshold

        self._s1_model, self._s1_h, self._s1_w = _load_stage1_model(
            stage1_ckpt, self._device
        )
        self._s2_model, self._s2_h, self._s2_w = _load_stage2_model(
            stage2_ckpt, self._device
        )

    def predict(self, *, image: Union[str, bytes]) -> PredictionResult:
        """对单张图像进行两阶段检测。

        Parameters
        ----------
        image : str 或 bytes（必须通过关键字参数传入）
            str  → 图片文件路径（PNG / JPEG / BMP 等 cv2 支持的格式）
            bytes → 图片原始字节数据

        Returns
        -------
        PredictionResult
        """
        # ── Stage 1 ─────────────────────────────────────────────────────────
        t1 = _preprocess_image(image, self._s1_h, self._s1_w)
        inp1 = t1.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logit = self._s1_model(inp1).squeeze()          # scalar logit
            s1_prob = torch.sigmoid(logit.float()).item()

        has_lesion = s1_prob >= self._threshold

        if not has_lesion:
            return PredictionResult(
                has_lesion=False,
                stage1_prob=s1_prob,
            )

        # ── Stage 2 ─────────────────────────────────────────────────────────
        t2 = _preprocess_image(image, self._s2_h, self._s2_w)
        inp2 = t2.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._s2_model(inp2)                   # (1, 3)
            probs  = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()

        pred_id   = int(np.argmax(probs))
        pred_name = _LESION_TYPE_NAMES[pred_id]

        return PredictionResult(
            has_lesion=True,
            stage1_prob=s1_prob,
            lesion_type=pred_name,
            lesion_type_id=pred_id,
            lesion_type_probs={
                "Mass":                 float(probs[0]),
                "Calcification":        float(probs[1]),
                "Asymmetry_Distortion": float(probs[2]),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 顶层便捷函数
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    *,
    stage1_ckpt: str = _DEFAULT_S1_CKPT,
    stage2_ckpt: str = _DEFAULT_S2_CKPT,
    image: Union[str, bytes],
    stage1_threshold: float = 0.1,
    device: str | None = None,
) -> PredictionResult:
    """单次推理的便捷函数（每次调用均重新加载模型）。

    适合脚本级单次调用；若需批量处理，请使用 MammoPearlPredictor 类。

    所有参数均为关键字参数。

    Parameters
    ----------
    stage1_ckpt      : Stage 1 检查点路径
    stage2_ckpt      : Stage 2 检查点路径
    image            : str（路径）或 bytes（字节数据）
    stage1_threshold : Stage 1 阳性判定阈值，默认 0.1
    device           : "cuda" / "cpu" / None（自动检测）

    Returns
    -------
    PredictionResult
    """
    predictor = MammoPearlPredictor(
        stage1_ckpt=stage1_ckpt,
        stage2_ckpt=stage2_ckpt,
        stage1_threshold=stage1_threshold,
        device=device,
    )
    return predictor.predict(image=image)
