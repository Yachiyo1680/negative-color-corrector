"""
通道补偿模块 — Channel Compensation

在色罩分析得到补偿比例后，对 RGB 各通道分别缩放。
校正原理：
  反相后的图像 = 正像 + 残留色罩
  对中性灰参考点做中性化处理，得到各通道的补偿系数。
"""

from dataclasses import dataclass
import numpy as np
from .mask_analyzer import MaskResult


def apply_channel_compensation(
    image: np.ndarray,
    mask: MaskResult,
) -> np.ndarray:
    """对反相后的图像应用通道补偿

    每个像素的 RGB 值乘以对应通道的补偿比例：
        result_R = clamp(input_R * scale_r, 0, 255)
        result_G = clamp(input_G * scale_g, 0, 255)
        result_B = clamp(input_B * scale_b, 0, 255)

    Args:
        image: 反相后的 RGB 图像 (H, W, 3), float32 (0-255)
        mask: 色罩分析结果（含 scale_r, scale_g, scale_b）

    Returns:
        通道补偿后的图像 (H, W, 3), float32
    """
    result = image.copy().astype(np.float32)

    result[:, :, 0] *= mask.scale_r  # R channel
    result[:, :, 1] *= mask.scale_g  # G channel
    result[:, :, 2] *= mask.scale_b  # B channel

    # 钳位到 [0, 255]
    return np.clip(result, 0, 255)


def manual_compensation(
    image: np.ndarray,
    scale_r: float,
    scale_g: float,
    scale_b: float,
) -> np.ndarray:
    """手动指定补偿比例的通道补偿

    用于 AI 反馈闭环中微调补偿比例。

    Args:
        image: 图像 (H, W, 3), float32
        scale_r, scale_g, scale_b: 各通道补偿系数

    Returns:
        补偿后图像
    """
    result = image.copy().astype(np.float32)
    result[:, :, 0] *= scale_r
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_b
    return np.clip(result, 0, 255)
