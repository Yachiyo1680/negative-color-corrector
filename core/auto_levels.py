"""
智能色阶模块 — Auto Levels

每通道独立做直方图拉伸，恢复对比度。
避免色罩去除后画面发灰。
"""

import numpy as np
from typing import Tuple


def auto_levels(
    image: np.ndarray,
    percentile: float = 0.2,
) -> np.ndarray:
    """每通道独立拉伸直方图

    对各通道：
        lo = percentile(channel, p)
        hi = percentile(channel, 100 - p)
        new = clamp((channel - lo) * 255 / (hi - lo), 0, 255)

    Args:
        image: RGB 图像 (H, W, 3), float32 (0-255)
        percentile: 两侧裁剪百分位，默认 0.2%

    Returns:
        色阶拉伸后的图像 (H, W, 3), float32
    """
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    total_pixels = h * w

    for c in range(3):
        channel = result[:, :, c]

        # 计算百分位值
        sorted_vals = np.sort(channel.flatten())
        lo_idx = max(0, int(total_pixels * percentile / 100.0))
        hi_idx = min(total_pixels - 1,
                     int(total_pixels * (100 - percentile) / 100.0))

        lo = sorted_vals[lo_idx]
        hi = sorted_vals[hi_idx]

        # 防止除零
        if hi - lo < 1:
            hi = lo + 1

        # 拉伸
        stretched = (channel - lo) * 255.0 / (hi - lo)
        result[:, :, c] = np.clip(stretched, 0, 255)

    return result


def auto_levels_combined(
    image: np.ndarray,
    percentile: float = 0.2,
) -> np.ndarray:
    """三通道统一拉伸（保留相对色彩关系）"""
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    total_pixels = h * w

    # 计算亮度通道
    luminance = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
    sorted_lum = np.sort(luminance.flatten())

    lo_idx = max(0, int(total_pixels * percentile / 100.0))
    hi_idx = min(total_pixels - 1,
                 int(total_pixels * (100 - percentile) / 100.0))

    lo = sorted_lum[lo_idx]
    hi = sorted_lum[hi_idx]

    if hi - lo < 1:
        hi = lo + 1

    for c in range(3):
        stretched = (result[:, :, c] - lo) * 255.0 / (hi - lo)
        result[:, :, c] = np.clip(stretched, 0, 255)

    return result
