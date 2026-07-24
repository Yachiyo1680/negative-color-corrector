"""
反相模块 — Invert

负片扫描图的像素值 = 255 - scene_exposure
所以需要执行反相：pixel_out = 255 - pixel_in

仅负片模式下执行，正片模式直接跳过。
"""

import numpy as np


def invert(image: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """对 RGB 图像执行反相

    Args:
        image: RGB 图像 (H, W, 3), float32
        max_val: 像素最大值（255.0 或 65535.0）

    Returns:
        反相后的图像
    """
    return max_val - image


def invert_float(image: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """反相并保持 float 格式（用于链式处理）"""
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    return max_val - image
