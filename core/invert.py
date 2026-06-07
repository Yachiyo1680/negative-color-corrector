"""
反相模块 — Invert

负片扫描图的像素值 = 255 - scene_exposure
所以需要执行反相：pixel_out = 255 - pixel_in

仅负片模式下执行，正片模式直接跳过。
"""

import numpy as np


def invert(image: np.ndarray) -> np.ndarray:
    """对 RGB 图像执行反相

    Args:
        image: RGB 图像 (H, W, 3), uint8 (0-255) 或 float32 (0-255)

    Returns:
        反相后的图像，与输入类型相同
    """
    if image.dtype == np.uint8:
        return (255 - image).astype(np.uint8)
    else:
        # float32 模式
        return 255.0 - image


def invert_float(image: np.ndarray) -> np.ndarray:
    """反相并保持 float 格式（用于链式处理）"""
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    return 255.0 - image
