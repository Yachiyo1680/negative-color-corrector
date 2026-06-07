"""
色罩分析模块 — Mask Analyzer

从负片扫描图中检测 C-41 橙红色罩，计算各通道的补偿比例。

三种采样策略：
  A → 边缘片基采样（最准确）
  B → 画面内中性灰检测（边缘被裁时用）
  C → 全局灰世界假设（最终 fallback）
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass
class MaskResult:
    """色罩分析结果"""
    method: str                 # 使用的采样策略: "edge" | "gray" | "global"
    ref_r: float                # 参考点 R 均值
    ref_g: float
    ref_b: float
    scale_r: float             # R 通道补偿比例
    scale_g: float             # G 通道补偿比例
    scale_b: float             # B 通道补偿比例
    confidence: float          # 置信度 0~1
    detail: str = ""           # 额外说明


def analyze_mask(image: np.ndarray) -> MaskResult:
    """分析图像色罩，返回补偿比例

    传入的 image 应是反相后的图像（float32, 0-255）。

    Args:
        image: 反相后的 RGB 图像 (H, W, 3), float32

    Returns:
        MaskResult 包含补偿比例
    """
    h, w = image.shape[:2]

    # ── 策略 A: 边缘片基采样 ──
    edge_width = max(int(w * 0.05), 20)  # 左侧 5% 列
    edge_region = image[:, :edge_width, :]
    edge_mean = np.mean(edge_region, axis=(0, 1))

    # 检查边缘是否确实为片基（低方差、偏橙色）
    edge_std = np.std(edge_region, axis=(0, 1)).mean()
    if edge_std < 30:
        return _compute_from_ref(image, edge_mean, method="edge",
                                 confidence=0.95,
                                 detail=f"片基均值 RGB=({edge_mean[0]:.0f},{edge_mean[1]:.0f},{edge_mean[2]:.0f})")

    # ── 策略 B: 画面内中性灰检测 ──
    # 在画面下半部分搜索亮度中等、饱和度最低的区域
    bottom_half = image[h//2:, :, :]
    gray_pixels = _find_gray_pixels(bottom_half)

    if np.sum(gray_pixels) > 200:
        gray_region = bottom_half[gray_pixels]
        gray_mean = np.mean(gray_region, axis=0)
        gray_std = np.std(gray_region, axis=0).mean()
        conf = max(0.6, 1.0 - gray_std / 60)
        return _compute_from_ref(image, gray_mean, method="gray",
                                 confidence=conf,
                                 detail=f"中性灰区域均值 RGB=({gray_mean[0]:.0f},{gray_mean[1]:.0f},{gray_mean[2]:.0f})")

    # ── 策略 C: 全局灰世界假设 ──
    global_mean = np.mean(image, axis=(0, 1))
    return _compute_from_ref(image, global_mean, method="global",
                             confidence=0.5,
                             detail=f"全局均值 RGB=({global_mean[0]:.0f},{global_mean[1]:.0f},{global_mean[2]:.0f})")


def _compute_from_ref(
    image: np.ndarray,
    ref_rgb: np.ndarray,
    method: str,
    confidence: float,
    detail: str = "",
) -> MaskResult:
    """根据参考点 RGB 计算补偿比例

    在反相空间做中性化：
        inv_R = ref_R
        inv_G = ref_G
        inv_B = ref_B
        target = (inv_R + inv_G + inv_B) / 3
        scale_R = target / inv_R (如果 inv_R > 0)

    Args:
        image: 原图（仅用于记录，不修改）
        ref_rgb: 参考点 RGB 均值 (3,)
        method: 策略名
        confidence: 置信度
        detail: 说明文字

    Returns:
        MaskResult
    """
    ref_r, ref_g, ref_b = ref_rgb

    target = (ref_r + ref_g + ref_b) / 3.0

    scale_r = target / ref_r if ref_r > 0 else 1.0
    scale_g = target / ref_g if ref_g > 0 else 1.0
    scale_b = target / ref_b if ref_b > 0 else 1.0

    return MaskResult(
        method=method,
        ref_r=ref_r,
        ref_g=ref_g,
        ref_b=ref_b,
        scale_r=scale_r,
        scale_g=scale_g,
        scale_b=scale_b,
        confidence=confidence,
        detail=detail,
    )


def _find_gray_pixels(img: np.ndarray) -> np.ndarray:
    """提取亮度适中、饱和度低的像素（可能是中性灰）"""
    gray = np.mean(img, axis=2)
    sat = (
        np.abs(img[:, :, 0].astype(float) - gray)
        + np.abs(img[:, :, 1].astype(float) - gray)
        + np.abs(img[:, :, 2].astype(float) - gray)
    )
    brightness_mask = (gray > 30) & (gray < 220)
    sat_mask = (sat / 3) < 30
    return brightness_mask & sat_mask
