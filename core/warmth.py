"""
暖调控制模块 — Warmth Controller

色罩补偿后的画面通常会偏冷微蓝，需要加暖。

关键原则：
  ✅ 黄方向 = R↑ + G↑（红绿一起加 → 黄色调，自然）
  ❌ 品红方向 = R↑ + B↑（红蓝一起加 → 容易偏紫）
"""

import numpy as np
from typing import Literal

WarmthStyle = Literal["none", "natural", "kodak_gold", "fuji_superia", "cool"]


# 暖调预设 (R偏移, G偏移, B偏移) 百分比
_WARMTH_PRESETS = {
    "none":         (0.00,  0.00,  0.00),  # 无色偏，最自然的颜色
    "natural":     (0.04,  0.02, -0.06),   # 自然暖调
    "kodak_gold":  (0.06,  0.03, -0.08),   # Kodak Gold 风格
    "fuji_superia": (0.03, 0.01, -0.05),   # Fuji Superia 风格
    "cool":        (0.00,  0.00, -0.02),   # 冷调
}


def apply_warmth(
    image: np.ndarray,
    style: WarmthStyle = "natural",
    strength: float = 1.0,
) -> np.ndarray:
    """对校色后图像应用暖调调整

    Args:
        image: RGB 图像 (H, W, 3), float32 (0-255)
        style: 暖调风格
        strength: 强度系数 0.0（无效果）~ 2.0，默认 1.0

    Returns:
        暖调调整后图像
    """
    if style not in _WARMTH_PRESETS:
        style = "natural"

    dr, dg, db = _WARMTH_PRESETS[style]

    result = image.copy().astype(np.float32)

    result[:, :, 0] *= (1.0 + dr * strength)  # R
    result[:, :, 1] *= (1.0 + dg * strength)  # G
    result[:, :, 2] *= (1.0 + db * strength)  # B

    return np.clip(result, 0, 255)


def get_warmth_presets() -> list[dict]:
    """返回暖调预设列表（供 UI 使用）"""
    return [
        {"id": "none",          "label": "无色偏",         "emoji": "⚪"},
        {"id": "natural",      "label": "自然暖调",       "emoji": "🌅"},
        {"id": "kodak_gold",   "label": "Kodak Gold",     "emoji": "🟡"},
        {"id": "fuji_superia", "label": "Fuji Superia",   "emoji": "🟢"},
        {"id": "cool",         "label": "冷调",           "emoji": "❄️"},
    ]
