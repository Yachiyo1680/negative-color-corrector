"""
校色引擎主入口 — Engine

整合所有模块：
  反相 → 色罩分析 → 通道补偿 → 智能色阶 → 暖调控制 → AI检测(闭环)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

from .invert import invert
from .mask_analyzer import analyze_mask, MaskResult
from .channel_comp import apply_channel_compensation
from .auto_levels import auto_levels
from .warmth import apply_warmth, WarmthStyle
from .cast_detector import DetectorFactory, CastDetector, CastResult


@dataclass
class CorrectionConfig:
    """校色参数配置"""
    film_type: str = "negative"        # "negative" | "positive"
    warmth_style: WarmthStyle = "natural"
    warmth_strength: float = 1.0       # 0.0 (不暖) ~ 2.0 (超暖)
    levels_percentile: float = 0.2     # 色阶裁剪百分位
    max_iterations: int = 10           # AI反馈最大迭代次数
    cast_threshold: float = 0.15       # 偏色容忍度 (低于此值视为OK)
    detector_mode: str = "auto"        # "heuristic" | "onnx" | "vlm_api" | "auto"


@dataclass
class CorrectionResult:
    """校色结果"""
    image: np.ndarray           # 校色后的 RGB 图像
    mask_info: MaskResult       # 色罩分析信息
    iterations: int             # AI反馈迭代次数
    final_cast: CastResult      # 最终偏色检测结果
    warm_style: str             # 使用的暖调风格


class Engine:
    """校色引擎"""

    def __init__(self, config: CorrectionConfig = None):
        self.config = config or CorrectionConfig()
        self.detector: Optional[CastDetector] = None

    def _get_detector(self) -> CastDetector:
        """延迟初始化偏色检测器"""
        if self.detector is None:
            self.detector = DetectorFactory.create({
                "detector_mode": self.config.detector_mode,
            })
        return self.detector

    def correct(self, image: np.ndarray) -> CorrectionResult:
        """
        执行完整校色流程。

        参数:
            image: RGB 图像 (H, W, 3), uint8 0-255

        返回:
            CorrectionResult
        """
        img = image.copy().astype(np.float32)
        h, w = img.shape[:2]

        # ── Step 1: 反相（仅负片） ──
        if self.config.film_type == "negative":
            img = invert(img)
            print("[Engine] 反相完成")
        else:
            print("[Engine] 正片模式，跳过反相")

        # ── Step 2: 色罩分析 ──
        mask_info = analyze_mask(img)
        print(f"[Engine] 色罩分析: {mask_info}")

        # ── Step 3: 通道补偿 ──
        img = apply_channel_compensation(img, mask_info)
        print(f"[Engine] 通道补偿完成: "
              f"R×{mask_info.scale_r:.3f} "
              f"G×{mask_info.scale_g:.3f} "
              f"B×{mask_info.scale_b:.3f}")

        # ── Step 4: 智能色阶 ──
        img = auto_levels(img, percentile=self.config.levels_percentile)
        print("[Engine] 智能色阶完成")

        # ── Step 5: 暖调控制 ──
        img = apply_warmth(img, style=self.config.warmth_style,
                           strength=self.config.warmth_strength)
        print(f"[Engine] 暖调完成: {self.config.warmth_style}")

        # ── Step 6: AI 偏色检测 + 反馈闭环 ──
        detector = self._get_detector()
        img_out = img.copy()
        final_cast = CastResult("ok", 0, 0, 1.0)
        # 累积调整比例（防止振荡）
        accum_r, accum_g, accum_b = 1.0, 1.0, 1.0

        for i in range(self.config.max_iterations):
            # 检测偏色
            cast = detector.detect(img_out.astype(np.uint8))

            if cast.is_ok or cast.severity < self.config.cast_threshold:
                print(f"[Engine] 偏色检测 OK (迭代 {i+1})")
                final_cast = cast
                break

            # 根据偏色方向微调（带阻尼系数，迭代越深调整越小）
            damping = max(0.95 ** i, 0.5)  # 逐次衰减，最小保留 50%
            img_out, adj_r, adj_g, adj_b = self._adjust_for_cast(
                img_out, cast, damping
            )
            accum_r *= adj_r
            accum_g *= adj_g
            accum_b *= adj_b
            print(f"[Engine] 迭代 {i+1}: 检测到 {cast.cast_type} "
                  f"(severity={cast.severity:.3f}) "
                  f"累计调整 R={accum_r:.4f} G={accum_g:.4f} B={accum_b:.4f}")

        else:
            # 达到最大迭代次数
            final_cast = detector.detect(img_out.astype(np.uint8))
            print(f"[Engine] 达到最大迭代次数，使用最终结果")

        return CorrectionResult(
            image=np.clip(img_out, 0, 255).astype(np.uint8),
            mask_info=mask_info,
            iterations=i+1,
            final_cast=final_cast,
            warm_style=self.config.warmth_style,
        )

    def _adjust_for_cast(self, img: np.ndarray,
                         cast: CastResult,
                         damping: float = 1.0) -> tuple:
        """根据偏色检测结果微调通道比例

        Returns:
            (调整后图像, R调整系数, G调整系数, B调整系数)
        """
        result = img.copy().astype(np.float32)
        # 调整幅度 = severity 映射到 1%~5%，再乘阻尼
        magnitude = 1.0 + cast.severity * 0.04 * damping

        adjustments = {
            "blue":     (1.000, 1.000, 1/magnitude),  # 偏蓝 → 减蓝
            "cyan":     (1.000, 1.000, 1/magnitude),  # 偏青 → 减蓝
            "green":    (1.000, 1/magnitude, 1.000),  # 偏绿 → 减绿
            "yellow":   (1.000, 1.000, magnitude),    # 偏黄 → 加蓝
            "magenta":  (1.000, magnitude, 1.000),    # 偏品红 → 加绿
            "warm":     (1.000, 1.000, magnitude),    # 过暖 → 加蓝
            "cool":     (1.000, 1.000, 1/magnitude),  # 过冷 → 减蓝
        }

        adj_r, adj_g, adj_b = adjustments.get(cast.cast_type, (1, 1, 1))
        result[:,:,0] *= adj_r
        result[:,:,1] *= adj_g
        result[:,:,2] *= adj_b

        return result, adj_r, adj_g, adj_b
