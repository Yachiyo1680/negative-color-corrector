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
    # VLM API 配置（仅在 detector_mode="vlm_api" 时使用）
    vlm_api_key: str = ""
    vlm_api_base: str = ""
    vlm_model: str = "openai/gpt-4o-mini"


@dataclass
class CorrectionResult:
    """校色结果"""
    image: np.ndarray           # 校色后的 RGB 图像
    mask_info: MaskResult       # 色罩分析信息
    iterations: int             # AI反馈迭代次数
    final_cast: CastResult      # 最终偏色检测结果
    warm_style: str             # 使用的暖调风格
    detector_warning: str = ""  # 检测器回退警告（非空表示降级）


class Engine:
    """校色引擎"""

    def __init__(self, config: CorrectionConfig = None):
        self.config = config or CorrectionConfig()
        self.detector: Optional[CastDetector] = None
        self._detector_warning: str = ""

    def _get_detector(self) -> CastDetector:
        """延迟初始化偏色检测器"""
        if self.detector is None:
            self.detector, self._detector_warning = DetectorFactory.create({
                "detector_mode": self.config.detector_mode,
                "vlm_api_key": self.config.vlm_api_key,
                "vlm_api_base": self.config.vlm_api_base,
                "vlm_model": self.config.vlm_model,
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
        vlm_cfg = None
        if self.config.vlm_api_key:
            vlm_cfg = {
                "api_base": self.config.vlm_api_base,
                "api_key": self.config.vlm_api_key,
                "model": self.config.vlm_model,
                "timeout": 30,
            }
        mask_info = analyze_mask(img, vlm_config=vlm_cfg)
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

        # ── Step 6: AI 偏色检测 + 修正 ──
        detector = self._get_detector()
        img_out = img.copy()

        # 启发式检测器可以迭代微调，VLM 只做单次判断
        is_heuristic = detector.name() == "heuristic"

        try:
            if is_heuristic:
                # 启发式：迭代反馈
                img_out, final_cast, iters = self._heuristic_feedback(detector, img_out)
            else:
                # VLM/ONNX：单次修正（防模型发散）
                img_out, final_cast = self._vlm_single_shot(detector, img_out)
                iters = 1
        except RuntimeError as e:
            # VLM API 调用失败——返回错误信息，不静默回退
            raise RuntimeError(
                f"偏色检测失败: {e}\n"
                f"请检查 API Key、网络连接或模型名称是否正确"
            ) from e

        return CorrectionResult(
            image=np.clip(img_out, 0, 255).astype(np.uint8),
            mask_info=mask_info,
            iterations=iters,
            final_cast=final_cast,
            warm_style=self.config.warmth_style,
            detector_warning=self._detector_warning,
        )

    def _heuristic_feedback(self, detector: CastDetector,
                            img_out: np.ndarray):
        """启发式检测器的迭代反馈

        Returns:
            (修改后的图像, final_cast, 迭代次数)
        """
        final_cast = CastResult("ok", 0, 0, 1.0)
        accum_r, accum_g, accum_b = 1.0, 1.0, 1.0
        prev_cast = ""
        iters = 0

        for i in range(self.config.max_iterations):
            cast = detector.detect(img_out.astype(np.uint8))
            iters = i + 1

            if cast.is_ok or cast.severity < self.config.cast_threshold:
                print(f"[Engine] 偏色检测 OK (迭代 {i+1})")
                final_cast = cast
                break

            # 方向翻转检测
            direction_pairs = {
                "warm": ["cool", "blue"],
                "cool": ["warm", "yellow"],
                "yellow": ["blue", "cool"],
                "blue": ["yellow", "warm"],
                "magenta": ["green"],
                "green": ["magenta"],
            }
            if prev_cast in direction_pairs.get(cast.cast_type, []):
                print(f"[Engine] 方向翻转 ({prev_cast}→{cast.cast_type})，已收敛")
                break
            prev_cast = cast.cast_type

            damping = max(0.97 ** i, 0.3)
            img_out, adj_r, adj_g, adj_b = self._adjust_img(
                img_out, cast, damping
            )
            accum_r *= adj_r
            accum_g *= adj_g
            accum_b *= adj_b
            print(f"[Engine] 迭代 {i+1}: 检测到 {cast.cast_type} "
                  f"(severity={cast.severity:.3f}) "
                  f"累计调整 R={accum_r:.4f} G={accum_g:.4f} B={accum_b:.4f}")

        return img_out, final_cast, iters

    def _vlm_single_shot(self, detector: CastDetector,
                         img_out: np.ndarray):
        """VLM 单次修正——检测偏色，做一次调整，完毕

        Returns:
            (修改后的图像, final_cast)
        """
        cast = detector.detect(img_out.astype(np.uint8))

        if cast.is_ok or cast.severity < self.config.cast_threshold:
            print(f"[Engine] VLM 偏色检测 OK (severity={cast.severity:.3f})")
            if cast.detail:
                print(f"[Engine] VLM detail: {cast.detail}")
            return img_out, cast

        print(f"[Engine] VLM 检测到 {cast.cast_type} (severity={cast.severity:.3f})，"
              f"一次性修正 {cast.severity*0.15:.1%}")
        if cast.detail:
            print(f"[Engine] VLM detail: {cast.detail}")

        # 单次调整，幅度 = severity × 15%
        img_out, _, _, _ = self._adjust_img(img_out, cast, damping=1.0, scale=0.15)

        return img_out, CastResult(
            cast_type=cast.cast_type,
            severity=0,
            confidence=1,
            neutral_score=0.85,
            detail=f"VLM 单次修正: {cast.cast_type}({cast.severity:.2f})"
        )

    def _adjust_img(self, img: np.ndarray,
                    cast: CastResult,
                    damping: float = 1.0,
                    scale: float = 0.04) -> tuple:
        """根据偏色检测结果微调通道比例

        Args:
            img: 输入图像
            cast: 偏色检测结果
            damping: 阻尼系数
            scale: 基础调整幅度（默认 4%）

        Returns:
            (调整后图像, R调整系数, G调整系数, B调整系数)
        """
        result = img.copy().astype(np.float32)
        magnitude = 1.0 + cast.severity * scale * damping

        adjustments = {
            "blue":     (1.004, 1.004, 1/magnitude),
            "cyan":     (1.004, 1.000, 1/magnitude),
            "green":    (1.004, 1/magnitude, 1.004),
            "yellow":   (0.998, 1.000, magnitude),
            "magenta":  (1.000, magnitude, 1.000),
            "warm":     (0.998, 1.000, magnitude),
            "cool":     (1.004, 1.000, 1/magnitude),
        }

        adj_r, adj_g, adj_b = adjustments.get(cast.cast_type, (1, 1, 1))
        result[:,:,0] *= adj_r
        result[:,:,1] *= adj_g
        result[:,:,2] *= adj_b

        return result, adj_r, adj_g, adj_b
