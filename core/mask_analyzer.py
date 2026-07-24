"""
色罩分析模块 — Mask Analyzer

从负片扫描图中检测 C-41 橙红色罩，计算各通道的补偿比例。

采样策略：
  A  → 边缘片基采样（最准确）
  B  → 画面内中性灰检测（算法，边缘被裁时用）
  B+ → VLM 二次校正（算法成功 + VLM 可用时，验证/优化参考点）
  B2 → VLM 直接定位中性灰（算法失败 + VLM 可用时）
  C  → 全局灰世界假设（最终 fallback）
"""

from dataclasses import dataclass
import json
import numpy as np
from typing import Tuple, Optional


@dataclass
class MaskResult:
    """色罩分析结果"""
    method: str                 # 使用的采样策略: "edge" | "gray" | "vlm_gray" | "global"
    ref_r: float                # 参考点 R 均值
    ref_g: float
    ref_b: float
    scale_r: float             # R 通道补偿比例
    scale_g: float             # G 通道补偿比例
    scale_b: float             # B 通道补偿比例
    confidence: float          # 置信度 0~1
    detail: str = ""           # 额外说明


def analyze_mask(image: np.ndarray, vlm_config: Optional[dict] = None) -> MaskResult:
    """分析图像色罩，返回补偿比例

    传入的 image 应是反相后的图像（float32, 0-255）。

    Args:
        image: 反相后的 RGB 图像 (H, W, 3), float32
        vlm_config: VLM API 配置（可选），用于 Strategy B2
            包含 api_base, api_key, model, timeout

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

    # ── 策略 B: 画面内中性灰检测（算法） ──
    # 在画面下半部分搜索亮度中等、饱和度最低的区域
    bottom_half = image[h//2:, :, :]
    gray_pixels = _find_gray_pixels(bottom_half)

    if np.sum(gray_pixels) > 200:
        gray_region = bottom_half[gray_pixels]
        gray_mean = np.mean(gray_region, axis=0)
        gray_std = np.std(gray_region, axis=0).mean()
        conf = max(0.6, 1.0 - gray_std / 60)

        # 有 VLM Key 时，强制二次校正参考点
        if vlm_config and vlm_config.get("api_key"):
            vlm_corrected = _vlm_correct_reference(
                image, vlm_config, algorithm_ref=gray_mean)
            if vlm_corrected is not None:
                return vlm_corrected
            # VLM 失败，保留算法结果

        return _compute_from_ref(image, gray_mean, method="gray",
                                 confidence=conf,
                                 detail=f"中性灰区域均值 RGB=({gray_mean[0]:.0f},{gray_mean[1]:.0f},{gray_mean[2]:.0f})")

    # ── 策略 B2: VLM 辅助中性灰定位 ──
    if vlm_config and vlm_config.get("api_key"):
        vlm_result = _vlm_find_neutral_gray(image, vlm_config)
        if vlm_result is not None:
            return vlm_result

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


def _vlm_correct_reference(image: np.ndarray, vlm_config: dict,
                           algorithm_ref: np.ndarray) -> Optional[MaskResult]:
    """VLM 二次校正算法找到的中性灰参考点

    把算法结果告诉 VLM，让 VLM 判断是否有更好的参考区域。

    Args:
        image: 反相后的 RGB 图像 (H, W, 3), float32 (任意范围)
        vlm_config: {"api_base", "api_key", "model", "timeout"}
        algorithm_ref: 算法找到的参考点 RGB (3,)

    Returns:
        MaskResult（VLM 确认或更优结果）或 None（VLM 失败时保留算法结果）
    """
    from .cast_detector import _encode_image, _call_vlm_api

    # 检测原始范围，归一化到 0-255 给 VLM
    img_max = float(image.max()) if image.max() > 0 else 255.0
    if img_max > 255:
        img_for_vlm = (image / img_max * 255).astype(np.float32)
    else:
        img_for_vlm = image.astype(np.float32)

    ref_rgb = algorithm_ref.tolist() if hasattr(algorithm_ref, 'tolist') else list(algorithm_ref)
    prompt = f"""Analyze this inverted film negative scan for neutral gray reference.

The algorithm found a reference point at RGB=[{ref_rgb[0]:.0f}, {ref_rgb[1]:.0f}, {ref_rgb[2]:.0f}].
Is this a GOOD neutral gray reference from the original scene?

If you see a BETTER neutral gray region the algorithm missed (e.g. a gray wall, concrete, asphalt road, metal fence), return it.
Otherwise confirm the algorithm's choice is good.

Respond in JSON only:
{{"confirmed": true}}
or
{{"confirmed": false, "regions": [{{"description": "...", "rgb": [R, G, B]}}]}}"""

    try:
        img_b64 = _encode_image(img_for_vlm)
        response = _call_vlm_api(
            api_base=vlm_config.get("api_base", ""),
            api_key=vlm_config.get("api_key", ""),
            model=vlm_config.get("model", "openai/gpt-4o-mini"),
            prompt=prompt,
            image_b64=img_b64,
            timeout=vlm_config.get("timeout", 30),
        )

        data = json.loads(response)
        print(f"[MaskAnalyzer] VLM 校正结果: {data}")

        if data.get("confirmed"):
            print(f"[MaskAnalyzer] VLM 确认算法参考点 RGB=({ref_rgb[0]:.0f},{ref_rgb[1]:.0f},{ref_rgb[2]:.0f})")
            return None  # 返回 None 让调用方保留算法结果

        regions = data.get("regions", [])
        rgbs = [r["rgb"] for r in regions
                if isinstance(r.get("rgb"), list) and len(r["rgb"]) == 3]
        if not rgbs:
            return None

        mean_rgb = np.mean(rgbs, axis=0).astype(float)
        # VLM 返回 0-255 范围的 RGB，还原到原始范围
        if img_max > 255:
            mean_rgb = mean_rgb / 255.0 * img_max

        descs = [r.get("description", "") for r in regions]
        detail = (f"VLM校正中性灰: {', '.join(descs)} "
                  f"RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})")
        print(f"[MaskAnalyzer] {detail}")
        return _compute_from_ref(image, mean_rgb, method="vlm_gray",
                                 confidence=0.8, detail=detail)

    except Exception as e:
        print(f"[MaskAnalyzer] VLM 二次校正失败: {e}")
        return None


def _vlm_find_neutral_gray(image: np.ndarray, vlm_config: dict) -> Optional[MaskResult]:
    """用 VLM 识别图中可能的中性灰区域，采样其 RGB 作为参考点

    Args:
        image: 反相后的 RGB 图像 (H, W, 3), float32 (任意范围)
        vlm_config: {"api_base", "api_key", "model", "timeout"}

    Returns:
        MaskResult 或 None（VLM 失败/无结果时）
    """
    from .cast_detector import _encode_image, _call_vlm_api

    # 检测原始范围，归一化到 0-255 给 VLM
    img_max = float(image.max()) if image.max() > 0 else 255.0
    if img_max > 255:
        img_for_vlm = (image / img_max * 255).astype(np.float32)
    else:
        img_for_vlm = image.astype(np.float32)

    prompt = """Analyze this inverted film negative scan for neutral gray reference points.
Find regions that are likely neutral gray in the ORIGINAL scene (before inversion):
- Gray walls, concrete, asphalt roads, metal fences, overcast sky reflections
- NOT colored objects (trees, blue sky, skin, clothes, red bricks)

Respond in JSON only:
{"regions": [{"description": "...", "rgb": [R, G, B]}]}

rgb values should be the MEAN color of that region in THIS inverted image (0-255 range).
Return 1-3 regions. If no neutral gray found, return {"regions": []}."""

    try:
        img_b64 = _encode_image(img_for_vlm)
        response = _call_vlm_api(
            api_base=vlm_config.get("api_base", ""),
            api_key=vlm_config.get("api_key", ""),
            model=vlm_config.get("model", "openai/gpt-4o-mini"),
            prompt=prompt,
            image_b64=img_b64,
            timeout=vlm_config.get("timeout", 30),
        )

        data = json.loads(response)
        regions = data.get("regions", [])

        if not regions:
            return None

        # 取所有 region 的 RGB 均值作为参考点
        rgbs = [r["rgb"] for r in regions if isinstance(r.get("rgb"), list) and len(r["rgb"]) == 3]
        if not rgbs:
            return None

        mean_rgb = np.mean(rgbs, axis=0).astype(float)
        # VLM 返回 0-255 范围的 RGB，还原到原始范围
        if img_max > 255:
            mean_rgb = mean_rgb / 255.0 * img_max

        descs = [r.get("description", "") for r in regions]
        detail = f"VLM中性灰: {', '.join(descs)} RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})"

        print(f"[MaskAnalyzer] {detail}")
        return _compute_from_ref(image, mean_rgb, method="vlm_gray",
                                 confidence=0.8, detail=detail)

    except Exception as e:
        print(f"[MaskAnalyzer] VLM 中性灰检测失败: {e}")
        return None


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
