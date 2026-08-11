"""
色罩分析模块 — Mask Analyzer

从负片扫描图中检测 C-41 橙红色罩，计算各通道的补偿比例。

采样策略：
  A  → VLM 直接定位中性灰（VLM 已配置时的首选）
  B  → 边缘片基采样（VLM 失败/无有效结果时）
  C  → 全局灰世界假设（最终 fallback）
"""

from dataclasses import dataclass
import json
import numpy as np
from typing import Optional


@dataclass
class MaskResult:
    """色罩分析结果"""
    method: str
    ref_r: float
    ref_g: float
    ref_b: float
    scale_r: float
    scale_g: float
    scale_b: float
    confidence: float
    detail: str = ""


def analyze_mask(image: np.ndarray, vlm_config: Optional[dict] = None) -> MaskResult:
    """分析图像色罩，返回补偿比例。

    image 是反相后的 RGB 图像（支持 8-bit 和 16-bit 数值范围）。
    """
    _h, _w = image.shape[:2]

    # ── 策略 A: 算法预校色 → VLM 定位中性灰 ──
    if vlm_config and vlm_config.get("api_key"):
        vlm_result = _vlm_find_neutral_gray(image, vlm_config)
        if vlm_result is not None:
            return vlm_result
        print("[MaskAnalyzer] VLM 无有效中性灰，回退到边缘片基")

    # ── 策略 B: 片基边缘采样（四条边择最均匀者） ──
    edge_result = _find_edge_reference(image)
    if edge_result is not None:
        edge_side, edge_mean = edge_result
        return _compute_from_ref(
            image, edge_mean, method="edge", confidence=0.95,
            detail=(f"片基{edge_side}侧均值 RGB="
                    f"({edge_mean[0]:.0f},{edge_mean[1]:.0f},{edge_mean[2]:.0f})"),
        )

    # ── 策略 C: 全局灰世界假设 ──
    global_mean = np.mean(image, axis=(0, 1))
    return _compute_from_ref(
        image, global_mean, method="global", confidence=0.5,
        detail=f"全局均值 RGB=({global_mean[0]:.0f},{global_mean[1]:.0f},{global_mean[2]:.0f})",
    )


def _compute_from_ref(
    image: np.ndarray,
    ref_rgb: np.ndarray,
    method: str,
    confidence: float,
    detail: str = "",
) -> MaskResult:
    """根据参考点 RGB 计算补偿比例。

    在反相空间中把参考点三个通道归一化到同一亮度。
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


def _find_edge_reference(image: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
    """在上、下、左、右四条边中寻找最均匀的片基候选。"""
    h, w = image.shape[:2]
    edge_width = max(int(w * 0.05), 20)
    edge_height = max(int(h * 0.05), 20)
    candidates = [
        ("左", image[:, :edge_width, :]),
        ("右", image[:, max(0, w - edge_width):, :]),
        ("上", image[:edge_height, :, :]),
        ("下", image[max(0, h - edge_height):, :, :]),
    ]

    max_val = 65535.0 if float(np.max(image)) > 255.0 else 255.0
    scored = []
    for side, region in candidates:
        normalized = region.astype(np.float32) / max_val * 255.0
        std = float(np.std(normalized, axis=(0, 1)).mean())
        scored.append((std, side, np.mean(region, axis=(0, 1))))

    std, side, mean = min(scored, key=lambda item: item[0])
    if std >= 30:
        return None
    return side, mean


def _vlm_find_neutral_gray(image: np.ndarray, vlm_config: dict) -> Optional[MaskResult]:
    """用 VLM 识别图中可能的中性灰区域，采样其 RGB 作为参考点。

    VLM 失败或没有返回有效区域时返回 None。
    """
    from .cast_detector import _encode_image

    img_max = float(image.max()) if image.max() > 0 else 255.0
    img_for_vlm = _prepare_vlm_preview(image, vlm_config)

    prompt = """Analyze this already-inverted color-negative scan.

Your task is to identify 1-3 regions in the depicted original scene that
are most likely neutral in color and suitable for removing the film orange
mask. You are the PRIMARY reference selector; do not defer to an algorithm.

Prefer actual neutral materials or objects:
- gray concrete, gray walls, asphalt, gray pavement
- unpainted metal, neutral white or gray paper/card, neutral white signs
- other broad, evenly lit neutral-colored background surfaces

Do not use:
- blue sky, green vegetation, skin, colorful clothing
- red/orange brick or painted colored objects
- strong shadows, specular highlights, overexposed white areas
- narrow edges or areas contaminated by adjacent colors

The supplied image has already been inverted from a color negative and has
been preliminarily white-balanced and normalized by an algorithm for viewing.
Use this preview only to identify materials and locations. The program will
sample the final RGB reference from the original-resolution inverted image.
Return a normalized center
coordinate [x, y] for each region (0.0-1.0, left-to-right/top-to-bottom).
The program will sample at least a 3x3 neighborhood around each center from
the original-resolution image, so do not use a boundary, highlight, or tiny
detail as the center.

For each region, describe its material and approximate location so the
selection can be audited. Respond in JSON only:
{"regions": [{"description": "...", "location": "...",
"center": [x, y], "confidence": 0.0}]}

Return 1-3 credible regions. Return {"regions": []} only when no credible
neutral reference exists."""

    try:
        img_b64 = _encode_image(img_for_vlm)
        data = _request_vlm_json(
            api_base=vlm_config.get("api_base", ""),
            api_key=vlm_config.get("api_key", ""),
            model=vlm_config.get("model", "openai/gpt-4o-mini"),
            prompt=prompt,
            image_b64=img_b64,
            timeout=vlm_config.get("timeout", 30),
        )
        regions = data.get("regions", [])
        if not regions:
            print("[MaskAnalyzer] VLM 中性灰检测未返回有效区域")
            return None

        samples = []
        valid_regions = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            center = region.get("center")
            if not isinstance(center, list) or len(center) != 2:
                continue
            sample = _sample_neighborhood(image, center)
            if sample is not None:
                samples.append(sample)
                valid_regions.append(region)

        if not samples:
            print("[MaskAnalyzer] VLM 中性灰检测返回的区域缺少有效中心坐标")
            return None

        mean_rgb = np.mean(samples, axis=0)

        descs = [r.get("description", "") for r in valid_regions]
        detail = f"VLM中性灰: {', '.join(descs)} RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})"
        print(f"[MaskAnalyzer] {detail}")
        return _compute_from_ref(image, mean_rgb, method="vlm_gray", confidence=0.8, detail=detail)

    except Exception as e:
        print(f"[MaskAnalyzer] VLM 中性灰检测失败: {e}")
        return None


def _find_gray_pixels(img: np.ndarray) -> np.ndarray:
    """提取亮度适中、饱和度低的像素（可能是中性灰）。"""
    # 阈值定义在 0-255 空间；16-bit 输入必须先归一化，否则会被误判为过亮。
    max_val = 65535.0 if float(np.max(img)) > 255.0 else 255.0
    normalized = img.astype(np.float32) / max_val * 255.0
    gray = np.mean(normalized, axis=2)
    sat = (
        np.abs(normalized[:, :, 0] - gray)
        + np.abs(normalized[:, :, 1] - gray)
        + np.abs(normalized[:, :, 2] - gray)
    )
    brightness_mask = (gray > 30) & (gray < 220)
    sat_mask = (sat / 3) < 30
    return brightness_mask & sat_mask


def _request_vlm_json(**kwargs) -> dict:
    """调用 VLM 并解析 JSON；对瞬时失败重试一次。"""
    from .cast_detector import _call_vlm_api

    last_error = None
    for attempt in range(2):
        try:
            response = _call_vlm_api(**kwargs)
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("VLM JSON 顶层不是对象")
            return data
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                print(f"[MaskAnalyzer] VLM 响应无效，重试一次: {exc}")

    raise last_error


def _restore_vlm_rgb(rgb: np.ndarray, image_max: float) -> np.ndarray:
    """将 VLM 返回的 RGB 统一到输入图像的数值范围。"""
    values = np.asarray(rgb, dtype=float)
    if image_max > 255 and np.max(values) <= 255:
        return values / 255.0 * image_max
    return np.clip(values, 0, image_max)


def _prepare_vlm_preview(image: np.ndarray, vlm_config: dict) -> np.ndarray:
    """用非 VLM 算法预校色，生成供 VLM 识别位置的 0-255 预览图。"""
    max_val = 65535.0 if float(np.max(image)) > 255.0 else 255.0
    algorithm_mask = _find_edge_reference(image)
    if algorithm_mask is None:
        global_mean = np.mean(image, axis=(0, 1))
        algorithm_result = _compute_from_ref(
            image, global_mean, method="global", confidence=0.5,
        )
    else:
        _side, edge_mean = algorithm_mask
        algorithm_result = _compute_from_ref(
            image, edge_mean, method="edge", confidence=0.95,
        )

    from .channel_comp import apply_channel_compensation
    from .auto_levels import auto_levels

    preview = apply_channel_compensation(image, algorithm_result, max_val=max_val)
    preview = auto_levels(
        preview,
        percentile=vlm_config.get("levels_percentile", 0.2),
        max_val=max_val,
    )
    return np.clip(preview / max_val * 255.0, 0, 255).astype(np.uint8)


def _sample_neighborhood(
    image: np.ndarray,
    center: list,
    radius: int = 1,
) -> Optional[np.ndarray]:
    """从原始图像按中心坐标采样至少 3x3 邻域均值。"""
    if len(center) != 2:
        return None
    try:
        x_norm, y_norm = float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return None

    h, w = image.shape[:2]
    x = min(w - 1, max(0, int(round(x_norm * (w - 1)))))
    y = min(h - 1, max(0, int(round(y_norm * (h - 1)))))
    if x < radius or x >= w - radius or y < radius or y >= h - radius:
        return None
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return np.mean(image[y0:y1, x0:x1, :], axis=(0, 1))
