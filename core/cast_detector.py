"""
AI 偏色检测器 — Color Cast Detector

负责：检测校色结果是否偏色，并返回偏色方向 + 程度
设计：抽象接口层 + 多后端支持（本地 / API / 启发式）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal
import io
import json
import base64
import requests
import numpy as np
from PIL import Image


# ─── 数据类型 ──────────────────────────────────────────────

CastType = Literal["blue", "green", "magenta", "yellow", "cyan",
                   "warm", "cool", "ok"]

@dataclass
class CastResult:
    """偏色检测结果"""
    cast_type: CastType        # 偏色方向
    severity: float            # 偏色程度 0.0 ~ 1.0（0=无，1=严重）
    confidence: float          # 置信度 0.0 ~ 1.0
    neutral_score: float       # 中性灰自然度 0.0 ~ 1.0
    detail: str = ""           # 额外说明

    @property
    def is_ok(self) -> bool:
        return self.cast_type == "ok" or self.severity < 0.15


# ─── 抽象基类 ──────────────────────────────────────────────

class CastDetector(ABC):
    """偏色检测器基类 — 所有后端继承此接口"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> CastResult:
        """输入 RGB 图像 (H,W,3)，返回偏色检测结果"""
        ...

    @abstractmethod
    def name(self) -> str:
        """返回检测器名称，用于 UI 显示"""
        ...

    @property
    def display_name(self) -> str:
        """用户友好的显示名称"""
        return self.name()


# ══════════════════════════════════════════════════════════
#  后端 1：启发式规则（零依赖，保底方案）
# ══════════════════════════════════════════════════════════

class HeuristicCastDetector(CastDetector):
    """
    基于 RGB 通道均值和中性灰区域分析的启发式检测。
    不需要任何 AI 模型，纯算法判断。
    """

    def name(self) -> str:
        return "heuristic"

    @property
    def display_name(self) -> str:
        return "启发式算法"

    def detect(self, image: np.ndarray) -> CastResult:
        h, w = image.shape[:2]

        # 1. 全图通道均值
        mean_r = np.mean(image[:,:,0])
        mean_g = np.mean(image[:,:,1])
        mean_b = np.mean(image[:,:,2])

        # 2. 采样画面下半部分的"可能中性灰"区域
        bottom = image[int(h*0.5):, :, :]
        gray_mask = self._find_gray_pixels(bottom)
        if np.sum(gray_mask) > 100:
            gray_r = np.mean(bottom[:,:,0][gray_mask])
            gray_g = np.mean(bottom[:,:,1][gray_mask])
            gray_b = np.mean(bottom[:,:,2][gray_mask])
        else:
            gray_r, gray_g, gray_b = mean_r, mean_g, mean_b

        # 3. 计算偏色方向
        neutral = (gray_r + gray_g + gray_b) / 3
        delta_r = (gray_r - neutral) / 255.0
        delta_g = (gray_g - neutral) / 255.0
        delta_b = (gray_b - neutral) / 255.0

        severity = max(abs(delta_r), abs(delta_g), abs(delta_b))
        cast_type = self._classify_cast(delta_r, delta_g, delta_b)
        neutral_score = max(0, 1.0 - severity * 3)

        return CastResult(
            cast_type=cast_type,
            severity=min(severity, 1.0),
            confidence=0.6,
            neutral_score=neutral_score,
            detail=f"中性灰参考 RGB: ({gray_r:.0f}, {gray_g:.0f}, {gray_b:.0f})"
        )

    def _find_gray_pixels(self, img: np.ndarray) -> np.ndarray:
        """提取亮度适中、饱和度低的像素（可能是中性灰）"""
        gray = np.mean(img, axis=2)
        sat = np.abs(img[:,:,0].astype(float) - gray) \
            + np.abs(img[:,:,1].astype(float) - gray) \
            + np.abs(img[:,:,2].astype(float) - gray)
        saturation = sat / 3
        brightness_mask = (gray > 30) & (gray < 220)
        sat_mask = saturation < 30
        return brightness_mask & sat_mask

    def _classify_cast(self, dr: float, dg: float, db: float) -> CastType:
        """将 RGB 偏差映射到偏色方向"""
        casts = {
            "blue":    (dr < 0,  dg < 0,  db > 0),
            "cyan":    (dr < 0,  dg > 0,  db > 0),
            "green":   (dr < 0,  dg > 0,  db < 0),
            "yellow":  (dr > 0,  dg > 0,  db < 0),
            "magenta": (dr > 0,  dg < 0,  db > 0),
            "warm":    (dr > 0,  dg > 0,  db <= 0),
            "cool":    (dr < 0,  dg < 0,  db >= 0),
        }
        best = "ok"
        best_score = 0
        for name, (r_c, g_c, b_c) in casts.items():
            score = 0
            if r_c == (dr > 0): score += abs(dr)
            if g_c == (dg > 0): score += abs(dg)
            if b_c == (db > 0): score += abs(db)
            if score > best_score:
                best_score = score
                best = name
        if best_score < 0.02:
            return "ok"
        return best  # type: ignore


# ══════════════════════════════════════════════════════════
#  后端 2：VL 模型 API（通过兼容 API 调用）
# ══════════════════════════════════════════════════════════

class VLModelCastDetector(CastDetector):
    """
    使用视觉语言模型的 API 进行偏色检测。
    兼容 OpenRouter / OpenAI / Ollama 等。
    """

    def __init__(self, api_base: str = "", api_key: str = "",
                 model: str = "openrouter/openai/gpt-4o-mini",
                 timeout: int = 30):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def name(self) -> str:
        return f"vlm_api:{self.model}"

    @property
    def display_name(self) -> str:
        return f"VL模型 ({self.model})"

    def detect(self, image: np.ndarray) -> CastResult:
        """调用 VL 模型 API 分析偏色"""
        try:
            img_b64 = _encode_image(image)
            response_text = _call_vlm_api(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
                prompt=self._build_prompt(),
                image_b64=img_b64,
                timeout=self.timeout,
            )
            return self._parse_response(response_text)
        except Exception as e:
            return CastResult(
                "ok", 0, 0, 0.5,
                detail=f"VL API 调用失败: {e}"
            )

    def _build_prompt(self) -> str:
        return """Analyze this color-negative-corrected photo for color cast.
Respond in this exact JSON format:
{"cast": "blue|green|magenta|yellow|cyan|warm|cool|ok",
"severity": 0.0-1.0,
"neutral_score": 0.0-1.0,
"detail": "short observation"}

Check: Are whites/greys neutral? Sky natural? Brick/warm tones red-brown?"""

    def _parse_response(self, response: str) -> CastResult:
        try:
            data = json.loads(response)
            return CastResult(
                cast_type=data.get("cast", "ok"),
                severity=float(data.get("severity", 0)),
                confidence=0.85,
                neutral_score=float(data.get("neutral_score", 0.7)),
                detail=data.get("detail", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return CastResult("ok", 0, 0, 0.5,
                              detail="模型响应解析失败")


# ══════════════════════════════════════════════════════════
#  后端 3：本地 ONNX 模型（轻量，设备端运行）
# ══════════════════════════════════════════════════════════

class ONNXCastDetector(CastDetector):
    """
    使用本地 ONNX 模型推理。
    优点：离线可用、低延迟、无 API 费用。
    """

    def __init__(self, model_path: str = "models/cast_detector.onnx"):
        self.model_path = model_path
        self.session = None

    def name(self) -> str:
        return "onnx_local"

    @property
    def display_name(self) -> str:
        return "本地ONNX模型"

    def _load_model(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
        except ImportError:
            raise RuntimeError("需要安装 onnxruntime: pip install onnxruntime")
        except FileNotFoundError:
            raise RuntimeError(f"模型文件不存在: {self.model_path}")

    def detect(self, image: np.ndarray) -> CastResult:
        if self.session is None:
            self._load_model()

        # 预处理
        img_pil = Image.fromarray(image)
        img_resized = img_pil.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        # 推理
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: img_array})[0]

        # TODO: 解析模型输出
        raise NotImplementedError("需要先训练偏色检测模型")


# ══════════════════════════════════════════════════════════
#  可用后端列表（供 UI 使用）
# ══════════════════════════════════════════════════════════

DETECTOR_BACKENDS = {
    "auto":       "自动选择 (推荐)",
    "heuristic":  "启发式算法 (离线, 无需API)",
    "onnx":       "本地模型 (ONNX, 需预先训练)",
    "vlm_api":    "VL模型 (需API Key, 最准确)",
}


def get_available_backends() -> list[dict]:
    """返回当前环境可用的后端列表（供UI显示）

    每个元素: {"mode": str, "label": str, "available": bool, "doc": str}
    """
    results = []
    for mode, label in DETECTOR_BACKENDS.items():
        available = True
        if mode == "onnx":
            try:
                import onnxruntime  # noqa
                import os
                available = os.path.exists("models/cast_detector.onnx")
            except ImportError:
                available = False
        elif mode == "vlm_api":
            available = True  # 始终显示，让用户配 Key
        results.append({
            "mode": mode,
            "label": label,
            "available": available,
            "doc": get_backend_doc(mode),
        })
    return results


def get_backend_label(mode: str) -> str:
    return DETECTOR_BACKENDS.get(mode, mode)


def get_backend_doc(mode: str) -> str:
    docs = {
        "auto": "自动尝试 ONNX → VL模型 → 启发式，选第一个可用的",
        "heuristic": "纯算法计算，零依赖，速度最快",
        "onnx": "加载本地 ONNX 模型，离线可用、低延迟",
        "vlm_api": "调用视觉语言模型 API 分析，准确度最高",
    }
    return docs.get(mode, "")


# ══════════════════════════════════════════════════════════
#  检测器工厂
# ══════════════════════════════════════════════════════════

class DetectorFactory:
    """根据配置创建偏色检测器

    UI 只需传 detector_mode 名称，工厂负责实例化。
    支持用户手动选择后端，也支持自动 fallback。
    """

    @staticmethod
    def create(config: dict) -> CastDetector:
        """创建偏色检测器

        Args:
            config: 包含以下可选字段
                - detector_mode: 后端模式名, 默认 "auto"
                - onnx_model_path: ONNX 模型路径
                - vlm_api_base/vlm_api_key/vlm_model/vlm_timeout

        Returns:
            CastDetector 实例

        Raises:
            RuntimeError: 用户指定的后端不可用
        """
        mode = config.get("detector_mode", "auto")

        creators = {
            "heuristic": lambda: HeuristicCastDetector(),
            "onnx": lambda: ONNXCastDetector(
                model_path=config.get("onnx_model_path",
                                      "models/cast_detector.onnx")
            ),
            "vlm_api": lambda: VLModelCastDetector(
                api_base=config.get("vlm_api_base", ""),
                api_key=config.get("vlm_api_key", ""),
                model=config.get("vlm_model",
                                 "openrouter/openai/gpt-4o-mini"),
                timeout=config.get("vlm_timeout", 30),
            ),
        }

        # ── 用户明确指定的单一后端 ──
        if mode != "auto" and mode in creators:
            try:
                return creators[mode]()
            except (RuntimeError, ImportError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"后端 '{mode}' 当前不可用: {e}\n"
                    f"请切换其他后端或使用自动选择模式"
                )

        # ── Auto 模式：逐个尝试 ──
        if mode == "auto":
            # 1st: ONNX
            try:
                return creators["onnx"]()
            except Exception:
                pass
            # 2nd: VL API
            try:
                return creators["vlm_api"]()
            except Exception:
                pass
            # 3rd: heuristic（一定有）
            print("[DetectorFactory] ONNX / VL API 均不可用，"
                  "回退到启发式算法")
            return HeuristicCastDetector()

        raise ValueError(
            f"未知后端模式: '{mode}'，"
            f"可用: {', '.join(creators.keys())}"
        )


# ══════════════════════════════════════════════════════════
#  Helper: 图片编码 & API 调用
# ══════════════════════════════════════════════════════════


def _encode_image(image: np.ndarray) -> str:
    """将 numpy 图像编码为 base64 JPEG"""
    img_pil = Image.fromarray(
        image.astype(np.uint8) if image.dtype != np.uint8
        else image
    )
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_vlm_api(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: int = 30,
) -> str:
    """调用 OpenAI 兼容的 VL 模型 API

    支持 OpenRouter / OpenAI / Ollama / 自定义等
    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 检测是否是 OpenRouter（需要额外 header）
    is_openrouter = "openrouter" in model or "openrouter" in api_base
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/Yachiyo1680/negative-color-corrector"
        headers["X-Title"] = "Negative Color Corrector"

    # 构建请求体
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    # 确定 API URL
    base = api_base.rstrip("/") if api_base else "https://openrouter.ai/api/v1"
    url = f"{base}/chat/completions"

    # 发送请求
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # 提取响应文本
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("API 响应为空")

    # 尝试提取 JSON（模型可能返回 markdown 包裹的 JSON）
    content = content.strip()
    if content.startswith("```"):
        # 去掉 markdown 代码块
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines
            if not line.startswith("```")
        )

    return content.strip()
