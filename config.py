# ═══════════════════════════════════════════
#  底片自动校色 App — 配置文件
# ═══════════════════════════════════════════

# ─── 偏色检测器 ───
# 可选: "auto" | "heuristic" | "onnx" | "vlm_api"
#   auto      = 自动尝试 ONNX → VL API → 启发式
#   heuristic = 纯算法 (零依赖，保底)
#   onnx      = 本地 ONNX 模型 (需预先训练)
#   vlm_api   = VL 模型 API (需 API Key)
DETECTOR_MODE = "auto"

# VL 模型 API 配置（使用 heuristic 时无需填写）
VLM_API_BASE = ""
VLM_API_KEY = ""
VLM_MODEL = "openrouter/openai/gpt-4o-mini"
VLM_TIMEOUT = 30

# ONNX 模型路径
ONNX_MODEL_PATH = "models/cast_detector.onnx"

# ─── 校色默认参数 ───
FILM_TYPE = "negative"        # "negative" | "positive"
WARMTH_STYLE = "natural"      # "natural" | "kodak_gold" | "fuji_superia"
WARMTH_STRENGTH = 1.0         # 0.0 ~ 2.0
LEVELS_PERCENTILE = 0.2       # 色阶裁剪百分位
MAX_FEEDBACK_ITERATIONS = 10  # AI 反馈最大迭代次数
CAST_THRESHOLD = 0.15         # 偏色容忍度
