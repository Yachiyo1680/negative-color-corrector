"""
配置文件管理 — Config Manager

模仿 OpenClaw 的设计：
- JSON5 格式（支持注释和尾逗号）
- `${ENV_VAR}` 引用环境变量
- 自动加载 + 热重载
- 最小化配置，空文件也能正常工作
"""

import os
import re
import json
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ─── 默认配置文件路径 ─────────────────────────────────────

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.negative-corrector")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.json5")


# ─── 配置数据结构 ─────────────────────────────────────────

@dataclass
class DetectorConfig:
    """偏色检测器配置"""
    mode: str = "auto"           # auto | heuristic | onnx | vlm_api
    model: str = "openai/gpt-4o-mini"
    api_base: str = ""           # 自定义 API 地址
    timeout: int = 30
    max_iterations: int = 10
    cast_threshold: float = 0.15


@dataclass
class CorrectionConfig:
    """校色参数配置"""
    film_type: str = "negative"   # negative | positive
    warmth_style: str = "natural" # natural | kodak_gold | fuji_superia
    warmth_strength: float = 1.0  # 0.0 ~ 2.0
    levels_percentile: float = 0.2


@dataclass
class ProvidersConfig:
    """Provider API 配置（敏感信息用环境变量引用）"""
    openrouter_api_key: str = "${OPENROUTER_API_KEY}"
    openai_api_key: str = "${OPENAI_API_KEY}"
    custom_api_key: str = "${NCC_API_KEY}"
    custom_api_base: str = ""


@dataclass
class AppConfig:
    """完整应用配置"""
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)


# ─── 环境变量解析 ─────────────────────────────────────────

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def resolve_env_vars(value: str) -> str:
    """解析字符串中的 ${ENV_VAR} 引用

    例如:
        "${OPENAI_API_KEY}"         → 读取环境变量
        "sk-${SUFFIX}"              → 部分替换
        "直接写死的key"             → 原样返回

    模仿 OpenClaw 的行为。
    """
    if not isinstance(value, str):
        return value

    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return _ENV_VAR_PATTERN.sub(_replace, value)


# ─── JSON5 解析（简易版，不引入额外依赖）───────────────────

def _strip_json5_comments(text: str) -> str:
    """移除 JSON5 的注释（// 和 /* */）"""
    # 移除多行注释
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # 移除单行注释（注意不在字符串里）
    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                result.append(ch)
                i += 1
                if i < len(text):
                    result.append(text[i])
                i += 1
                continue
            if ch == string_char:
                in_string = False
            result.append(ch)
            i += 1
            continue
        if ch in '"\'':
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < len(text):
            if text[i + 1] == '/':
                # 跳到行尾
                while i < len(text) and text[i] != '\n':
                    i += 1
                result.append('\n')
                i += 1
                continue
        result.append(ch)
        i += 1
    return ''.join(result)


def _trailing_comma_clean(text: str) -> str:
    """移除对象/数组最后一个元素后的逗号"""
    # 简单的正则替换（99% 情况够用）
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text


def _convert_keys_to_lowercase(obj):
    """将 JSON 对象的 key 转为蛇形命名（保持兼容性）"""
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _convert_keys_to_lowercase(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys_to_lowercase(v) for v in obj]
    return obj


def _camel_to_snake(name: str) -> str:
    """驼峰转蛇形: openrouterApiKey → openrouter_api_key"""
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def parse_json5(file_path: str) -> dict:
    """解析 JSON5 配置文件为 Python dict

    步骤:
        1. 移除注释
        2. 清理尾逗号
        3. 转换为标准 JSON
        4. json.loads
    """
    text = Path(file_path).read_text(encoding="utf-8")
    text = _strip_json5_comments(text)
    text = _trailing_comma_clean(text)
    # JSON5 允许 key 不加引号、单引号
    # 保险起见用 key 加引号的正则替换
    text = re.sub(r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)(?=\s*:)', r'"\1"', text)
    text = text.replace("'", '"')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[Config] JSON5 解析失败: {e}")
        print(f"[Config] 文件内容预览:\n{text[:500]}")
        return {}
    return _convert_keys_to_lowercase(data)


# ─── 默认配置文件生成 ─────────────────────────────────────

DEFAULT_CONFIG_CONTENT = """// ~/.negative-corrector/config.json5
//
// 底片自动校色 App 配置文件
// 支持 JSON5 语法：注释、尾逗号、单引号
// API Key 建议用 ${ENV_VAR} 引用环境变量，如：
//   openrouter_api_key: "${OPENROUTER_API_KEY}"
// 也可以直接写明文（不推荐）：
//   openrouter_api_key: "sk-or-v1-..."

{
  // ─── Provider API 配置 ───
  providers: {
    // OpenRouter API Key
    openrouter_api_key: "${OPENROUTER_API_KEY}",

    // OpenAI API Key
    openai_api_key: "${OPENAI_API_KEY}",

    // 自定义 API Key（OpenAI 兼容接口）
    custom_api_key: "${NCC_API_KEY}",
    custom_api_base: "",
  },

  // ─── 偏色检测器 ───
  detector: {
    // 可选: "auto" | "heuristic" | "onnx" | "vlm_api"
    mode: "auto",

    // VL 模型名（detector.mode = "vlm_api" 时使用）
    model: "openai/gpt-4o-mini",

    // 自定义 API 地址（留空使用provider默认地址）
    api_base: "",

    // API 超时（秒）
    timeout: 30,

    // AI 反馈最大迭代次数
    max_iterations: 10,

    // 偏色容忍度（低于此值视为 OK）
    cast_threshold: 0.15,
  },

  // ─── 校色参数 ───
  correction: {
    // 底片类型: "negative" | "positive"
    film_type: "negative",

    // 暖调风格: "natural" | "kodak_gold" | "fuji_superia"
    warmth_style: "natural",

    // 暖调强度: 0.0 ~ 2.0
    warmth_strength: 1.0,

    // 色阶裁剪百分位: 0.1 ~ 0.5
    levels_percentile: 0.2,
  },
}
"""


# ─── 配置管理器 ───────────────────────────────────────────

class ConfigManager:
    """配置管理器

    模仿 OpenClaw 的设计：
        - JSON5 格式配置文件
        - ${ENV_VAR} 引用环境变量
        - 自动加载 + 热重载
        - 极简上手（空文件也能用）

    用法:
        config = ConfigManager()
        api_key = config.get_api_key("openrouter")
        detector_mode = config.get("detector.mode")
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "NCC_CONFIG_PATH", DEFAULT_CONFIG_PATH
        )
        self._raw_data: dict = {}
        self._app_config: Optional[AppConfig] = None
        self._mtime: float = 0

    # ── 加载 ──

    def load(self) -> AppConfig:
        """加载配置文件（如果文件不存在则使用默认值）"""
        path = Path(self.config_path)

        if path.exists():
            try:
                self._raw_data = parse_json5(self.config_path)
                self._mtime = path.stat().st_mtime
            except Exception as e:
                print(f"[Config] 加载配置文件失败: {e}")
                print(f"[Config] 使用默认配置")
                self._raw_data = {}
        else:
            print(f"[Config] 配置文件不存在: {self.config_path}")
            print(f"[Config] 使用默认配置")
            print(f"[Config] 可创建配置文件: {self.config_path}")
            self._raw_data = {}

        self._app_config = self._build_config(self._raw_data)
        return self._app_config

    def init_default_config(self) -> str:
        """创建默认配置文件（如果还不存在）"""
        path = Path(self.config_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_CONFIG_CONTENT, encoding="utf-8")
            print(f"[Config] 已创建默认配置文件: {self.config_path}")
        return self.config_path

    # ── 热重载 ──

    def check_reload(self) -> bool:
        """检查配置文件是否已变更，是则重载

        模仿 OpenClaw 的 hot reload 机制。
        """
        path = Path(self.config_path)
        if not path.exists():
            return False
        try:
            new_mtime = path.stat().st_mtime
            if new_mtime > self._mtime:
                self.load()
                print(f"[Config] 配置文件已热重载")
                return True
        except Exception:
            pass
        return False

    # ── 配置读取 ──

    def get(self, key_path: str, default: Any = None) -> Any:
        """按路径读取配置（点号分隔）

        例如:
            config.get("detector.mode")         → "auto"
            config.get("providers.openrouter_api_key") → "..."
            config.get("correction.film_type")  → "negative"
        """
        if self._app_config is None:
            self.load()

        parts = key_path.split(".")
        current = asdict(self._app_config) if self._app_config else {}

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, {})
            else:
                return default

        return current if current != {} else default

    def get_api_key(self, provider: str) -> str:
        """获取指定 Provider 的 API Key

        读取顺序:
        1. config.json5 中的 providers.xxx_api_key
        2. 自动解析 ${ENV_VAR} 引用
        3. 如果解析后为空，尝试直接读环境变量
        """
        key_map = {
            "openrouter": "openrouter_api_key",
            "openai": "openai_api_key",
            "custom": "custom_api_key",
        }

        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "custom": "NCC_API_KEY",
        }

        if self._app_config is None:
            self.load()

        # 1. 从配置文件读取
        key_name = key_map.get(provider)
        if key_name:
            raw = getattr(self._app_config.providers, key_name, "")
            resolved = resolve_env_vars(raw)
            if resolved:
                return resolved

        # 2. fallback：直接读环境变量
        env_name = env_map.get(provider)
        if env_name:
            env_val = os.environ.get(env_name, "")
            if env_val:
                return env_val

        return ""

    def set(self, key_path: str, value: Any) -> bool:
        """设置配置值并保存到文件"""
        # 这里简化处理，直接修改文件会比较复杂
        # 实际 UI 上修改配置时，应该用专门的配置编辑器
        print(f"[Config] 设置 {key_path} = {value} （需要实现）")
        return False

    # ── 内部构建 ──

    def _build_config(self, data: dict) -> AppConfig:
        """从 dict 构建 AppConfig"""

        providers_data = data.get("providers", {})
        detector_data = data.get("detector", {})
        correction_data = data.get("correction", {})

        return AppConfig(
            providers=ProvidersConfig(
                openrouter_api_key=str(
                    providers_data.get("openrouter_api_key",
                                       "${OPENROUTER_API_KEY}")
                ),
                openai_api_key=str(
                    providers_data.get("openai_api_key",
                                       "${OPENAI_API_KEY}")
                ),
                custom_api_key=str(
                    providers_data.get("custom_api_key", "${NCC_API_KEY}")
                ),
                custom_api_base=str(
                    providers_data.get("custom_api_base", "")
                ),
            ),
            detector=DetectorConfig(
                mode=str(detector_data.get("mode", "auto")),
                model=str(detector_data.get("model", "openai/gpt-4o-mini")),
                api_base=str(detector_data.get("api_base", "")),
                timeout=int(detector_data.get("timeout", 30)),
                max_iterations=int(detector_data.get("max_iterations", 10)),
                cast_threshold=float(detector_data.get("cast_threshold", 0.15)),
            ),
            correction=CorrectionConfig(
                film_type=str(correction_data.get("film_type", "negative")),
                warmth_style=str(
                    correction_data.get("warmth_style", "natural")
                ),
                warmth_strength=float(
                    correction_data.get("warmth_strength", 1.0)
                ),
                levels_percentile=float(
                    correction_data.get("levels_percentile", 0.2)
                ),
            ),
        )
