"""
模型 Provider 管理 — Model Provider

管理 VL 模型的 Provider 选择、API 配置、模型列表。
支持多个后端：OpenRouter / OpenAI / Ollama / 自定义
"""

from dataclasses import dataclass, field
from typing import Optional
import requests


# ─── Provider 定义 ────────────────────────────────────────

@dataclass
class Provider:
    """一个 API Provider 的配置"""
    id: str                          # 唯一标识
    label: str                       # UI 显示名称
    base_url: str                    # API 基础地址
    api_key_env: str                 # 环境变量名（提示用）
    requires_key: bool = True        # 是否需要 API Key
    supports_list_models: bool = True  # 是否支持列出模型
    models: list[str] = field(default_factory=list)  # 可选模型列表
    default_model: str = ""          # 默认模型

    def api_url(self, endpoint: str = "") -> str:
        """生成完整 API URL"""
        base = self.base_url.rstrip("/")
        if endpoint:
            return f"{base}/{endpoint.lstrip('/')}"
        return base


# ─── 预置 Provider ────────────────────────────────────────

PROVIDERS = {
    "openrouter": Provider(
        id="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    ),
    "ollama": Provider(
        id="ollama",
        label="Ollama (本地)",
        base_url="http://localhost:11434",
        api_key_env="",
        requires_key=False,
        default_model="llava",
    ),
    "custom": Provider(
        id="custom",
        label="自定义 (OpenAI兼容)",
        base_url="",
        api_key_env="CUSTOM_API_KEY",
        default_model="",
    ),
}


# ─── 用户配置（持久化） ──────────────────────────────────

@dataclass
class APIConfig:
    """用户在 UI 上配置的 API 信息"""
    provider_id: str = "openrouter"   # 选择的 Provider
    api_key: str = ""                 # API Key
    base_url: str = ""                # 自定义 URL（custom provider 用）
    model: str = "openai/gpt-4o-mini"  # 具体模型名
    timeout: int = 30                 # 超时秒数

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig":
        return cls(
            provider_id=data.get("provider_id", "openrouter"),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", ""),
            model=data.get("model", "openai/gpt-4o-mini"),
            timeout=data.get("timeout", 30),
        )


# ─── 模型列表管理 ────────────────────────────────────────

def fetch_available_models(api_config: APIConfig) -> list[str]:
    """从 Provider API 获取可用模型列表

    用于 UI 的「选择模型」下拉框。
    如果 API 不可用或 Provider 不支持列出模型，返回预置列表。
    """
    provider = PROVIDERS.get(api_config.provider_id)
    if not provider:
        return []

    if not provider.supports_list_models:
        return provider.models

    try:
        headers = {}
        if provider.requires_key and api_config.api_key:
            headers["Authorization"] = f"Bearer {api_config.api_key}"

        base_url = api_config.base_url or provider.base_url
        url = f"{base_url.rstrip('/')}/models"

        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if api_config.provider_id == "openrouter":
                return [m["id"] for m in data.get("data", [])]
            elif api_config.provider_id == "openai":
                return [m["id"] for m in data.get("data", [])]
            elif api_config.provider_id == "ollama":
                return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"[ModelProvider] 获取模型列表失败: {e}")

    # 失败时返回预置的常见模型
    return get_default_models(api_config.provider_id)


def get_default_models(provider_id: str) -> list[str]:
    """返回 Provider 的常见模型（预置列表，供 fallback 和 UI 提示用）"""
    defaults = {
        "openrouter": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.5-flash-001",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "qwen/qwen2.5-vl-72b-instruct",
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
        ],
        "ollama": [
            "llava",
            "llava:13b",
            "bakllava",
            "moondream",
        ],
        "custom": [],
    }
    return defaults.get(provider_id, [])


def get_suitable_cast_models(api_config: APIConfig) -> list[str]:
    """返回适合做偏色检测的模型（过滤后的子集）

    偏色检测不需要最强模型，gpt-4o-mini / llava 级别就够了。
    """
    all_models = fetch_available_models(api_config)
    # 过滤视觉模型
    keywords = ["vision", "vl", "vlm", "mini", "flash",
                "llava", "moondream", "4o"]
    suitable = [m for m in all_models
                if any(kw in m.lower() for kw in keywords)]
    return suitable if suitable else all_models


# ─── 来自 API Config 的检测器配置 ─────────────────────────

def build_vlm_config(api_config: APIConfig) -> dict:
    """将用户的 API 配置转换为 DetectorFactory 可用的 config dict"""
    provider = PROVIDERS.get(api_config.provider_id)
    base_url = api_config.base_url or (provider.base_url if provider else "")
    # 标记模型为完整路径（OpenRouter 需要保留 provider/model 格式）
    return {
        "detector_mode": "vlm_api",
        "vlm_api_base": base_url,
        "vlm_api_key": api_config.api_key,
        "vlm_model": api_config.model,
        "vlm_timeout": api_config.timeout,
    }
