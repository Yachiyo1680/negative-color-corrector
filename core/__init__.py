"""
core — 校色引擎核心模块
"""

from .engine import Engine, CorrectionConfig, CorrectionResult
from .invert import invert, invert_float
from .mask_analyzer import MaskResult, analyze_mask
from .channel_comp import apply_channel_compensation, manual_compensation
from .auto_levels import auto_levels, auto_levels_combined
from .warmth import apply_warmth, WarmthStyle, get_warmth_presets
from .cast_detector import (
    CastDetector, CastResult, CastType,
    HeuristicCastDetector, VLModelCastDetector,
    DetectorFactory, get_available_backends,
)
from .model_provider import (
    Provider, APIConfig, PROVIDERS,
    fetch_available_models, get_default_models,
    get_suitable_cast_models, build_vlm_config,
)
from .credential_store import CredentialStore
from .config_manager import ConfigManager, AppConfig

__all__ = [
    "Engine", "CorrectionConfig", "CorrectionResult",
    "invert", "invert_float",
    "MaskResult", "analyze_mask",
    "apply_channel_compensation", "manual_compensation",
    "auto_levels", "auto_levels_combined",
    "apply_warmth", "WarmthStyle", "get_warmth_presets",
    "CastDetector", "CastResult", "CastType",
    "HeuristicCastDetector", "VLModelCastDetector",
    "DetectorFactory", "get_available_backends",
    "Provider", "APIConfig", "PROVIDERS",
    "fetch_available_models", "get_default_models",
    "get_suitable_cast_models", "build_vlm_config",
    "CredentialStore",
    "ConfigManager", "AppConfig",
]
