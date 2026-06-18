"""
底片自动校色 App — Web UI (Gradio)

简洁、美观的 Web 界面：
  1. 上传图片
  2. 选择正片/负片
  3. 调节暖调等参数
  4. 点校色 → 并排显示前后对比
"""

import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr

from core import Engine, CorrectionConfig
from core.warmth import get_warmth_presets
from core.cast_detector import get_available_backends
from core.config_manager import ConfigManager


def _correct(
    input_img,
    film_type,
    warmth_style,
    warmth_strength,
    detector_mode,
):
    """校色处理函数（Gradio 回调）"""
    if input_img is None:
        return None, "请上传图片"

    img = np.array(input_img, dtype=np.float32)

    # 加载配置文件中的 API Key
    cfg_mgr = ConfigManager()
    cfg_mgr.init_default_config()
    cfg = cfg_mgr.load()
    api_key = cfg_mgr.get_api_key("openrouter") or cfg_mgr.get_api_key("custom")
    api_base = cfg.detector.api_base
    model = cfg.detector.model

    config = CorrectionConfig(
        film_type=film_type,
        warmth_style=warmth_style,
        warmth_strength=warmth_strength,
        detector_mode=detector_mode,
        vlm_api_key=api_key,
        vlm_api_base=api_base,
        vlm_model=model,
    )

    try:
        engine = Engine(config)
        result = engine.correct(img)

        summary = (
            f"✅ 校色完成\n"
            f"  • 胶片: {'负片' if film_type == 'negative' else '正片'}\n"
            f"  • 色罩: {result.mask_info.method}\n"
            f"  • 暖调: {result.warm_style} (强度 {warmth_strength})\n"
            f"  • 迭代: {result.iterations} 次\n"
            f"  • 偏色: {result.final_cast.cast_type} "
            f"(程度 {result.final_cast.severity:.2f})\n"
            f"  • {result.final_cast.detail}"
        )
        return result.image, summary

    except Exception as e:
        return None, f"❌ 校色失败: {e}"


def launch(server_name: str = "127.0.0.1", server_port: int = 7860):
    """启动 Gradio Web UI"""

    warmth_presets = get_warmth_presets()
    warmth_choices = [(p["label"], p["id"]) for p in warmth_presets]

    backends = get_available_backends()
    backend_choices = [(b["label"], b["mode"]) for b in backends]

    with gr.Blocks(
        title="底片自动校色",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        ),
        css="""
        .corrected-image img { border-radius: 8px; }
        .summary-box { min-height: 120px; }
        footer { display: none !important; }
        """
    ) as demo:
        gr.Markdown(
            "# 🎞️ 底片自动校色\n"
            "上传胶片扫描图，一键校色。支持负片 C-41 色罩去除 + AI 偏色检测。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="上传图片",
                    type="pil",
                    height=400,
                )

                with gr.Accordion("校色参数", open=False):
                    film_type = gr.Radio(
                        label="胶片类型",
                        choices=[("负片 (Negative)", "negative"),
                                 ("正片 (Positive)", "positive")],
                        value="negative",
                    )
                    warmth_style = gr.Dropdown(
                        label="暖调风格",
                        choices=warmth_choices,
                        value="natural",
                    )
                    warmth_strength = gr.Slider(
                        label="暖调强度",
                        minimum=0.0, maximum=2.0,
                        value=1.0, step=0.1,
                    )
                    detector_mode = gr.Dropdown(
                        label="偏色检测后端",
                        choices=backend_choices,
                        value="auto",
                    )

                correct_btn = gr.Button(
                    "🎨 开始校色",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="校色结果",
                    type="numpy",
                    height=400,
                    elem_classes="corrected-image",
                )
                summary = gr.Textbox(
                    label="校色报告",
                    lines=6,
                    elem_classes="summary-box",
                )

        correct_btn.click(
            fn=_correct,
            inputs=[input_img, film_type, warmth_style,
                    warmth_strength, detector_mode],
            outputs=[output_img, summary],
        )

        gr.Markdown(
            "---\n"
            "🐙 [Yachiyo1680/negative-color-corrector]"
            "(https://github.com/Yachiyo1680/negative-color-corrector)"
        )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
    )


if __name__ == "__main__":
    launch()
