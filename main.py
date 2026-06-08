#!/usr/bin/env python3
"""
底片自动校色 App — 启动入口

用法：
    python main.py <image> [options]

    python main.py scan.tiff
    python main.py scan.tiff --film-type positive --warmth kodak_gold
    python main.py scan.tiff --output result.jpg
    python main.py scan.tiff --detector heuristic
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="底片自动校色 App — 从负片/正片扫描图自动校色",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default="",
                        help="输出路径（默认在原文件名上加 _corrected）")
    parser.add_argument("-f", "--film-type", default="negative",
                        choices=["negative", "positive"],
                        help="胶片类型: negative(负片) / positive(正片)")
    parser.add_argument("-w", "--warmth", default="natural",
                        choices=["natural", "kodak_gold", "fuji_superia", "cool"],
                        help="暖调风格")
    parser.add_argument("-s", "--strength", type=float, default=1.0,
                        help="暖调强度 0.0~2.0")
    parser.add_argument("-d", "--detector", default="auto",
                        choices=["auto", "heuristic", "vlm_api"],
                        help="偏色检测后端")
    parser.add_argument("--api-key", default="",
                        help="VL 模型 API Key")
    parser.add_argument("--api-base", default="",
                        help="VL 模型 API 地址")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="VL 模型名")
    parser.add_argument("--percentile", type=float, default=0.2,
                        help="色阶裁剪百分位")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="AI 反馈最大迭代次数")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="偏色容忍度")
    parser.add_argument("--no-cli", action="store_true",
                        help="启动 GUI 界面（而非 CLI 模式）")
    parser.add_argument("--gui", action="store_true",
                        help="启动 GUI 界面")

    args = parser.parse_args()

    # ── GUI 模式 ──
    if args.gui or args.no_cli:
        _launch_gui()
        return

    # ── CLI 模式 ──
    _run_cli(args)


def _run_cli(args):
    """CLI 模式执行校色"""
    from core import (
        Engine, CorrectionConfig,
        apply_channel_compensation, manual_compensation,
        analyze_mask, invert, auto_levels, apply_warmth,
    )

    # 加载图片
    print(f"[NCC] 加载: {args.input}")
    img_pil = Image.open(args.input)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img = np.array(img_pil, dtype=np.float32)

    # 配置引擎
    config = CorrectionConfig(
        film_type=args.film_type,
        warmth_style=args.warmth,
        warmth_strength=args.strength,
        levels_percentile=args.percentile,
        max_iterations=args.max_iter,
        cast_threshold=args.threshold,
        detector_mode=args.detector,
        vlm_api_key=args.api_key,
        vlm_api_base=args.api_base,
        vlm_model=args.model,
    )

    # 执行校色
    print(f"[NCC] 开始校色 (胶片类型: {args.film_type})")
    engine = Engine(config)
    result = engine.correct(img)

    # 输出信息
    print(f"[NCC] 完成!")
    print(f"     色罩分析: {result.mask_info.method} "
          f"(scale R={result.mask_info.scale_r:.3f}, "
          f"G={result.mask_info.scale_g:.3f}, "
          f"B={result.mask_info.scale_b:.3f})")
    print(f"     暖调: {result.warm_style}")
    print(f"     偏色检测迭代: {result.iterations} 次")
    print(f"     最终偏色: {result.final_cast.cast_type} "
          f"(severity={result.final_cast.severity:.3f})")

    # 保存
    output_path = args.output or _default_output(args.input)
    out_pil = Image.fromarray(result.image)
    out_pil.save(output_path)
    print(f"[NCC] 已保存: {output_path}")


def _default_output(input_path: str) -> str:
    """生成默认输出路径"""
    base, ext = os.path.splitext(input_path)
    return f"{base}_corrected{ext}"


def _launch_gui():
    """启动 GUI 界面"""
    try:
        from ui.app import launch
        launch()
    except ImportError as e:
        print(f"[NCC] GUI 启动失败: {e}")
        print("[NCC] 请安装依赖: pip install pillow numpy requests")
        sys.exit(1)


if __name__ == "__main__":
    main()
