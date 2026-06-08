#!/usr/bin/env python3
"""
底片自动校色 App — 启动入口

用法：
    python main.py <image> [options]
    python main.py <img1> <img2> ... [options]   ← 批量处理

    python main.py scan.tiff
    python main.py scan.tiff --film-type positive --warmth kodak_gold
    python main.py scan.tiff --output result.jpg
    python main.py scan.tiff --detector heuristic
    python main.py img1.tiff img2.tiff img3.tiff -w none   ← 批量
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

BATCH_LIMIT = 40


def main():
    parser = argparse.ArgumentParser(
        description="底片自动校色 App — 从负片/正片扫描图自动校色",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="*", default=[],
                        help="输入图片路径（支持多个，批量处理最多40张）")
    parser.add_argument("-o", "--output", default="",
                        help="输出路径（默认在原文件名上加 _corrected）")
    parser.add_argument("-f", "--film-type", default="negative",
                        choices=["negative", "positive"],
                        help="胶片类型: negative(负片) / positive(正片)")
    parser.add_argument("-w", "--warmth", default="natural",
                        choices=["none", "natural", "kodak_gold", "fuji_superia", "cool"],
                        help="暖调风格 (none=无色偏)")
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
    if not args.input:
        parser.print_usage()
        print("[NCC] 错误: CLI 模式需要输入图片路径，使用 --gui 启动图形界面")
        sys.exit(1)

    if len(args.input) > BATCH_LIMIT:
        print(f"[NCC] 错误: 一次最多处理 {BATCH_LIMIT} 张图片，当前 {len(args.input)} 张")
        sys.exit(1)

    _run_cli(args)


def _run_cli(args):
    """CLI 模式执行校色（支持单张和批量）"""
    from core import (
        Engine, CorrectionConfig, ConfigManager,
        apply_channel_compensation, manual_compensation,
        analyze_mask, invert, auto_levels, apply_warmth,
    )

    # ── 加载配置文件（~/.negative-corrector/config.json5）──
    cfg_mgr = ConfigManager()
    cfg_mgr.init_default_config()
    cfg = cfg_mgr.load()

    # 配置文件值作为基础，CLI 参数覆盖
    film_type = args.film_type or cfg.correction.film_type
    warmth_style = args.warmth or cfg.correction.warmth_style
    warmth_strength = args.strength if args.strength != 1.0 else cfg.correction.warmth_strength
    levels_percentile = args.percentile if args.percentile != 0.2 else cfg.correction.levels_percentile
    detector_mode = args.detector if args.detector != "auto" else cfg.detector.mode
    api_key = args.api_key or cfg_mgr.get_api_key("openrouter") or cfg_mgr.get_api_key("custom")
    api_base = args.api_base or cfg.detector.api_base
    model = args.model if args.model != "openai/gpt-4o-mini" else cfg.detector.model
    max_iter = args.max_iter if args.max_iter != 10 else cfg.detector.max_iterations
    threshold = args.threshold if args.threshold != 0.15 else cfg.detector.cast_threshold

    # 配置引擎（各批处理图片共享同一配置）
    config = CorrectionConfig(
        film_type=film_type,
        warmth_style=warmth_style,
        warmth_strength=warmth_strength,
        levels_percentile=levels_percentile,
        max_iterations=max_iter,
        cast_threshold=threshold,
        detector_mode=detector_mode,
        vlm_api_key=api_key,
        vlm_api_base=api_base,
        vlm_model=model,
    )

    inputs = args.input  # 可能有多张图
    is_batch = len(inputs) > 1
    total = len(inputs)

    if is_batch:
        print(f"\n{'='*50}")
        print(f"  📦 批量处理 {total} 张图片")
        print(f"{'='*50}")

    for idx, input_path in enumerate(inputs, 1):
        try:
            if is_batch:
                print(f"\n--- [{idx}/{total}] {os.path.basename(input_path)} ---")

            # 加载图片
            print(f"[NCC] 加载: {input_path}")
            img_pil = Image.open(input_path)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            img = np.array(img_pil, dtype=np.float32)

            # 执行校色
            engine = Engine(config)
            result = engine.correct(img)

            # 保存
            output_path = args.output or _default_output(input_path)
            out_pil = Image.fromarray(np.clip(result.image, 0, 255).astype(np.uint8))
            out_pil.save(output_path)

            # 单图模式输出详细信息，批量模式只输出文件名
            if is_batch:
                print(f"  ✅ {os.path.basename(output_path)}")
            else:
                print(f"[NCC] 完成!")
                print(f"     色罩分析: {result.mask_info.method} "
                      f"(scale R={result.mask_info.scale_r:.3f}, "
                      f"G={result.mask_info.scale_g:.3f}, "
                      f"B={result.mask_info.scale_b:.3f})")
                print(f"     暖调: {result.warm_style}")
                print(f"     偏色检测迭代: {result.iterations} 次")
                print(f"     最终偏色: {result.final_cast.cast_type} "
                      f"(severity={result.final_cast.severity:.3f})")
                print(f"[NCC] 已保存: {output_path}")

        except Exception as e:
            print(f"[NCC] ❌ 处理失败 [{idx}/{total}] {input_path}: {e}")

    if is_batch:
        print(f"\n{'='*50}")
        print(f"  ✅ 批量处理完成 ({total} 张)")
        print(f"{'='*50}\n")


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
