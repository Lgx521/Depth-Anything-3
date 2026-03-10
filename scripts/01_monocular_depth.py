#!/usr/bin/env python3
"""
单目深度估计 + 单目点云生成脚本
使用 DA3METRIC-LARGE 模型，输出：
  - 深度图 (metric depth, 单位: 米)
  - 置信度图
  - 彩色深度可视化 (depth_vis)
  - 点云 GLB 文件（可在 3D 查看器中打开）
  - mini_npz (depth / conf / exts / ixts)

用法:
  # 处理单张图片
  python scripts/01_monocular_depth.py --input path/to/image.jpg

  # 处理图片目录
  python scripts/01_monocular_depth.py --input path/to/images/

  # 自定义输出目录
  python scripts/01_monocular_depth.py --input path/to/image.jpg --output-dir workspace/mono_out

  # 使用高分辨率处理
  python scripts/01_monocular_depth.py --input path/to/image.jpg --process-res 756
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DA3METRIC-LARGE 单目深度估计 + 点云生成"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="assets/examples/SOH/000.png",
        help="输入图片路径或图片目录",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="workspace/mono_output",
        help="输出目录",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="模型路径（HuggingFace repo 或本地目录）",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="推理分辨率（长边上限），默认 504",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="分辨率缩放策略",
    )
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="点云置信度过滤百分位（越小保留点越多），默认 40.0",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=1_000_000,
        help="点云最大点数，默认 1000000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def collect_images(input_path: str) -> list[str]:
    """收集图片路径列表（支持单文件 / 目录）"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff", "*.tif")
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(input_path, ext)))
        images.sort()
        if not images:
            raise ValueError(f"目录 '{input_path}' 中未找到图片文件")
        return images
    raise ValueError(f"输入路径不存在: {input_path}")


def save_depth_as_image(depth: np.ndarray, save_path: str) -> None:
    """将深度图保存为 16-bit PNG（深度值单位: 毫米）"""
    depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(save_path)


def print_prediction_info(prediction, images: list[str]) -> None:
    """打印预测结果摘要"""
    print("\n" + "=" * 50)
    print("📊 推理结果摘要")
    print("=" * 50)
    print(f"  输入图片数量: {len(images)}")
    print(f"  深度图形状:   {prediction.depth.shape}  (N, H, W) float32")
    print(f"  置信度形状:   {prediction.conf.shape}   (N, H, W) float32")
    print(f"  预处理图形状: {prediction.processed_images.shape}  (N, H, W, 3) uint8")

    depth = prediction.depth
    valid = depth > 0
    if valid.any():
        print(f"\n  深度统计 (有效像素):")
        print(f"    min  = {depth[valid].min():.3f} m")
        print(f"    max  = {depth[valid].max():.3f} m")
        print(f"    mean = {depth[valid].mean():.3f} m")

    if hasattr(prediction, "intrinsics") and prediction.intrinsics is not None:
        fx = prediction.intrinsics[0, 0, 0]
        fy = prediction.intrinsics[0, 1, 1]
        print(f"\n  相机内参 (第一帧):")
        print(f"    fx = {fx:.2f}  fy = {fy:.2f}")
    print("=" * 50 + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. 收集输入图片
    images = collect_images(args.input)
    print(f"🖼️  找到 {len(images)} 张图片: {[os.path.basename(p) for p in images]}")

    # 2. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(args.output_dir)}")

    # 3. 加载模型
    print(f"\n🔧 加载模型: {args.model_dir}  (设备: {args.device})")
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=args.device)
    print("✅ 模型加载完成")

    # 4. 推理
    # 导出格式: mini_npz（数值数据）+ glb（点云）+ depth_vis（彩色深度图）
    print("\n🚀 开始推理 ...")
    prediction = model.inference(
        image=images,
        export_dir=args.output_dir,
        export_format="mini_npz-glb-depth_vis",
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        show_cameras=True,
    )
    print("✅ 推理完成")

    # 5. 额外保存 16-bit 深度 PNG（metric depth，单位 mm）
    depth_png_dir = os.path.join(args.output_dir, "depth_16bit")
    os.makedirs(depth_png_dir, exist_ok=True)
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_depth_as_image(
            prediction.depth[i],
            os.path.join(depth_png_dir, f"{stem}_depth16.png"),
        )
    print(f"💾 16-bit 深度 PNG 已保存至: {depth_png_dir}")

    # 6. 打印摘要
    print_prediction_info(prediction, images)

    print("🎉 全部完成！输出文件:")
    print(f"   点云 (GLB)     : {args.output_dir}/scene.glb")
    print(f"   深度数值 (NPZ) : {args.output_dir}/scene.npz")
    print(f"   彩色深度图     : {args.output_dir}/depth_vis/")
    print(f"   16-bit 深度 PNG: {depth_png_dir}/")
    print()
    print("💡 提示: DA3METRIC-LARGE 输出的深度已是真实公制尺度（米）")
    print("         若需从其他模型获取公制深度，请参考 README 中 FAQ 的转换公式")


if __name__ == "__main__":
    main()
