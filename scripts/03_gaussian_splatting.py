#!/usr/bin/env python3
"""
3D Gaussian Splatting (3DGS) 生成脚本
使用 DA3-GIANT（或 DA3NESTED-GIANT-LARGE）模型，输入多视角图片，输出：
  - 3DGS PLY 文件（可在 SuperSplat / SPARK 等查看器中打开）
  - 3DGS 渲染视频
  - GLB 点云
  - mini_npz (depth / conf / exts / ixts)

注意: infer_gs=True 仅支持 da3-giant 和 da3nested-giant-large 两种模型！
      建议 GPU 显存 ≥ 24 GB（giant 模型较大）。

用法:
  # 基础 3DGS 生成（PLY + 渲染视频）
  python scripts/03_gaussian_splatting.py --input assets/examples/SOH/

  # 只导出 PLY，不渲染视频（节省时间）
  python scripts/03_gaussian_splatting.py --input assets/examples/SOH/ --no-video

  # 使用 nested 模型（含公制深度，效果更好）
  python scripts/03_gaussian_splatting.py \
      --input assets/examples/SOH/ \
      --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1

  # 指定渲染轨迹和分辨率
  python scripts/03_gaussian_splatting.py \
      --input assets/examples/SOH/ \
      --trj-mode interpolate_smooth \
      --render-chunk-size 4

  # 使用更精准的 ray head 位姿
  python scripts/03_gaussian_splatting.py \
      --input assets/examples/SOH/ \
      --use-ray-pose
"""

import argparse
import glob
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.npz import export_to_mini_npz


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DA3-GIANT / DA3NESTED 3D Gaussian Splatting 生成"
    )
    # 输入输出
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="assets/examples/SOH",
        help="输入图片目录（或逗号分隔的多个图片路径）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="workspace/gs_output",
        help="输出目录",
    )
    # 模型
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3-GIANT-1.1",
        help="模型路径，必须是 da3-giant 或 da3nested-giant-large 系列",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    # 分辨率
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="推理分辨率，默认 504",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    # 位姿估计
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="使用 ray head 估计位姿（更精准，略慢）",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        choices=["saddle_balanced", "saddle_sim_range", "first", "middle"],
        help="参考视角选择策略",
    )
    # 3DGS 导出控制
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成 3DGS 渲染视频（节省时间和显存）",
    )
    parser.add_argument(
        "--gs-views-interval",
        type=int,
        default=1,
        help="每隔 N 帧导出一次 3DGS，默认 1（每帧都导出）",
    )
    # 渲染视频参数
    parser.add_argument(
        "--trj-mode",
        type=str,
        default="interpolate_smooth",
        help="渲染轨迹模式: interpolate_smooth / 其他预设轨迹",
    )
    parser.add_argument(
        "--render-chunk-size",
        type=int,
        default=8,
        help="渲染批大小（显存不足时可减小），默认 8",
    )
    parser.add_argument(
        "--vis-depth",
        type=str,
        default=None,
        help="深度可视化拼接方式: hcat（水平）/ vcat（垂直）/ None（不显示深度）",
    )
    parser.add_argument(
        "--video-quality",
        type=str,
        default="high",
        choices=["high", "medium", "low"],
        help="渲染视频质量",
    )
    # 点云参数
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="GLB 点云置信度过滤百分位",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=1_000_000,
        help="GLB 点云最大点数",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def collect_images(input_path: str) -> list[str]:
    """支持目录、逗号分隔路径、单文件"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff", "*.tif")
    if "," in input_path:
        paths = [p.strip() for p in input_path.split(",")]
        for p in paths:
            if not os.path.isfile(p):
                raise ValueError(f"文件不存在: {p}")
        return sorted(paths)
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


def check_model_supports_gs(model_dir: str) -> None:
    """检查模型是否支持 GS，不支持则给出明确提示"""
    name = model_dir.lower()
    supported_keywords = ["giant", "nested"]
    if not any(kw in name for kw in supported_keywords):
        print("\n" + "!" * 60)
        print("⚠️  警告: 3DGS 仅支持 da3-giant 和 da3nested-giant-large 系列模型")
        print(f"   当前模型: {model_dir}")
        print("   如果模型不支持，推理将报错。请确认模型名称正确。")
        print("!" * 60 + "\n")


def print_summary(prediction, images: list[str], export_format: str) -> None:
    print("\n" + "=" * 55)
    print("📊 3DGS 推理结果摘要")
    print("=" * 55)
    print(f"  输入图片数量 : {len(images)}")
    print(f"  导出格式     : {export_format}")
    print(f"  深度图形状   : {prediction.depth.shape}")
    print(f"  置信度形状   : {prediction.conf.shape}")

    depth = prediction.depth
    valid = depth > 0
    if valid.any():
        print(f"\n  深度统计 (有效像素):")
        print(f"    min  = {depth[valid].min():.4f}")
        print(f"    max  = {depth[valid].max():.4f}")
        print(f"    mean = {depth[valid].mean():.4f}")

    if hasattr(prediction, "extrinsics") and prediction.extrinsics is not None:
        print(f"\n  估计外参形状 : {prediction.extrinsics.shape}  (N, 3, 4)")
    if hasattr(prediction, "intrinsics") and prediction.intrinsics is not None:
        print(f"  估计内参形状 : {prediction.intrinsics.shape}  (N, 3, 3)")

    # Gaussian 数量
    if hasattr(prediction, "aux") and prediction.aux is not None:
        gaussians = prediction.aux.get("gaussians")
        if gaussians is not None:
            print(f"\n  Gaussian 数量 : {gaussians}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. 收集图片
    images = collect_images(args.input)
    print(f"🖼️  找到 {len(images)} 张图片")
    if len(images) < 2:
        print("⚠️  警告: 3DGS 生成建议至少提供 3 张以上图片以获得好效果")

    # 2. 检查模型兼容性
    check_model_supports_gs(args.model_dir)

    # 3. 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(args.output_dir)}")

    # 4. 加载模型
    print(f"\n🔧 加载模型: {args.model_dir}  (设备: {args.device})")
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=args.device)
    print("✅ 模型加载完成")

    # 5. 推理（不传 export_dir，避免 api.py 强制追加 gs_video）
    # 已安装版本的 api.py 在 infer_gs=True 且 format 含 'gs' 时会强制追加 gs_video，
    # 绕过方法：先拿到 prediction，再手动调用各导出函数。
    print(f"\n🚀 开始 3DGS 推理 ...")
    print(f"   infer_gs          = True")
    print(f"   use_ray_pose      = {args.use_ray_pose}")
    print(f"   ref_view_strategy = {args.ref_view_strategy}")

    prediction = model.inference(
        image=images,
        infer_gs=True,
        export_dir=None,                        # ← 不让 api 自动导出
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
    )
    print("✅ 推理完成")

    # 6. 手动导出（精确控制，不触发强制 gs_video）
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n💾 开始导出 ...")

    print("   导出 mini_npz ...")
    export_to_mini_npz(prediction, args.output_dir)

    print("   导出 GLB 点云 ...")
    export_to_glb(
        prediction,
        args.output_dir,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        show_cameras=True,
    )

    print("   导出 3DGS PLY ...")
    export_to_gs_ply(
        prediction,
        args.output_dir,
        gs_views_interval=args.gs_views_interval,
    )

    if not args.no_video:
        from depth_anything_3.utils.export.gs import export_to_gs_video
        print("   渲染 3DGS 视频 ...")
        print(f"   trj_mode={args.trj_mode}  chunk_size={args.render_chunk_size}  vis_depth={args.vis_depth}")
        export_to_gs_video(
            prediction,
            args.output_dir,
            trj_mode=args.trj_mode,
            chunk_size=args.render_chunk_size,
            vis_depth=args.vis_depth,
            video_quality=args.video_quality,
            enable_tqdm=True,
        )
    print("✅ 导出完成")

    # 8. 打印摘要
    export_desc = "mini_npz-glb-gs_ply" + ("" if args.no_video else "-gs_video")
    print_summary(prediction, images, export_desc)

    print("🎉 全部完成！输出文件:")
    print(f"   点云 (GLB)      : {args.output_dir}/scene.glb")
    print(f"   深度数值 (NPZ)  : {args.output_dir}/scene.npz")
    print(f"   3DGS PLY        : {args.output_dir}/gs_ply/")
    if not args.no_video:
        print(f"   3DGS 渲染视频   : {args.output_dir}/gs_video/")
    print()
    print("💡 提示:")
    print("   - PLY 文件可在 https://superspl.at/editor 或 https://sparkjs.dev/viewer/ 中查看")
    print("   - 若 GPU 显存不足，可减小 --render-chunk-size 或添加 --no-video")
    print("   - 若需公制尺度 3DGS，请使用 DA3NESTED-GIANT-LARGE-1.1 模型")


if __name__ == "__main__":
    main()
