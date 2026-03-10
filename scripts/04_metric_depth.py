#!/usr/bin/env python3
"""
米制深度估计脚本
支持三种模式，均输出单位为【米】的深度图：

  模式 A: DA3METRIC-LARGE        — 直接输出公制深度，无需相机参数
  模式 B: DA3NESTED-GIANT-LARGE  — 直接输出公制深度，同时估计相机位姿
  模式 C: DA3-GIANT / DA3-LARGE  — 相对深度，通过 FAQ 公式转换为米：
            metric_depth = focal * net_output / 300.
            （需通过 --focal 传入像素焦距，或让脚本从 EXIF 中自动读取）

输出文件:
  - depth_16bit/<name>_depth16.png   : 16-bit PNG，单位毫米，无损
  - depth_npy/<name>_depth.npy       : float32 numpy，单位米
  - depth_vis/<name>_depth_vis.*     : 彩色伪彩色深度可视化图

用法:
  # 模式 A: DA3METRIC 直接米制深度
  python scripts/04_metric_depth.py \
      --input dataset/ \
      --model-dir ./checkpoints/DA3METRIC

  # 模式 B: DA3NESTED 米制深度（精度更高）
  python scripts/04_metric_depth.py \
      --input dataset/ \
      --model-dir ./checkpoints/DA3NESTED

  # 模式 C: DA3GIANT 相对深度 -> 米制（需已知焦距）
  python scripts/04_metric_depth.py \
      --input dataset/ \
      --model-dir ./checkpoints/ \
      --focal 800

  # 模式 C: 焦距从 EXIF 自动读取（需 piexif）
  python scripts/04_metric_depth.py \
      --input dataset/ \
      --model-dir ./checkpoints/ \
      --focal-from-exif
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
from depth_anything_3.utils.export.npz import export_to_mini_npz


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="米制深度估计（支持 DA3METRIC / DA3NESTED / DA3-GIANT 三种模式）"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入图片路径或图片目录")
    parser.add_argument("--output-dir", "-o", type=str, default="workspace/metric_depth_output",
                        help="输出目录")
    parser.add_argument("--model-dir", type=str, default="depth-anything/DA3METRIC-LARGE",
                        help="模型路径（HuggingFace repo 或本地目录）")
    parser.add_argument("--process-res", type=int, default=504,
                        help="推理分辨率（长边上限），默认 504")
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize",
                        choices=["upper_bound_resize", "lower_bound_resize"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # 模式 C 专用: 相对深度 -> 米制 转换参数
    fov_group = parser.add_mutually_exclusive_group()
    fov_group.add_argument("--focal", type=float, default=None,
                           help="[模式C] 相机像素焦距 (fx≈fy)，用于 metric = focal * depth / 300")
    fov_group.add_argument("--focal-from-exif", action="store_true",
                           help="[模式C] 从图片 EXIF 自动读取焦距（需要 piexif 或 Pillow EXIF 支持）")
    fov_group.add_argument("--fov", type=float, default=None,
                           help="[模式C] 水平视角（度），自动换算为焦距，如 --fov 60")

    # 导出控制
    parser.add_argument("--no-glb", action="store_true",
                        help="不导出 GLB 点云（节省时间）")
    parser.add_argument("--conf-thresh-percentile", type=float, default=40.0)
    parser.add_argument("--num-max-points", type=int, default=1_000_000)
    return parser.parse_args()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def collect_images(input_path: str) -> list[str]:
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


def read_focal_from_exif(img_path: str) -> float | None:
    """尝试从 EXIF 读取等效焦距并换算为像素焦距"""
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None
        # Tag 37386: FocalLength (mm), Tag 41989: FocalLengthIn35mmFilm
        focal_mm = exif_data.get(37386)
        if focal_mm is not None:
            focal_mm = focal_mm[0] / focal_mm[1] if isinstance(focal_mm, tuple) else focal_mm
        focal_35mm = exif_data.get(41989)
        w, h = img.size
        long_edge = max(w, h)
        if focal_35mm:
            # 35mm 全画幅对角线 = 43.27mm，换算到像素
            sensor_diag_mm = 43.27
            img_diag_px = math.sqrt(w ** 2 + h ** 2)
            focal_px = focal_35mm * img_diag_px / sensor_diag_mm
            return focal_px
        if focal_mm:
            # 粗估：假设传感器长边 ≈ 6mm（手机）或 36mm（全幅），取中间值
            # 这里只是粗略估计，建议手动指定
            print("  ⚠️  EXIF 只有原始焦距(mm)，无 35mm 等效值，估计可能不准")
            sensor_long_mm = 6.0
            focal_px = focal_mm * long_edge / sensor_long_mm
            return focal_px
    except Exception:
        pass
    return None


def fov_to_focal(fov_deg: float, width: int) -> float:
    """水平视角（度）转像素焦距"""
    return width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


def save_depth_16bit(depth_m: np.ndarray, save_path: str) -> None:
    """保存 16-bit PNG，单位毫米（0~65.535m 范围内无损）"""
    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(save_path)


def detect_model_type(model_dir: str) -> str:
    """根据 config.json 的 model_name 判断模型类型"""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return "unknown"
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("model_name", "unknown").lower()


def print_summary(images: list[str], depth: np.ndarray, model_type: str,
                  is_metric: bool, focal_used: float | None) -> None:
    print("\n" + "=" * 55)
    print("📊 米制深度估计结果摘要")
    print("=" * 55)
    print(f"  模型类型   : {model_type}")
    print(f"  是否直接米制: {'是' if is_metric else '否（FAQ 公式转换）'}")
    if focal_used is not None:
        print(f"  使用焦距   : {focal_used:.2f} px")
    print(f"  图片数量   : {len(images)}")
    print(f"  深度图形状 : {depth.shape}  (N, H, W)  float32  单位: 米")
    valid = depth > 0
    if valid.any():
        print(f"\n  深度统计 (有效像素):")
        print(f"    min  = {depth[valid].min():.3f} m")
        print(f"    max  = {depth[valid].max():.3f} m")
        print(f"    mean = {depth[valid].mean():.3f} m")
        print(f"    std  = {depth[valid].std():.3f} m")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    images = collect_images(args.input)
    print(f"🖼️  找到 {len(images)} 张图片")
    os.makedirs(args.output_dir, exist_ok=True)

    # 判断模型类型
    model_type = detect_model_type(args.model_dir)
    is_direct_metric = any(kw in model_type for kw in ("metric", "nested"))
    print(f"🔍 检测到模型类型: {model_type}")
    if is_direct_metric:
        print("   → 直接输出公制深度（米），无需额外转换")
    else:
        print("   → 输出相对深度，需通过 FAQ 公式转换为米")
        if args.focal is None and not args.focal_from_exif and args.fov is None:
            print("   ⚠️  未指定 --focal / --fov / --focal-from-exif，将使用假设 60° FOV 估算焦距")

    # 加载模型
    print(f"\n🔧 加载模型: {args.model_dir}  (设备: {args.device})")
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=args.device)
    print("✅ 模型加载完成")

    # 推理（不传 export_dir，避免 api.py 的自动导出问题）
    print("\n🚀 开始推理 ...")
    prediction = model.inference(
        image=images,
        export_dir=None,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    )
    print("✅ 推理完成")

    depth = prediction.depth.copy()  # (N, H, W) float32

    # ── 模式 C: 相对深度 → 米制 ──────────────────
    focal_used = None
    if not is_direct_metric:
        N, H, W = depth.shape

        # 确定各帧焦距
        focals: list[float] = []
        for i, img_path in enumerate(images):
            fx = None
            if args.focal is not None:
                fx = args.focal
            elif args.fov is not None:
                fx = fov_to_focal(args.fov, W)
            elif args.focal_from_exif:
                fx = read_focal_from_exif(img_path)
                if fx is None:
                    print(f"  ⚠️  [{os.path.basename(img_path)}] EXIF 无焦距信息，回退到 60° FOV 估算")
            if fx is None:
                fx = fov_to_focal(60.0, W)
            focals.append(fx)

        focal_used = focals[0]

        # 用模型输出的内参焦距覆盖（若模型估计了内参）
        if prediction.intrinsics is not None:
            print("   检测到模型预测了相机内参，使用预测焦距进行单位转换")
            for i in range(N):
                fx_pred = float(prediction.intrinsics[i, 0, 0])
                fy_pred = float(prediction.intrinsics[i, 1, 1])
                fx_avg = (fx_pred + fy_pred) / 2.0
                # FAQ: metric_depth = focal * net_output / 300.
                depth[i] = fx_avg * depth[i] / 300.0
            focal_used = float(prediction.intrinsics[0, 0, 0])
        else:
            for i in range(N):
                depth[i] = focals[i] * depth[i] / 300.0

        # 写回 prediction 供后续导出
        prediction.depth = depth

    # ── 合成内参和外参（GLB 导出两者都需要）────────
    N, H, W = depth.shape
    if prediction.intrinsics is None:
        fx = focal_used if focal_used is not None else fov_to_focal(60.0, W)
        K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]], dtype=np.float32)
        prediction.intrinsics = np.tile(K[None], (N, 1, 1))
        print(f"   [合成内参] fx=fy={fx:.1f}")
    if prediction.extrinsics is None:
        # 默认外参：每帧相机在世界原点朝前，w2c = [R|t] = [I|0]
        ext = np.eye(3, 4, dtype=np.float32)
        prediction.extrinsics = np.tile(ext[None], (N, 1, 1))
        print("   [合成外参] 使用单位矩阵（相机位于世界原点）")
    if prediction.conf is None:
        # GLB 导出需要置信度图，DA3METRIC 不输出，用全1代替（所有点均保留）
        prediction.conf = np.ones_like(depth)
        print("   [合成置信度] 使用全1（保留所有点）")

    # ── 导出 depth_vis ────────────────────────────
    export_to_depth_vis(prediction, args.output_dir)

    # ── 导出 mini_npz ─────────────────────────────
    export_to_mini_npz(prediction, args.output_dir)

    # ── 导出 16-bit PNG ───────────────────────────
    depth_16bit_dir = os.path.join(args.output_dir, "depth_16bit")
    os.makedirs(depth_16bit_dir, exist_ok=True)
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(depth_16bit_dir, f"{stem}_depth16.png")
        save_depth_16bit(depth[i], save_path)

    # ── 导出 float32 npy ──────────────────────────
    depth_npy_dir = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(depth_npy_dir, exist_ok=True)
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(depth_npy_dir, f"{stem}_depth.npy"), depth[i])

    # ── 可选：导出 GLB 点云 ───────────────────────
    if not args.no_glb:
        from depth_anything_3.utils.export.glb import export_to_glb
        export_to_glb(
            prediction,
            args.output_dir,
            conf_thresh_percentile=args.conf_thresh_percentile,
            num_max_points=args.num_max_points,
            show_cameras=is_direct_metric is False and prediction.extrinsics is not None,
        )

    print_summary(images, depth, model_type, is_direct_metric, focal_used)

    print("🎉 全部完成！输出文件:")
    print(f"   彩色深度图      : {args.output_dir}/depth_vis/")
    print(f"   16-bit PNG (mm) : {depth_16bit_dir}/")
    print(f"   Float32 NPY (m) : {depth_npy_dir}/")
    print(f"   数值存档 (NPZ)  : {args.output_dir}/scene.npz")
    if not args.no_glb:
        print(f"   点云 (GLB)      : {args.output_dir}/scene.glb")
    print()
    print("💡 读取深度示例:")
    print("   depth_m = np.load('depth_npy/xxx_depth.npy')         # float32, 单位 m")
    print("   depth_m = np.array(Image.open('depth_16bit/xxx.png')) / 1000.0  # uint16→m")


if __name__ == "__main__":
    main()
