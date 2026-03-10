#!/usr/bin/env python3
"""
多视角点云生成脚本
使用 DA3-LARGE（或指定任意 any-view 模型），输入多张图片，输出：
  - GLB 点云（含相机位姿线框）
  - mini_npz (depth / conf / exts / ixts)
  - 彩色深度可视化 (depth_vis)

适用场景:
  - 多张不同视角拍摄的同一场景（无需已知相机位姿）
  - 已有 COLMAP 位姿时可通过 --colmap-dir 传入，启用位姿条件推理

用法:
  # 基础多视角（自动估计位姿）
  python scripts/02_multiview_pointcloud.py --input assets/examples/SOH/

  # 指定模型和分辨率
  python scripts/02_multiview_pointcloud.py \
      --input assets/examples/SOH/ \
      --model-dir depth-anything/DA3-LARGE-1.1 \
      --process-res 756

  # 从 COLMAP 数据读取已知位姿（位姿条件推理）
  python scripts/02_multiview_pointcloud.py \
      --input path/to/colmap_dataset \
      --colmap-dir path/to/colmap_dataset \
      --sparse-subdir 0

  # 使用更精准的 ray head 位姿
  python scripts/02_multiview_pointcloud.py \
      --input assets/examples/SOH/ \
      --use-ray-pose
"""

import argparse
import glob
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DA3 多视角点云生成"
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
        default="workspace/multiview_output",
        help="输出目录",
    )
    # 模型
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3-LARGE-1.1",
        help="模型路径（HuggingFace repo 或本地目录）",
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
    # 位姿
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
        help="参考视角选择策略（视频序列推荐 middle）",
    )
    # COLMAP 已知位姿（可选）
    parser.add_argument(
        "--colmap-dir",
        type=str,
        default=None,
        help="COLMAP 数据集目录（含 images/ 和 sparse/），用于位姿条件推理",
    )
    parser.add_argument(
        "--sparse-subdir",
        type=str,
        default="",
        help="COLMAP sparse 子目录，如 '0' 表示 sparse/0/",
    )
    # 点云参数
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="点云置信度过滤百分位",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=2_000_000,
        help="点云最大点数（多视角可适当增大），默认 2000000",
    )
    parser.add_argument(
        "--no-show-cameras",
        action="store_true",
        help="GLB 中不显示相机线框",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def collect_images(input_path: str) -> list[str]:
    """支持目录、逗号分隔路径、单文件"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff", "*.tif")
    # 逗号分隔多路径
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


def load_colmap_poses(colmap_dir: str, sparse_subdir: str):
    """
    从 COLMAP 数据读取外参和内参，返回 (extrinsics, intrinsics, image_paths)
    extrinsics: (N, 4, 4)  world-to-camera
    intrinsics: (N, 3, 3)
    """
    from depth_anything_3.utils.read_write_model import (
        read_cameras_binary,
        read_images_binary,
    )

    sparse_path = os.path.join(colmap_dir, "sparse", sparse_subdir)
    cameras_bin = os.path.join(sparse_path, "cameras.bin")
    images_bin = os.path.join(sparse_path, "images.bin")

    if not os.path.exists(cameras_bin):
        raise FileNotFoundError(f"找不到 COLMAP cameras.bin: {cameras_bin}")
    if not os.path.exists(images_bin):
        raise FileNotFoundError(f"找不到 COLMAP images.bin: {images_bin}")

    cameras = read_cameras_binary(cameras_bin)
    images_data = read_images_binary(images_bin)

    # 按图片名称排序
    sorted_images = sorted(images_data.values(), key=lambda x: x.name)

    image_paths, exts, ixts = [], [], []
    images_dir = os.path.join(colmap_dir, "images")

    for img in sorted_images:
        img_path = os.path.join(images_dir, img.name)
        if not os.path.exists(img_path):
            print(f"  ⚠️  图片不存在，跳过: {img_path}")
            continue

        image_paths.append(img_path)

        # 外参 (w2c 4×4)
        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3:] = t
        exts.append(w2c)

        # 内参
        cam = cameras[img.camera_id]
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = cam.params[0]
            cx, cy = cam.params[1], cam.params[2]
        elif cam.model in ("PINHOLE", "OPENCV", "RADIAL", "FULL_OPENCV"):
            fx, fy = cam.params[0], cam.params[1]
            cx, cy = cam.params[2], cam.params[3]
        else:
            fx = fy = max(cam.width, cam.height)
            cx, cy = cam.width / 2.0, cam.height / 2.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        ixts.append(K)

    return np.stack(exts), np.stack(ixts), image_paths


def print_summary(prediction, images: list[str], has_poses: bool) -> None:
    print("\n" + "=" * 55)
    print("📊 多视角推理结果摘要")
    print("=" * 55)
    print(f"  输入图片数量 : {len(images)}")
    print(f"  位姿来源     : {'COLMAP 已知位姿（位姿条件推理）' if has_poses else '模型自动估计'}")
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
        fx = prediction.intrinsics[0, 0, 0]
        fy = prediction.intrinsics[0, 1, 1]
        print(f"  第一帧焦距   : fx={fx:.2f}  fy={fy:.2f}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. 收集图片
    if args.colmap_dir:
        print(f"📐 从 COLMAP 目录加载: {args.colmap_dir}")
        extrinsics, intrinsics, images = load_colmap_poses(
            args.colmap_dir, args.sparse_subdir
        )
        has_poses = True
        print(f"   读取到 {len(images)} 张有效图片（含已知位姿）")
    else:
        images = collect_images(args.input)
        extrinsics, intrinsics = None, None
        has_poses = False
        print(f"🖼️  找到 {len(images)} 张图片")

    if len(images) < 2:
        print("⚠️  警告: 多视角推理建议至少 2 张图片，当前输入仅 1 张")

    # 2. 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(args.output_dir)}")

    # 3. 加载模型
    print(f"\n🔧 加载模型: {args.model_dir}  (设备: {args.device})")
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=args.device)
    print("✅ 模型加载完成")

    # 4. 构建推理参数
    inference_kwargs = dict(
        image=images,
        export_dir=args.output_dir,
        export_format="mini_npz-glb-depth_vis",
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        show_cameras=not args.no_show_cameras,
    )

    # 已知位姿时：传入外参/内参，并对齐到输入尺度
    if has_poses:
        inference_kwargs["extrinsics"] = extrinsics
        inference_kwargs["intrinsics"] = intrinsics
        inference_kwargs["align_to_input_ext_scale"] = True

    # 5. 推理
    print(f"\n🚀 开始多视角推理 ...")
    print(f"   use_ray_pose      = {args.use_ray_pose}")
    print(f"   ref_view_strategy = {args.ref_view_strategy}")
    prediction = model.inference(**inference_kwargs)
    print("✅ 推理完成")

    # 6. 额外保存各视角深度为 numpy 文件
    depth_npy_dir = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(depth_npy_dir, exist_ok=True)
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        np.save(
            os.path.join(depth_npy_dir, f"{stem}_depth.npy"),
            prediction.depth[i],
        )
    print(f"💾 逐帧深度 .npy 已保存至: {depth_npy_dir}")

    # 7. 打印摘要
    print_summary(prediction, images, has_poses)

    print("🎉 全部完成！输出文件:")
    print(f"   点云 (GLB)     : {args.output_dir}/scene.glb")
    print(f"   深度数值 (NPZ) : {args.output_dir}/scene.npz")
    print(f"   彩色深度图     : {args.output_dir}/depth_vis/")
    print(f"   逐帧深度 (npy) : {depth_npy_dir}/")
    print()
    print("💡 提示: 可用 da3 gallery --gallery-dir workspace 在浏览器中查看 GLB")


if __name__ == "__main__":
    main()
