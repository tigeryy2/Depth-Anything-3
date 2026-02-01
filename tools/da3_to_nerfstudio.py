#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from typing import Optional, List, Tuple


IMG_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")


def load_poses(poses_path: Path) -> np.ndarray:
    poses = []
    with poses_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 16:
                raise ValueError(f"Invalid pose line (need 16 floats): {line[:80]}")
            pose = np.array(values, dtype=np.float64).reshape(4, 4)
            poses.append(pose)
    return np.stack(poses, axis=0)


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    intrinsics = []
    with intrinsics_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 4:
                raise ValueError(f"Invalid intrinsics line (need 4 floats): {line[:80]}")
            intrinsics.append(values)
    return np.array(intrinsics, dtype=np.float64)


def list_images(images_dir: Path) -> List[Path]:
    images = [p for p in images_dir.iterdir() if p.suffix in IMG_EXTS]
    images.sort(key=lambda p: p.name)
    return images


def select_indices(
    num_frames: int, stride: Optional[int], max_frames: Optional[int]
) -> List[int]:
    if stride is not None and stride > 1:
        return list(range(0, num_frames, stride))
    if max_frames is not None and max_frames < num_frames:
        idx = np.linspace(0, num_frames - 1, num=max_frames, dtype=int)
        return idx.tolist()
    return list(range(num_frames))


def ensure_images(
    images: List[Path],
    out_dir: Path,
    mode: str,
) -> List[str]:
    if mode == "reference":
        return [str(p.resolve()) for p in images]

    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = []
    for src in images:
        dst = out_images_dir / src.name
        rel_paths.append(str(Path("images") / src.name))
        if dst.exists():
            continue
        if mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown images mode: {mode}")
    return rel_paths


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def write_transforms(
    out_dir: Path,
    image_paths: List[str],
    poses_c2w_cv: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: Tuple[int, int],
    include_ply: bool,
) -> Path:
    h, w = image_hw
    intrinsics_constant = np.allclose(intrinsics, intrinsics[0], rtol=1e-6, atol=1e-6)

    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])

    data: dict[str, object] = {
        "camera_model": "OPENCV",
        "w": int(w),
        "h": int(h),
        "frames": [],
    }

    if intrinsics_constant:
        fx, fy, cx, cy = intrinsics[0]
        data.update(
            {
                "fl_x": float(fx),
                "fl_y": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
            }
        )

    if include_ply:
        data["ply_file_path"] = "combined_pcd.ply"

    frames = []
    for file_path, c2w_cv, intr in zip(image_paths, poses_c2w_cv, intrinsics):
        c2w_gl = c2w_cv @ flip_yz
        frame = {
            "file_path": file_path,
            "transform_matrix": c2w_gl.tolist(),
        }
        if not intrinsics_constant:
            fx, fy, cx, cy = intr
            frame.update(
                {
                    "fl_x": float(fx),
                    "fl_y": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                }
            )
        frames.append(frame)

    data["frames"] = frames

    out_path = out_dir / "transforms.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    return out_path


def find_latest_config(outputs_root: Path) -> Optional[Path]:
    configs = list(outputs_root.rglob("config.yml"))
    if not configs:
        return None
    configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def detect_load_3d_points_flag() -> Optional[str]:
    try:
        result = subprocess.run(
            ["ns-train", "--help"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    helptext = f"{result.stdout}\n{result.stderr}"
    for flag in (
        "--pipeline.datamanager.dataparser.load_3D_points",
        "--pipeline.datamanager.dataparser.load_3d_points",
    ):
        if flag in helptext:
            return flag
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DA3-Streaming outputs to Nerfstudio dataset")
    parser.add_argument("--da3-output", required=True, help="DA3 output directory")
    parser.add_argument("--images-dir", required=True, help="Directory of input frames")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--images-mode",
        default="symlink",
        choices=["symlink", "copy", "reference"],
        help="How to place images in dataset (default: symlink)",
    )
    parser.add_argument("--stride", type=int, default=None, help="Keep every Nth frame")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Uniformly subsample to this many frames",
    )
    parser.add_argument("--skip-ply", action="store_true", help="Do not include point cloud")
    parser.add_argument(
        "--ply-path",
        default=None,
        help="Override combined_pcd.ply path (default: <da3-output>/pcd/combined_pcd.ply)",
    )
    parser.add_argument("--train", action="store_true", help="Run ns-train after prepare")
    parser.add_argument("--export", action="store_true", help="Run ns-export after prepare/train")
    parser.add_argument(
        "--splat-method",
        default="splatfacto",
        help="Nerfstudio method (default: splatfacto)",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory to search for latest ns config.yml",
    )
    parser.add_argument(
        "--ns-config",
        default=None,
        help="Explicit config.yml for ns-export (overrides auto-detect)",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Output dir for ns-export (default: <out-dir>/exports/splat)",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to ns-train (prefix with --)",
    )
    parser.add_argument(
        "--load-3d-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass load_3D_points flag to ns-train when supported (default: true)",
    )

    args = parser.parse_args()

    da3_output = Path(args.da3_output)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_path = da3_output / "camera_poses.txt"
    intrinsics_path = da3_output / "intrinsic.txt"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing camera poses: {poses_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")

    poses = load_poses(poses_path)
    intrinsics = load_intrinsics(intrinsics_path)
    images = list_images(images_dir)

    if len(images) != len(poses) or len(images) != len(intrinsics):
        raise ValueError(
            f"Count mismatch: images={len(images)}, poses={len(poses)}, intrinsics={len(intrinsics)}"
        )

    indices = select_indices(len(images), args.stride, args.max_frames)
    images = [images[i] for i in indices]
    poses = poses[indices]
    intrinsics = intrinsics[indices]

    with Image.open(images[0]) as im:
        w, h = im.size

    image_paths = ensure_images(images, out_dir, args.images_mode)

    include_ply = not args.skip_ply
    if include_ply:
        ply_src = Path(args.ply_path) if args.ply_path else da3_output / "pcd" / "combined_pcd.ply"
        if ply_src.exists():
            copy_or_link(ply_src, out_dir / "combined_pcd.ply", args.images_mode)
        else:
            print(f"Warning: PLY not found at {ply_src}; proceeding without it.")
            include_ply = False

    transforms_path = write_transforms(
        out_dir=out_dir,
        image_paths=image_paths,
        poses_c2w_cv=poses,
        intrinsics=intrinsics,
        image_hw=(h, w),
        include_ply=include_ply,
    )

    print(f"Wrote {transforms_path}")

    if args.train:
        cmd = [
            "ns-train",
            args.splat_method,
            "--data",
            str(out_dir),
        ]
        if args.load_3d_points:
            load_flag = detect_load_3d_points_flag()
            if load_flag is None:
                print(
                    "Warning: ns-train does not advertise a load_3D_points flag; "
                    "skipping it."
                )
            else:
                cmd.extend([load_flag, "True"])
        if args.train_args:
            cmd.extend(args.train_args)
        run_cmd(cmd)

    if args.export:
        if args.ns_config:
            config_path = Path(args.ns_config)
        else:
            config_path = find_latest_config(Path(args.outputs_root))
            if config_path is None:
                raise FileNotFoundError(
                    f"No config.yml found under {args.outputs_root}. Use --ns-config."
                )
        export_dir = Path(args.export_dir) if args.export_dir else out_dir / "exports" / "splat"
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_dir),
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
