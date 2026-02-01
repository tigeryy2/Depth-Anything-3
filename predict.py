import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath
from imageio_ffmpeg import get_ffmpeg_exe


class Predictor(BasePredictor):
    def setup(self) -> None:
        repo_root = Path(__file__).resolve().parent
        self.repo_root = repo_root
        self.da3_streaming = repo_root / "da3_streaming" / "da3_streaming.py"
        self.da3_config = repo_root / "da3_streaming" / "configs" / "base_config.yaml"
        self.converter = repo_root / "tools" / "da3_to_nerfstudio.py"

        if not self.da3_streaming.exists():
            raise FileNotFoundError(f"Missing DA3 streaming script: {self.da3_streaming}")
        if not self.da3_config.exists():
            raise FileNotFoundError(f"Missing DA3 config: {self.da3_config}")
        if not self.converter.exists():
            raise FileNotFoundError(f"Missing converter: {self.converter}")

        self._ensure_weights()

        env = os.environ.copy()
        src_path = repo_root / "src"
        if src_path.exists():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else str(src_path)
        self.subprocess_env = env

    def _ensure_weights(self) -> None:
        weights_dir = self.repo_root / "weights"
        required = [
            weights_dir / "config.json",
            weights_dir / "model.safetensors",
            weights_dir / "dino_salad.ckpt",
        ]
        if all(path.exists() for path in required):
            return

        script = self.repo_root / "da3_streaming" / "scripts" / "download_weights.sh"
        if not script.exists():
            missing = ", ".join(str(path) for path in required if not path.exists())
            raise FileNotFoundError(
                f"Missing weights ({missing}) and no download script found at {script}"
            )

        print("Downloading DA3 streaming weights (one-time setup)...", flush=True)
        try:
            subprocess.run(["sh", str(script)], check=True, cwd=self.repo_root)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Failed to download DA3 streaming weights") from exc

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Weights download incomplete, missing: {', '.join(missing)}")

    def predict(
        self,
        video: CogPath = Input(description="Input MP4 video"),
        fps: float = Input(
            description="Frame extraction FPS (lower = faster)",
            default=5.0,
            ge=0.1,
            le=60.0,
        ),
        max_width: int = Input(
            description="Resize video to this max width (0 keeps original)",
            default=640,
            ge=0,
        ),
        max_frames: int = Input(
            description="Uniformly subsample to this many frames for training",
            default=300,
            ge=10,
        ),
    ) -> CogPath:
        tmp_dir = Path(tempfile.mkdtemp(prefix="da3_splat_"))
        frames_dir = tmp_dir / "frames"
        da3_out = tmp_dir / "da3_output"
        ns_out = tmp_dir / "splat_dataset"
        export_dir = tmp_dir / "splat_export"
        frames_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg = get_ffmpeg_exe()
        vf_parts = [f"fps={fps}"]
        if max_width > 0:
            vf_parts.append(f"scale={max_width}:-1")
        vf = ",".join(vf_parts)

        frame_pattern = frames_dir / "frame_%06d.png"
        extract_cmd = [
            ffmpeg,
            "-i",
            str(video),
            "-vf",
            vf,
            str(frame_pattern),
        ]
        subprocess.run(extract_cmd, check=True)

        da3_cmd = [
            sys.executable,
            str(self.da3_streaming),
            "--image_dir",
            str(frames_dir),
            "--config",
            str(self.da3_config),
            "--output_dir",
            str(da3_out),
        ]
        subprocess.run(da3_cmd, check=True, env=self.subprocess_env, cwd=self.repo_root)

        convert_cmd = [
            sys.executable,
            str(self.converter),
            "--da3-output",
            str(da3_out),
            "--images-dir",
            str(frames_dir),
            "--out-dir",
            str(ns_out),
            "--max-frames",
            str(max_frames),
            "--train",
            "--export",
            "--export-dir",
            str(export_dir),
        ]
        subprocess.run(convert_cmd, check=True, env=self.subprocess_env)

        ply_files = sorted(export_dir.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No .ply found in {export_dir}")

        output_dir = Path(tempfile.mkdtemp(prefix="da3_splat_out_"))
        output_ply = output_dir / "splat.ply"
        shutil.copy2(ply_files[0], output_ply)

        return CogPath(output_ply)
