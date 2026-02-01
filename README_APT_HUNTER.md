# Depth Anything 3 (Local Setup Notes)

Short, practical notes for our deployment/setup and DA3-Streaming usage.

## Setup (Ubuntu GPU host)

```bash
git submodule update --init --recursive

# create/update lockfile, then sync deps (Python >= 3.10)
uv lock --preview-features extra-build-dependencies
uv sync --extra streaming --upgrade
uv pip install -e .
```

Notes:
- The `da3_streaming/loop_utils/salad` dependency is a git submodule; it must be initialized.
- `uv lock` is intended to run on the Ubuntu GPU host (not on Apple Silicon).
- This project is currently pinned to Python 3.10 for Nerfstudio/gsplat compatibility.

## DA3-Streaming prediction (video in ./temp)

1) Extract frames from the video (uses the bundled ffmpeg from `imageio-ffmpeg`):

```bash
mkdir -p ./temp/extract_images
FFMPEG_BIN=$(python - <<'PY'
from imageio_ffmpeg import get_ffmpeg_exe
print(get_ffmpeg_exe())
PY
)
"$FFMPEG_BIN" -i ./temp/apt_01.mp4 -vf "fps=5,scale=640:-1" ./temp/extract_images/frame_%06d.png
```

2) Run streaming inference:

```bash
uv run python da3_streaming.py \
  --image_dir ./temp/extract_images \
  --output_dir ./temp/da3_streaming_output
```

## Splat pipeline (DA3 -> Nerfstudio -> .ply)

This uses `tools/da3_to_nerfstudio.py` to build `transforms.json`, train `splatfacto`,
and export a Gaussian Splat `.ply`.

```bash
uv run python tools/da3_to_nerfstudio.py \
  --da3-output ./temp/da3_streaming_output \
  --images-dir ./temp/extract_images \
  --out-dir ./temp/splat \
  --max-frames 300 \
  --train \
  --export
```

Preview during training:
- Nerfstudio viewer will start automatically; open the URL printed by `ns-train` (usually `http://localhost:7007`).
- For remote hosts, use SSH tunneling: `ssh -L 7007:127.0.0.1:7007 <user>@<host>`.

## Cog (MP4 -> Splat)

We provide `predict.py` and `cog.yaml` so Cog can run the full pipeline
from MP4 to `.ply`.

Build:

```bash
sudo cog build
```

Run (note: requires sudo):

```bash
sudo cog run python -m cog predict \
  -i video=@./temp/apt_02.mp4 \
  -i fps=5 \
  -i max_width=640 \
  -i max_frames=300
```

The output is a `splat.ply` file.
