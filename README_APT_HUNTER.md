# Depth Anything 3 (Local Setup Notes)

Short, practical notes for our deployment/setup and DA3-Streaming usage.

## Setup (Ubuntu GPU host)

```bash
git submodule update --init --recursive

# create/update lockfile, then sync deps (Python >= 3.10)
uv lock --preview-features extra-build-dependencies
uv sync --extra streaming
uv pip install -e .
```

Notes:
- The `da3_streaming/loop_utils/salad` dependency is a git submodule; it must be initialized.
- `uv lock` is intended to run on the Ubuntu GPU host (not on Apple Silicon).

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
