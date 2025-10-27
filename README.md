## Sora2WatermarkRemover

AI-powered watermark removal for images and videos using Microsoft Florence-2 and LaMa inpainting. Includes a simple GUI and a CLI for batch processing.

### Why this repo exists
- I needed something lightweight and adjustable that runs well on an older NVIDIA GTX 1070.
- Clear controls for speed vs quality, with sensible defaults and small-VRAM friendliness.
- I translated the original materials to English and wrote a more comprehensive README.

What’s different from the original:
- English-first docs (French guides `DEMARRAGE_RAPIDE.md` and `INSTALLATION_FR.md` are still included).
- Expanded CLI reference, practical examples, and tips for older GPUs.
- Original: https://github.com/GitHub30/Sora2WatermarkRemover

### Demo
- TBD
---

### Features
- Detects Sora watermark regions using Florence-2
- Removes watermarks via LaMa inpainting (images) or fills regions while reconstructing video
- Batch process a directory of images and videos
- Optional transparent output for images
- Force output format (PNG, WEBP, JPG, MP4, AVI)
- GUI with progress and logs, plus full CLI

---

### Requirements
- Python 3.10+ (3.12 recommended)
- FFmpeg (optional but recommended for videos to keep original audio)

GPU (optional):
- Install a PyTorch wheel that matches your GPU/CUDA. For GTX 10xx cards (e.g., 1070), CUDA 11.8 builds are typically the sweet spot:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU-only is fine too (slower):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Quick Start (Windows)
Option A — One command PowerShell install (creates `.venv`):
```
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

Option B — Batch file:
```
install_windows.bat
```

Manual installation using Python venv:
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
iopaint download --model lama
python remwmgui.py
```

---

### Windows 10 GPU acceleration (WSL2 + Ubuntu)
Windows 10 does not natively support Linux GPU containers/apps like Windows 11 WSLg. The most reliable path for GPU acceleration here is to run the project inside WSL2 (Ubuntu) and install a CUDA-enabled PyTorch build.

High-level steps:
- Enable WSL2 and install Ubuntu
- Install the latest NVIDIA driver on Windows with WSL support (GeForce/Studio)
- Create a Python venv inside Ubuntu, install deps, and install a CUDA-enabled PyTorch wheel

1) Enable WSL2 and install Ubuntu (from elevated PowerShell):
```
wsl --install -d Ubuntu
wsl --set-default-version 2
```

2) Install latest NVIDIA Windows driver (with WSL support), then reboot Windows.

3) Inside Ubuntu (WSL2):
```
# Update packages and install basics (optional but recommended)
sudo apt update
sudo apt install -y python3-venv ffmpeg

# From the project directory (inside WSL)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install CUDA-enabled PyTorch (CUDA 11.8 generally works well for GTX 10xx like 1070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download LaMa weights
iopaint download --model lama

# Verify GPU is visible to PyTorch
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'CUDA', torch.version.cuda)"

# Run (CLI is recommended on Win10 WSL2; GUI may need an X server)
python remwm.py ./samples ./out --device cuda
```

Notes:
- Different GPUs/newer drivers may require a different PyTorch wheel index (e.g., cu121/cu124). Use the official PyTorch selector to match your setup.
- On Windows 10, WSLg is not bundled; GUI apps from WSL may require an external X server. Prefer the CLI in WSL, or run the GUI on native Windows.
- If videos have no audio, install `ffmpeg` in WSL as shown above.

---

### Quick Start (macOS/Linux)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
iopaint download --model lama
python remwmgui.py
```

Install FFmpeg if you want video audio preserved when processing videos.

---

### GUI Usage
1) Choose mode: Single File or Directory
2) Set Input Path and Output Path
3) Options:
   - Overwrite Existing Files
   - Make Watermark Transparent (images only)
   - Max BBox Percent (default 10%)
   - Force Format: PNG, WEBP, JPG, MP4, AVI, or None
4) Click Start

Notes for videos:
- Transparency is not supported for videos
- FFmpeg is required to merge the processed video with the original audio

---

### CLI Usage
```
python remwm.py INPUT_PATH OUTPUT_PATH \
  [--overwrite] [--transparent] \
  [--max-bbox-percent 10.0] \
  [--force-format {PNG,WEBP,JPG,MP4,AVI}] \
  [--frame-step 1] [--target-fps 0] \
  [--videos-only] [--num-workers N] \
  [--device {auto,cpu,cuda}] \
  [--florence-model microsoft/Florence-2-large] [--torch-threads N] \
  [--detect-prompt "watermark Sora logo"] \
  [--bbox-expand-px 4] [--mask-dilate-px 2] [--min-score 0.0] \
  [--florence-num-beams 1] [--florence-max-new-tokens 256] \
  [--detect-every 5] \
  [--lama-steps 40] [--lama-resize-limit 1280]
```

Defaults and tips:
- Device: `auto` chooses CUDA if available; force GPU with `--device cuda`.
- Florence-2: default `microsoft/Florence-2-large`. Use `...-base` for lower VRAM or CPU.
- Directories: add `--videos-only` to restrict; `--num-workers` enables CPU parallelism (GPU runs sequentially to avoid contention).

Examples:
```
# Image → auto format
python remwm.py input.jpg output.png

# Image with transparent watermark
python remwm.py input.png output.png --transparent

# Directory (images + videos), overwrite existing, limit bbox to 8%
python remwm.py ./samples ./out --overwrite --max-bbox-percent 8

# Video → force mp4, process every 2nd frame, target 24 fps, use GPU
python remwm.py input.mov output.mp4 --force-format MP4 --frame-step 2 --target-fps 24 --device cuda

# CPU parallel processing of directory (videos only) with 4 workers
python remwm.py ./videos ./out --videos-only --num-workers 4 --device cpu --florence-model microsoft/Florence-2-base --torch-threads 4

# Fastest on older GPU (GTX 1070 friendly)
python remwm.py ./in ./out \
  --device cuda \
  --florence-model microsoft/Florence-2-base \
  --florence-num-beams 1 --florence-max-new-tokens 128 \
  --detect-every 10 --frame-step 2 \
  --lama-steps 20 --lama-resize-limit 1024

# Highest quality (slower) on GPU
python remwm.py ./in ./out \
  --device cuda \
  --florence-model microsoft/Florence-2-large \
  --florence-num-beams 3 --florence-max-new-tokens 384 \
  --detect-every 1 --frame-step 1 --min-score 0.1 \
  --lama-steps 80 --lama-resize-limit 2048
```

---

### Performance Tips
- GPU is auto-detected; CUDA greatly speeds up processing. Force with `--device cuda`.
- Reduce `--max-bbox-percent` if the watermark is small
- For videos, increasing `--frame-step` can speed up processing at the cost of quality
 - On GTX 1070-class GPUs: prefer `Florence-2-base`, `--florence-num-beams 1`, `--florence-max-new-tokens 128`, and keep `--lama-resize-limit` ≤ 1280 to reduce VRAM pressure

---

### Troubleshooting
- Conda not found: install Miniconda and restart your terminal
- Florence-2 download slow or fails: ensure stable internet; retry the command
- LaMa model missing: run `iopaint download --model lama`
- FFmpeg not found: install FFmpeg and ensure it’s on PATH to keep audio in videos
- OpenCV codec errors on video: try forcing MP4 or AVI output

---

### License
See `LICENSE`.

---

### Acknowledgements
- Florence-2 by Microsoft
- LaMa Inpainting

