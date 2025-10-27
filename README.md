## Sora2WatermarkRemover

AI-powered watermark removal for images and videos using Microsoft Florence-2 and LaMa inpainting. Includes a simple GUI and a CLI for batch processing.

### Demo
- Removed watermark: [out.webm](https://github.com/user-attachments/assets/d902d040-f54c-4958-8d27-8b3c3bcbb6dd)
- Source clip: https://github.com/user-attachments/assets/8deffd66-b961-4ec2-9dc0-97695b0f91c5

### Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Iqu4RZ9WAhcbO1Jn0wCkMOsw2l1p6z62?usp=sharing)

### Video Overview
[![YouTube](https://img.youtube.com/vi/HkXD4zwk6WY/0.jpg)](https://www.youtube.com/watch?v=HkXD4zwk6WY)

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
  [--videos-only] [--num-workers N] [--device {auto,cpu,cuda}] \
  [--florence-model microsoft/Florence-2-base] [--torch-threads N]
```

Examples:
```
# Image → auto format
python remwm.py input.jpg output.png

# Image with transparent watermark
python remwm.py input.png output.png --transparent

# Directory (images + videos), overwrite existing, limit bbox to 8%
python remwm.py ./samples ./out --overwrite --max-bbox-percent 8

# Video → force mp4, process every 2nd frame, target 24 fps
python remwm.py input.mov output.mp4 --force-format MP4 --frame-step 2 --target-fps 24

# CPU parallel processing of directory (videos only) with 4 workers
python remwm.py ./videos ./out --videos-only --num-workers 4 --device cpu --florence-model microsoft/Florence-2-base --torch-threads 4
```

---

### Performance Tips
- GPU is auto-detected; CUDA greatly speeds up processing
- Reduce `--max-bbox-percent` if the watermark is small
- For videos, increasing `--frame-step` can speed up processing at the cost of quality

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

