# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mosaika (モザイカ) is a Python CLI tool that applies pixelation/mosaic censoring to MP4 videos using a two-stage detection pipeline: YOLOv11 pose estimation to locate crotch regions, then NudeNet (and optionally YOLO NSFW models) to detect genitalia within those regions. This spatial filtering approach reduces false positives on similar regions like hands and underarms.

## Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for full video processing. Single-frame mode does not require it.

## Running the Tool

```bash
# Full video
./run.sh douga/input.MP4 --block-size 15 --confidence 0.25

# Single frame (faster, no ffmpeg needed — use this for iterative debugging)
./run.sh douga/input.MP4 --frame 42 --debug
```

## Architecture

The entire implementation is in `mosaic_censor.py` (single file). Data flow:

1. **Pose detection** (`get_crotch_boxes`): YOLO11L-Pose extracts left/right hip keypoints (COCO-17 indices 11, 12) and derives crotch search boxes with padding.
2. **NSFW detection** (`detect_genitalia_multicrop`, `detect_yolo_genitalia_multicrop`): NudeNet and/or YOLO NSFW model runs on the full frame.
3. **Spatial filtering** (`filter_by_crotch`): Detected boxes are kept only if they overlap with crotch boxes, eliminating false positives.
4. **Deduplication** (`merge_adopted_boxes`): IoU-based merging across multiple detector outputs.
5. **Mosaic rendering** (`apply_mosaic`): OpenCV pixelation applied to adopted boxes.
6. **Frame interpolation**: Undetected frames between detections are interpolated (configurable gap limit via `--interpolate-gap`).
7. **Output**: Video frames written via `cv2.VideoWriter`, then audio merged via FFmpeg subprocess.

## Key CLI Options

| Option | Default | Notes |
|--------|---------|-------|
| `--frame N` | — | Single-frame mode; outputs source/censored/debug JPGs |
| `--debug` | off | Overlay skeleton, keypoints, and detection boxes |
| `--confidence F` | 0.03 | NudeNet confidence threshold |
| `--block-size N` | 15 | Mosaic pixel block size |
| `--detect-every N` | 1 | Run detection every N frames |
| `--interpolate-gap N` | 1 | Max consecutive undetected frames to interpolate |
| `--yolo-nsfw-model PATH` | — | Optional additional YOLO NSFW model (.pt) |
| `--frames START-END` | all | Process only a frame range |

## Testing

No automated test suite. Before submitting changes:

```bash
# Always validate with a single-frame debug pass first
./run.sh douga/input.MP4 --frame 48 --debug

# For video-path changes, also run a short full-video check
./run.sh douga/input.MP4 --frames 100-200
```

Pure helpers (`apply_mosaic`, bounding-box calculations) are good candidates for unit tests if a test suite is added.

## Coding Conventions

- 4-space indentation, `snake_case` for functions/variables, uppercase constants (e.g., `CENSOR_LABELS`)
- Use `Path` for path handling, type hints where they aid clarity
- Keep user-facing CLI help and log messages consistent with existing Japanese text
- Commits: concise imperative messages scoped to one behavior change (e.g., `Tune crotch box padding`)
- PRs for detection/rendering changes should include before/after debug images

## Model Files

- `yolo11l-pose.pt` — auto-downloads from Ultralytics on first run (note: AGENTS.md references `yolov8n-pose.pt` but the implementation uses `yolo11l-pose.pt`)
- `640m.onnx` — NudeNet model, must be in the script directory
- YOLO NSFW model (e.g., erax, Felldude, Throaway variants) — optional, passed via `--yolo-nsfw-model`; confidence default is auto-detected by filename (EraX: 0.3, others: 0.03)

## Media Artifacts

Sample/working media lives in `douga/`. Generated videos, debug frames, and logs are local artifacts — do not commit them.
