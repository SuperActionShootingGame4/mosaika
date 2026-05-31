# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python video-processing tool. The main implementation is `mosaic_censor.py`, which provides the CLI, pose detection, NudeNet filtering, mosaic rendering, logging, and output handling. `run.sh` executes the script with the local `.venv` interpreter. Runtime dependencies are in `requirements.txt`.

Sample and working media live in `douga/`. Keep large input videos, generated frame JPGs, debug images, and `.log` files there or in another ignored workspace. The pose model weight `yolov8n-pose.pt` is stored at the repository root.

## Build, Test, and Development Commands

Create or refresh the local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the tool through the wrapper:

```bash
./run.sh douga/input.MP4 --intensity 15 --confidence 0.25
```

Process one frame for faster debugging:

```bash
./run.sh douga/input.MP4 --frame 42 --debug
```

Video processing requires `ffmpeg` on `PATH`. Single-frame mode does not require `ffmpeg`.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where they improve clarity, and small functions that separate model setup, detection, drawing, and file output. Existing code uses `snake_case` for functions and variables, uppercase constants such as `CENSOR_LABELS`, and `Path` for path handling.

Keep user-facing CLI help and logs consistent with the current Japanese text. Keep edits focused and easy to validate on a single frame.

## Testing Guidelines

There is currently no automated test suite. Before submitting changes, run at least one single-frame debug pass and inspect the generated JPG/log:

```bash
./run.sh douga/DJI_20250901130224_0024_D.MP4 --frame 48 --debug
```

For video-path changes, also run a short full-video check. Prefer future tests around pure helpers such as `apply_mosaic()` and bounding-box calculations before model-dependent behavior.

## Commit & Pull Request Guidelines

No usable Git history is available in this checkout, so use concise imperative commit messages such as `Tune crotch box padding` or `Add frame debug output`. Keep commits scoped to one behavior change.

Pull requests should include a short description, commands run, affected outputs, and before/after debug images when detection or rendering changes. Mention dependency, model, or `ffmpeg` assumptions explicitly.

## Security & Configuration Tips

Do not commit private, sensitive, or unnecessary media. Treat generated videos, debug frames, and logs as local artifacts unless they are required for review.
