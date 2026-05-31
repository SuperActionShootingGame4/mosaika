# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mosaika (モザイカ) is a Python CLI tool that applies pixelation/mosaic censoring to MP4 videos using a two-stage detection pipeline: pose estimation to locate crotch regions, then NudeNet (and optionally YOLO NSFW models) to detect genitalia within those regions. This spatial filtering approach reduces false positives on similar regions like hands and underarms.

## Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for full video processing. Single-frame mode does not require it.

Pose model による追加依存：

| `--pose-model` | 追加インストール |
|---|---|
| `vitpose-h` | `pip install transformers Pillow` |
| `rtmpose` / `rtmpose-wholebody` | `pip install rtmlib` |

## Source Files

- **`mosaic_censor.py`** — メイン処理（CLI・検出・モザイク・CSV出力・post処理すべて）
- **`pre_csv_editor.py`** — `_pre.csv` を目視編集する PyQt6 GUI

## Running the Tool

```bash
# Single frame debug (no ffmpeg, fastest iteration)
./run.sh douga/input.MP4 --frame 42 --debug

# Full video
./run.sh douga/input.MP4 --intensity 15 --confidence 0.25

# pre/post workflow
./run.sh douga/input.MP4 --pre --frames 6900-7100 --pose-model rtmpose --interpolate-gap 10
python pre_csv_editor.py douga/input_frames6900-7100_pre.csv
./run.sh --post douga/input_frames6900-7100_pre.csv

# GUI editor standalone
python pre_csv_editor.py  # ファイル選択ダイアログが開く
```

## Architecture

### mosaic_censor.py — 処理フロー

1. **Pose detection** (`get_crotch_boxes`): 選択バックエンドで骨格推定し、左右腰骨（COCO-17 indices 11, 12）から股間検索ボックスを算出。バックエンドは `load_pose_model(backend)` が `("yolo"|"rtmpose"|"vitpose", *models)` タプルを返し、`get_crotch_boxes` がディスパッチ。
2. **NSFW detection** (`detect_genitalia_multicrop`, `detect_yolo_genitalia_multicrop`): NudeNet + オプションの YOLO NSFW モデルでフル画面を検出。
3. **Spatial filtering** (`filter_by_crotch`): 通常は股間ボックスと重なる検出矩形のみ採用。`--no-crotch` 指定時はしきい値以上の検出矩形をすべて採用。
4. **Deduplication** (`merge_adopted_boxes`): IoU ベースで複数検出器の重複を除去。
5. **Mosaic rendering** (`apply_mosaic`): OpenCV ピクセル化。
6. **Frame interpolation** (`interpolate_adopted_boxes`): 前後検出フレーム間を座標線形補間してモザイク適用。
7. **Output**: `cv2.VideoWriter` → FFmpeg で音声マージ。`--pre` 時は CSV も同時出力。

### pre/post ワークフロー

- `--pre`: 通常の検出処理 ＋ `_pre.mp4`（モザイク済み確認用）と `_pre.csv`（フレームごとのモザイク座標）を出力
- `pre_csv_editor.py`: CSV の `source_video` にある元動画を横に映しながら CSV のモザイク矩形を GUI で追加・修正
- `--post`: CSV のメタデータから `source_video`・`intensity` などを読み取り、`_post.mp4` を生成

### CSV フォーマット

```
source_video,/abs/path/to/input.mp4   ← メタデータ行（key,value）
intensity,15
confidence,0.03
pose_model,rtmpose
...
frame_no,nsfw_detection_count,crotch_detected,comment,mosaic1_on,mosaic1_type,mosaic1_score,mosaic1_x1,mosaic1_y1,mosaic1_x2,mosaic1_y2,...
6900,,1,penis,100,200,300,400,...
```

メタデータ行は `row[0] == "frame_no"` になるまで続く。`read_csv_meta_and_rows()` で読み込む。

### 出力ファイル命名

ファイル名にポーズモデル名が含まれる（例: `input_NudeNet_rtmpose_frame42_debug.jpg`）。`detector_name_suffix(yolo_nsfw_model_path, pose_backend)` が生成。

### ポーズバックエンド

| `--pose-model` | 実装関数 | 備考 |
|---|---|---|
| `yolo11`（デフォルト） | `_get_crotch_boxes_yolo` | yolo11l-pose.pt |
| `yolo8` | `_get_crotch_boxes_yolo` | yolov8n-pose.pt |
| `rtmpose` | `_get_crotch_boxes_rtmpose` | rtmlib Body |
| `rtmpose-wholebody` | `_get_crotch_boxes_rtmpose` | rtmlib Wholebody（133点） |
| `vitpose-h` | `_get_crotch_boxes_vitpose` | nielsr/vitpose-base-simple |

すべて `list[np.ndarray]`（各人物 `[17, 3]` の x/y/conf）を返す共通フォーマット。

## Key CLI Options

| Option | Default | Notes |
|--------|---------|-------|
| `--frame N` | — | Single-frame mode; outputs source/censored/debug JPGs |
| `--frames START-END` | all | Process only a frame range |
| `--pre` | off | `_pre.mp4` + `_pre.csv` を出力 |
| `--post` | off | CSV を渡して `_post.mp4` を生成（オプション不要、CSV から読む） |
| `--debug` | off | Overlay skeleton, keypoints, and detection boxes |
| `--no-crotch` | off | Apply censoring to all detections above threshold without crotch overlap filtering |
| `--pose-model NAME` | `yolo11` | Pose backend |
| `--confidence F` | 0.03 | NudeNet confidence threshold |
| `--intensity N` | 15 | Mosaic or blur intensity |
| `--detect-every N` | 1 | Run detection every N frames |
| `--interpolate-gap N` | 10 | Max consecutive undetected frames to interpolate |
| `--yolo-nsfw-model PATH` | — | Additional YOLO NSFW model (.pt) |

## Testing

No automated test suite.

```bash
# 常にまずシングルフレームで検証
./run.sh douga/input.MP4 --frame 48 --debug

# 動画パス変更時は短い範囲でも確認
./run.sh douga/input.MP4 --frames 100-200
```

## Coding Conventions

- 4-space indentation, `snake_case`, uppercase constants (e.g., `CENSOR_LABELS`)
- `Path` for path handling, type hints where they aid clarity
- CLI ヘルプ・ログは日本語で統一
- Commits: imperative, scoped to one behavior (e.g., `Tune crotch box padding`)
- 検出・描画変更の PR にはデバッグ画像の before/after を添付

## Model Files

- `640m.onnx` — NudeNet model, must be in the script directory（GitHub: notAI-tech/NudeNet releases v3.4-weights）
- `yolo11l-pose.pt` — auto-downloads from Ultralytics on first run
- YOLO NSFW `.pt` — optional, via `--yolo-nsfw-model`; EraX は信頼度閾値 0.3 自動設定

## Media Artifacts

`douga/` に作業用動画・生成ファイルを置く。生成物（動画・JPG・ログ・CSV）はコミットしない。
