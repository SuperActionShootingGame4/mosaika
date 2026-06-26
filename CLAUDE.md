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

- **`mosaic_censor.py`** — メイン処理（CLI・検出・モザイク・CSV出力・post処理すべて）。GUI からはライブラリとしても呼ばれる
- **`pre_csv_editor.py`** — PyQt6 GUI。「レシピ生成」（検出を走らせて `_pre.csv` を作る）と CSV の目視編集の両方を担う大規模アプリ（4000行超）。`mosaic_censor` から `process_video` / `create_blank_pre_csv` 等を直接 import してワーカースレッドで実行する（subprocess ではない）
- **`app_version.py`** — `APP_VERSION`（GUI・ビルドで参照）
- **`ci_prepare_model.py`** — CI で `640m.onnx` を探す/ダウンロードするヘルパー（`nudenet` を import せず `pip show` で探索）
- **`AGENTS.md`** / **`README.md`** — それぞれ汎用エージェント向けガイドとユーザー向け README。CLI の全オプション・全ポーズモデルの詳細は README に網羅されている

GUI 設定は `config.toml` に永続化される（`.gitignore` 済み）。`[recipe_generation]` がレシピ生成ダイアログの前回値、`[editor]` の `last_recipe_path` が最後に開いた CSV。

## Running the Tool

```bash
# Single frame debug (no ffmpeg, fastest iteration)
./run.sh douga/input.MP4 --frame 42 --debug

# Full video
./run.sh douga/input.MP4 --intensity 15 --confidence 0.25

# pre/post workflow（CLI）
./run.sh douga/input.MP4 --pre --frames 6900-7100 --pose-model rtmpose --interpolate-gap 10
python pre_csv_editor.py douga/input_frames6900-7100_pre.csv
./run.sh --post douga/input_frames6900-7100_pre.csv

# GUI editor（レシピ生成・編集とも GUI 内で完結できる）
./run_edit.sh                 # venv 経由ラッパー
python pre_csv_editor.py      # 直接起動。引数なしはメニューからレシピ生成/CSV選択
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

### pre/post（レシピ）ワークフロー

`_pre.csv` は GUI 上では「レシピ」と呼ばれる。フローは2系統：

- **CLI**: `--pre` で通常検出 ＋ `_pre.mp4`（確認用）と `_pre.csv` を出力 → `pre_csv_editor.py` で編集 → `--post` で `_post.mp4` を清書生成
- **GUI 完結**: `pre_csv_editor.py` のレシピ生成ダイアログで動画とパラメータを指定し、ワーカースレッドで `process_video(..., csv_only=True)` を呼んで CSV を生成 → そのまま編集
- **空レシピ（blank recipe）**: `create_blank_pre_csv` で生成。股間ボックスは入るがモザイク枠は OFF の状態で、全モザイクを手動付与する用途。GUI の「空レシピを生成」チェックに対応
- `--post`: CSV のメタデータから `source_video`・`intensity`・`effect`・`shape` などを読み取り、ON の枠だけで `_post.mp4` を生成（CLI オプション不要、すべて CSV から読む）

GUI エディタの主な機能（README に詳細）: 元動画を見ながら枠を左ドラッグで移動/リサイズ・右ドラッグで新規作成、`mosaicプレビュー`、テンプレートマッチング＋オプティカルフローによる枠の自動追跡（`Trace`）、`effect`/`shape`/`intensity` のインライン編集。表示画像は `_pre.mp4` ではなく CSV の `source_video`（見つからなければ CSV と同階層の同名動画）から読む。

### CSV フォーマット

```
source_video,/abs/path/to/input.mp4   ← メタデータ行（key,value）
intensity,15
effect,mosaic
shape,square
confidence,0.03
pose_model,rtmpose
yolo_nsfw_model,
yolo_confidence,
detect_every,1
interpolate_gap,10
no_crotch,0
skip_no_person,0
frame_no,nsfw_detection_count,crotch_detected,comment,mosaic1_on,mosaic1_type,mosaic1_score,mosaic1_crotch_no,mosaic1_crotch_center,mosaic1_x1,mosaic1_y1,mosaic1_x2,mosaic1_y2,...
6900,,1,penis,1,penis,0.42,0,"640,360",100,200,300,400,...
```

各モザイク枠の suffix は `_on,_type,_score,_crotch_no,_crotch_center,_x1,_y1,_x2,_y2`（`MOSAIC_CSV_SUFFIXES`、最大 `MAX_CSV_MOSAICS` 枠）。`_score` は手動追加・座標編集した枠では空欄。メタデータ行は `row[0] == "frame_no"` になるまで続く。`read_csv_meta_and_rows()` で読み込む。ヘッダ生成は `mosaic_csv_header()`、メタ書き込みは `write_pre_csv_meta()`。

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
| `--skip-no-person` | off | 人物未検出フレームを早期スキップして高速化（`yolo11n.pt` を使う） |
| `--pose-model NAME` | `yolo11` | Pose backend |
| `--effect NAME` | `mosaic` | `mosaic` または `blur`（`CENSOR_EFFECTS`） |
| `--shape NAME` | `square` | `square` または `circle`（外接矩形を楕円でマスク、`CENSOR_SHAPES`） |
| `--confidence F` | 0.03 | NudeNet confidence threshold |
| `--intensity N` | 15 | Mosaic or blur intensity |
| `--detect-every N` | 1 | Run detection every N frames |
| `--no-interpolate` | off | フレーム間補間を無効化 |
| `--interpolate-gap N` | 10 | Max consecutive undetected frames to interpolate |
| `--yolo-nsfw-model PATH` | — | Additional YOLO NSFW model (.pt) |
| `--yolo-confidence F` | 自動 | YOLO NSFW 閾値（EraX:0.3 / その他:0.03 を自動設定） |

`--effect`/`--shape` の CLI デフォルトは `None`（CSV/内部デフォルトに委譲）。`choices` は `CENSOR_EFFECTS`・`CENSOR_SHAPES` で定義。

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
- `yolo11n.pt` — `--skip-no-person` の人物検出に使用（初回自動ダウンロード）
- YOLO NSFW `.pt` — optional, via `--yolo-nsfw-model`; EraX は信頼度閾値 0.3 自動設定

すべての `*.pt` / `*.onnx` は `.gitignore` 済み（コミットしない）。

## Build / Distribution

PyInstaller で単一実行ファイルを配布する。

- `./build_linux_editor.sh` → `dist/mosaika-pre-csv-editor`（GUI エディタ、Linux）
- `build_windows.bat` → Windows ビルド
- `mosaika_cli.spec`（CLI）/ `mosaika_editor.spec`（GUI）— PyInstaller spec
- `build/` と `dist/` は `.gitignore` 済み。バージョンは `app_version.py` の `APP_VERSION`

## Media Artifacts

`douga/` に作業用動画・生成ファイルを置く。生成物（動画・JPG・ログ・CSV）と `config.toml` はコミットしない（`.gitignore` 済み）。
