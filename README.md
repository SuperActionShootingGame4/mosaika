# mosaika

MP4動画の性器にモザイクまたはぼかしをかけるツール。

ポーズ検出で腰骨から股間エリアを特定し、NudeNet（および任意追加の YOLO NSFWモデル）で検出した矩形のうち股間エリアと重なるものだけを採用してモザイクまたはぼかしをかける二段フィルタ方式。手や脇など股間以外の誤検出を抑制する。`--no-crotch` を指定すると、股間領域との重なりチェックを無効化できる。

## 仕組み

1. **Pose 検出** — 選択したポーズモデルで骨格を推定し、左右腰骨（COCO-17 キーポイント 11・12）から股間エリアを算出
2. **NSFW 検出** — NudeNet（`640m.onnx`）でフル画面を検出。オプションで追加 YOLO NSFWモデルも併用可能
3. **フィルタ** — 股間エリアと重なる検出矩形のみ採用し、選択したエフェクトを適用
4. **補間** — 前後フレームで検出があれば間の未検出フレームに座標を**線形補間**してエフェクトを適用

## 必要なもの

### システム
- Python 3.10+
- ffmpeg（`PATH` が通っていること）

### モデルファイル（スクリプトと同じディレクトリに置く）

| ファイル | 用途 | 必須 |
|---|---|---|
| `640m.onnx` | NudeNet 本体 | ○ |
| `yolo11l-pose.pt` | 骨格推定・yolo11（起動時に自動ダウンロード） | ○ |
| `erax-anti-nsfw-yolo11m-v1.1.pt` 等 | 追加 YOLO NSFWモデル | 任意 |

### Python パッケージ

```
pip install -r requirements.txt
```

または venv 環境を使う場合：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

ポーズモデルによって追加パッケージが必要：

| ポーズモデル | 追加インストール |
|---|---|
| `vitpose-h` | `pip install transformers Pillow` |
| `rtmpose` / `rtmpose-wholebody` | `pip install rtmlib` |

## 使い方

### 動画全体を処理（基本）

```bash
python mosaic_censor.py input.mp4
```

出力: `input_NudeNet_censored.mp4`（元ファイルと同じディレクトリ）

### venv 経由で実行

```bash
./run.sh input.mp4
```

### 主なオプション

```bash
# ガウシアンぼかしを使う
python mosaic_censor.py input.mp4 --effect blur

# モザイクまたはぼかしの強度を変える（デフォルト: 15）
python mosaic_censor.py input.mp4 --intensity 20

# NudeNet の信頼度閾値を変える（デフォルト: 0.03）
python mosaic_censor.py input.mp4 --confidence 0.25

# 股間領域との重なりチェックを外し、しきい値以上の性器検出をすべて採用する
# 処理漏れを減らせるが、誤検出にもエフェクトが適用されやすくなる
python mosaic_censor.py input.mp4 --no-crotch

# 追加の YOLO NSFWモデルを使う
python mosaic_censor.py input.mp4 --yolo-nsfw-model erax-anti-nsfw-yolo11m-v1.1.pt

# 処理するフレーム範囲を絞る
python mosaic_censor.py input.mp4 --frames 100-500

# 1フレームだけ JPG で確認する
python mosaic_censor.py input.mp4 --frame 121

# 目視確認用 pre 動画と編集用CSVを作る
python mosaic_censor.py input.mp4 --pre --frames 6900-7100 --pose-model rtmpose --interpolate-gap 10

# CSVだけを渡して、元動画から清書 post 動画を作る
python mosaic_censor.py input_frames6900-7100_pre.csv --post

# _pre.csv をGUIで編集する
python pre_csv_editor.py input_frames6900-7100_pre.csv

# デバッグ動画を出力（骨格・検出ボックス表示）
python mosaic_censor.py input.mp4 --debug

# N フレームごとに検出（高速化、品質低下あり）
python mosaic_censor.py input.mp4 --detect-every 3

# ポーズ検出モデルを変える
python mosaic_censor.py input.mp4 --pose-model rtmpose
python mosaic_censor.py input.mp4 --pose-model rtmpose-wholebody
python mosaic_censor.py input.mp4 --pose-model vitpose-h
```

CSV エディタでは、`←` キーで1つ前のフレーム、`→` キーで次のフレームに移動できる。CSV行マトリクスの選択行は、`↑` または `PgUp` キーで1行前、`↓` または `PgDn` キーで1行後へ移動できる。
「CSV行」には `frame_no`、モザイク数/しきい値以上の NSFW 検出数、クロッチ検出あり/なし、`comment` を表示する。モザイクがかかっている行はピンク色、かかっていない行は灰色で表示する。
`現在フレームを元に戻す` ボタンを押すと、表示中フレームのモザイク枠と `comment` を CSV エディタ起動時の状態へ戻す。
モザイク詳細マトリクスの `score` 列には、検出時のスコアを表示する。手動追加や座標編集を行った枠では空欄になる。
画像上の既存枠は左ドラッグで移動・リサイズする。新規作成または選択中枠の作り直しは、画像上を右ドラッグして範囲を指定する。枠外の左クリックでは新しい枠を作成しない。
`mosaicプレビュー` を ON にすると、有効な枠へ CSV の `effect` と `intensity` で指定されたエフェクトを適用して表示する。
CSV エディタの表示画像は、pre 動画ではなく CSV の `source_video` に記録された元動画から読み込む。
Linux 向け単一実行ファイルは `./build_linux_editor.sh` で `dist/mosaika-pre-csv-editor` に生成する。配布先では `./mosaika-pre-csv-editor input_pre.csv` で起動する。CSV の `source_video` に記録されたパスが見つからない場合は、CSV と同じフォルダにある同名動画を読み込む。
モザイク詳細マトリクスで枠を選択し、`Del` キーを押すと選択中の枠を OFF にできる。
CSV行マトリクスの上には `intensity`、`effect`、`confidence`、`pose_model`、`yolo_nsfw_model`、`interpolate_gap`、`no_crotch` を表示する。`effect` と `intensity` は編集でき、`mosaicプレビュー` が ON の場合は表示へ即時反映する。

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--pre` | 無効 | 目視確認用の `_pre.mp4` と編集用CSVを作成 |
| `--post` | 無効 | CSVの `source_video` を読み、ONの座標だけで `_post.mp4` を作成 |
| `--effect NAME` | `mosaic` | エフェクト（`mosaic` または `blur`） |
| `--intensity N` | 15 | モザイクまたはぼかしの強度 |
| `--confidence F` | 0.03 | NudeNet の信頼度閾値 |
| `--detect-every N` | 1 | N フレームごとに検出（1 = 全フレーム） |
| `--debug` | 無効 | デバッグ動画を追加出力 |
| `--no-crotch` | 無効 | 股間領域との重なりチェックを無効化し、しきい値以上の性器検出をすべて採用 |
| `--no-interpolate` | 無効 | フレーム間補間を無効化 |
| `--interpolate-gap N` | 10 | 補間する最大連続未検出フレーム数 |
| `--yolo-nsfw-model PATH` | 無効 | 追加 YOLO NSFWモデルのパス |
| `--yolo-confidence F` | 自動 | YOLO NSFW の信頼度閾値（EraX: 0.3、その他: 0.03） |
| `--frames START-END` | 全体 | 処理フレーム範囲（例: `100-500`） |
| `--frame N` | — | 単フレームを JPG で出力（確認用） |
| `--pose-model NAME` | `yolo11` | ポーズ検出バックエンド（下表参照） |

### ポーズモデル一覧

| `--pose-model` | モデル | 特徴 |
|---|---|---|
| `yolo11`（デフォルト） | YOLO11l-pose | 速度・精度バランス良好 |
| `yolo8` | YOLOv8n-pose | 軽量・高速 |
| `rtmpose` | RTMPose-m | CPU 高速・ONNX 最適化済み |
| `rtmpose-wholebody` | RTMPose Wholebody-l | 133点（顔・手・足含む）、高精度 |
| `vitpose-h` | ViTPose-base | 高精度（要 `transformers`） |

## 出力ファイル

ファイル名には使用した NudeNet・YOLO NSFWモデルとポーズモデルが含まれる（例: `--pose-model rtmpose` の場合）。

動画処理時（例: `input.mp4`）:

| ファイル | 内容 |
|---|---|
| `input_NudeNet_yolo11_censored.mp4` | エフェクト適用済み動画 |
| `input_NudeNet_yolo11_log.txt` | 検出ログ |
| `input_NudeNet_yolo11_debug.mp4` | デバッグ動画（`--debug` 指定時） |

単フレーム確認時（`--frame 121`）:

| ファイル | 内容 |
|---|---|
| `input_NudeNet_yolo11_frame121_source.jpg` | 元フレーム |
| `input_NudeNet_yolo11_frame121_censored.jpg` | エフェクト適用済みフレーム |
| `input_NudeNet_yolo11_frame121_debug.jpg` | デバッグ画像（`--debug` 指定時） |
| `input_NudeNet_yolo11_frame121_log.txt` | 検出ログ |

## 対応 YOLO NSFWモデル

`--yolo-nsfw-model` で指定できるモデル。ファイル名でモデル名を自動判定する。

| ファイル名パターン | 表示名 |
|---|---|
| `erax*` | erax |
| `felldude*` / `*yolo_nsfw_n*` | Felldude |
| `throaway*` / `*penis_detection*` | Throaway |
| その他 | YOLO |
