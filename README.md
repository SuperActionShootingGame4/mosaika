# mozaika

MP4動画の性器にモザイクをかけるツール。

YOLOv11 Pose で腰骨から股間エリアを特定し、NudeNet（および任意追加の YOLO NSFWモデル）で検出した矩形のうち股間エリアと重なるものだけを採用してモザイクをかける二段フィルタ方式。手や脇など股間以外の誤検出を抑制する。

## 仕組み

1. **Pose 検出** — `yolo11l-pose.pt` で骨格を推定し、左右腰骨（キーポイント 11・12）から股間エリアを算出
2. **NSFW 検出** — NudeNet（`640m.onnx`）でフル画面を検出。オプションで追加 YOLO NSFWモデルも併用可能
3. **フィルタ** — 股間エリアと重なる検出矩形のみ採用し、モザイクを適用
4. **補間** — 前後フレームで検出があれば間の未検出フレームにも自動でモザイクを補間

## 必要なもの

### システム
- Python 3.10+
- ffmpeg（`PATH` が通っていること）

### モデルファイル（スクリプトと同じディレクトリに置く）

| ファイル | 用途 | 必須 |
|---|---|---|
| `640m.onnx` | NudeNet 本体 | ○ |
| `yolo11l-pose.pt` | 骨格推定（起動時に自動ダウンロード） | ○ |
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
# モザイクのブロックサイズを変える（デフォルト: 15）
python mosaic_censor.py input.mp4 --block-size 20

# NudeNet の信頼度閾値を変える（デフォルト: 0.03）
python mosaic_censor.py input.mp4 --confidence 0.25

# 追加の YOLO NSFWモデルを使う
python mosaic_censor.py input.mp4 --yolo-nsfw-model erax-anti-nsfw-yolo11m-v1.1.pt

# 処理するフレーム範囲を絞る
python mosaic_censor.py input.mp4 --frames 100-500

# 1フレームだけ JPG で確認する
python mosaic_censor.py input.mp4 --frame 121

# デバッグ動画を出力（骨格・検出ボックス表示）
python mosaic_censor.py input.mp4 --debug

# N フレームごとに検出（高速化、品質低下あり）
python mosaic_censor.py input.mp4 --detect-every 3
```

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--block-size N` | 15 | モザイクのブロックサイズ |
| `--confidence F` | 0.03 | NudeNet の信頼度閾値 |
| `--detect-every N` | 1 | N フレームごとに検出（1 = 全フレーム） |
| `--debug` | 無効 | デバッグ動画を追加出力 |
| `--no-interpolate` | 無効 | フレーム間補間を無効化 |
| `--interpolate-gap N` | 1 | 補間する最大連続未検出フレーム数 |
| `--yolo-nsfw-model PATH` | 無効 | 追加 YOLO NSFWモデルのパス |
| `--yolo-confidence F` | 自動 | YOLO NSFW の信頼度閾値（EraX: 0.3、その他: 0.03） |
| `--frames START-END` | 全体 | 処理フレーム範囲（例: `100-500`） |
| `--frame N` | — | 単フレームを JPG で出力（確認用） |

## 出力ファイル

動画処理時（例: `input.mp4`）:

| ファイル | 内容 |
|---|---|
| `input_NudeNet_censored.mp4` | モザイク済み動画 |
| `input_NudeNet_log.txt` | 検出ログ |
| `input_NudeNet_debug.mp4` | デバッグ動画（`--debug` 指定時） |

単フレーム確認時（`--frame 121`）:

| ファイル | 内容 |
|---|---|
| `input_NudeNet_frame121_source.jpg` | 元フレーム |
| `input_NudeNet_frame121_censored.jpg` | モザイク済みフレーム |
| `input_NudeNet_frame121_debug.jpg` | デバッグ画像（`--debug` 指定時） |
| `input_NudeNet_frame121_log.txt` | 検出ログ |

## 対応 YOLO NSFWモデル

`--yolo-nsfw-model` で指定できるモデル。ファイル名でモデル名を自動判定する。

| ファイル名パターン | 表示名 |
|---|---|
| `erax*` | erax |
| `felldude*` / `*yolo_nsfw_n*` | Felldude |
| `throaway*` / `*penis_detection*` | Throaway |
| その他 | YOLO |
