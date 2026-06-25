#!/usr/bin/env python3
"""_pre.csv editor for mosaic rectangles."""

from __future__ import annotations

import argparse
import bisect
import csv
import faulthandler
import logging
import platform
import tempfile
import threading
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, Qt, QThread, QTimer, pyqtSignal, qInstallMessageHandler
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_version import APP_VERSION
from mosaic_censor import (
    CENSOR_EFFECTS,
    POSE_BACKENDS,
    SKELETON_EDGES,
    effective_yolo_confidence,
    frame_in_ranges,
    get_crotch_boxes,
    load_pose_model,
    normalize_frame_ranges,
    parse_frame_ranges,
    post_output_path_from_csv,
    process_post_from_csv,
    process_video,
)

MAX_MOSAICS = 255
DEFAULT_VISIBLE_MOSAICS = 5
HANDLE_SIZE = 8
DEFAULT_INTENSITY_SLIDER_MAX = 100
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15
TRACE_TO_END_MIN_SCORE = 0.35
POSE_OVERLAY_MIN_SCORE = 0.3

APP_LOGGER = logging.getLogger("pre_csv_editor")
APP_LOG_FILE_HANDLE = None


def _get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _get_log_dir() -> Path:
    return _get_app_base_dir() / "logs"


def setup_app_logging() -> Path:
    global APP_LOG_FILE_HANDLE
    log_path = None
    log_error: Exception | None = None
    for log_dir in (
        _get_log_dir(),
        Path.home() / ".mosaika" / "logs",
        Path(tempfile.gettempdir()) / "mosaika_logs",
    ):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            candidate = log_dir / f"pre_csv_editor_{datetime.now().strftime('%Y%m%d')}.log"
            with open(candidate, "a", encoding="utf-8"):
                pass
            log_path = candidate
            break
        except OSError as exc:
            log_error = exc
    if log_path is None:
        raise RuntimeError(f"ログファイルを作成できません: {log_error}")

    APP_LOGGER.setLevel(logging.INFO)
    APP_LOGGER.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    APP_LOGGER.addHandler(handler)
    APP_LOGGER.propagate = False

    APP_LOG_FILE_HANDLE = open(log_path, "a", encoding="utf-8")
    faulthandler.enable(file=APP_LOG_FILE_HANDLE, all_threads=True)

    APP_LOGGER.info("アプリ起動: argv=%s", sys.argv)
    APP_LOGGER.info(
        "環境情報: python=%s executable=%s platform=%s cwd=%s frozen=%s",
        sys.version.replace("\n", " "),
        sys.executable,
        platform.platform(),
        Path.cwd(),
        getattr(sys, "frozen", False),
    )
    return log_path


def log_user_action(action: str, **details) -> None:
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    APP_LOGGER.info("ユーザー操作: %s%s%s", action, " " if detail_text else "", detail_text)


def install_exception_logging() -> None:
    def excepthook(exc_type, exc_value, exc_traceback) -> None:
        APP_LOGGER.critical(
            "未捕捉例外でアプリが終了します",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def thread_excepthook(args) -> None:
        APP_LOGGER.critical(
            "スレッド未捕捉例外: thread=%s",
            getattr(args.thread, "name", None),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def qt_message_handler(mode, context, message) -> None:
        level = logging.WARNING
        mode_name = getattr(mode, "name", str(mode))
        if "Critical" in mode_name or "Fatal" in mode_name:
            level = logging.ERROR
        file_name = getattr(context, "file", "") or ""
        line = getattr(context, "line", 0) or 0
        APP_LOGGER.log(level, "Qtメッセージ: %s %s (%s:%s)", mode_name, message, file_name, line)

    sys.excepthook = excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = thread_excepthook
    qInstallMessageHandler(qt_message_handler)


class LoggingApplication(QApplication):
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            APP_LOGGER.critical(
                "Qtイベント処理中の未捕捉例外: receiver=%r event=%r",
                receiver,
                event.type() if event is not None else None,
                exc_info=True,
            )
            raise


def _get_config_path() -> Path:
    return _get_app_base_dir() / "config.toml"

CONFIG_PATH = _get_config_path()
RECIPE_CONFIG_SECTION = "recipe_generation"
DISPLAY_SEQUENTIAL_MAX_SKIP = 120


class RecipeGenerationCancelled(Exception):
    pass


def progress_text(current: int, total: int, elapsed: float) -> str:
    if current <= 0 or total <= 0:
        return "準備中..."
    remaining = max(0.0, elapsed * (total - current) / current)
    return f"{current}/{total} frame  残り {time.strftime('%H:%M:%S', time.gmtime(remaining))}"


def load_recipe_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        if tomllib is not None:
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
        else:
            data = load_recipe_config_fallback(CONFIG_PATH)
    except Exception:
        APP_LOGGER.warning("設定ファイルを読み込めません: %s", CONFIG_PATH, exc_info=True)
        return {}
    section = data.get(RECIPE_CONFIG_SECTION, {})
    return section if isinstance(section, dict) else {}


def load_recipe_config_fallback(path: Path) -> dict:
    data: dict[str, dict] = {}
    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if section is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        data[section][key] = parse_simple_toml_value(value)
    return data


def parse_simple_toml_value(value: str):
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_recipe_config(settings: dict) -> None:
    lines = [
        f"[{RECIPE_CONFIG_SECTION}]",
        f"video_path = {toml_value(settings['video_path'])}",
        f"pose_model = {toml_value(settings['pose_model'])}",
        f"all_frames = {toml_value(settings['all_frames'])}",
        f"start_frame = {toml_value(settings['start_frame'])}",
        f"end_frame = {toml_value(settings['end_frame'])}",
        f"confidence = {toml_value(settings['confidence'])}",
        f"intensity = {toml_value(settings['intensity'])}",
        f"effect = {toml_value(settings['effect'])}",
        f"detect_every = {toml_value(settings['detect_every'])}",
        f"interpolate_gap = {toml_value(settings['interpolate_gap'])}",
        f"yolo_nsfw_model = {toml_value(settings['yolo_nsfw_model'])}",
        f"yolo_confidence = {toml_value(settings['yolo_confidence'])}",
        f"no_crotch = {toml_value(settings['no_crotch'])}",
        f"skip_no_person = {toml_value(settings['skip_no_person'])}",
        "",
    ]
    CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    return value if isinstance(value, bool) else default


def config_int(config: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def config_float(config: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        return default
    return max(minimum, min(maximum, float(value)))


def config_text(config: dict, key: str, default: str) -> str:
    value = config.get(key, default)
    return value if isinstance(value, str) else default


class PreCreateWorker(QObject):
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: Path, csv_path: Path, log_path: Path, options: dict) -> None:
        super().__init__()
        self.video_path = video_path
        self.csv_path = csv_path
        self.log_path = log_path
        self.options = options
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def emit_progress(self, current: int, total: int, elapsed: float) -> None:
        if self.cancel_requested:
            raise RecipeGenerationCancelled()
        self.progress.emit(current, total, elapsed)

    def run(self) -> None:
        try:
            APP_LOGGER.info(
                "レシピ生成開始: video=%s csv=%s log=%s options=%s",
                self.video_path,
                self.csv_path,
                self.log_path,
                self.options,
            )
            with open(self.log_path, "w", encoding="utf-8") as lf:
                yolo_confidence = effective_yolo_confidence(
                    self.options["yolo_nsfw_model"],
                    self.options["yolo_confidence"],
                )
                process_video(
                    input_path=str(self.video_path),
                    output_path=None,
                    intensity=self.options["intensity"],
                    effect=self.options["effect"],
                    confidence=self.options["confidence"],
                    detect_every=self.options["detect_every"],
                    log_file=lf,
                    debug_path=None,
                    interpolate=True,
                    yolo_nsfw_model_path=self.options["yolo_nsfw_model"],
                    yolo_confidence=yolo_confidence,
                    max_interpolate_gap=self.options["interpolate_gap"],
                    frame_range=self.options["frame_range"],
                    pose_backend=self.options["pose_model"],
                    no_crotch=self.options["no_crotch"],
                    csv_path=str(self.csv_path),
                    render_debug_to_output=False,
                    csv_only=True,
                    progress_callback=self.emit_progress,
                    skip_no_person=self.options["skip_no_person"],
                )
            APP_LOGGER.info("レシピ生成完了: csv=%s", self.csv_path)
            self.finished.emit(str(self.csv_path))
        except RecipeGenerationCancelled:
            APP_LOGGER.info("レシピ生成キャンセル: csv=%s", self.csv_path)
            self.csv_path.unlink(missing_ok=True)
            self.failed.emit("レシピ生成をキャンセルしました。")
        except Exception as exc:
            APP_LOGGER.exception("レシピ生成エラー: video=%s csv=%s", self.video_path, self.csv_path)
            self.failed.emit(str(exc))


class PostWorker(QObject):
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, csv_path: Path, output_path: Path, log_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.output_path = output_path
        self.log_path = log_path

    def run(self) -> None:
        try:
            APP_LOGGER.info(
                "エンコード開始: csv=%s output=%s log=%s",
                self.csv_path,
                self.output_path,
                self.log_path,
            )
            with open(self.log_path, "w", encoding="utf-8") as lf:
                process_post_from_csv(
                    csv_path=str(self.csv_path),
                    output_path=str(self.output_path),
                    log_file=lf,
                    progress_callback=self.progress.emit,
                )
            APP_LOGGER.info("エンコード完了: output=%s", self.output_path)
            self.finished.emit(str(self.output_path))
        except Exception as exc:
            APP_LOGGER.exception("エンコードエラー: csv=%s output=%s", self.csv_path, self.output_path)
            self.failed.emit(str(exc))


class PreCreateDialog(QDialog):
    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.config = load_recipe_config()
        self.video_path = video_path
        self.result_csv: Path | None = None
        self.thread: QThread | None = None
        self.worker: PreCreateWorker | None = None
        self.worker_failed = False
        self.running = False
        self.setWindowTitle("レシピ生成ウィンドウ")
        self.resize(520, 360)
        self.total_frames = self.detect_total_frames()
        APP_LOGGER.info("レシピ生成ダイアログ表示: video=%s total_frames=%s", self.video_path, self.total_frames)
        self.build_ui()

    def detect_total_frames(self) -> int:
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            if not cap.isOpened():
                APP_LOGGER.warning("動画を開けないため総フレーム数を取得できません: %s", self.video_path)
                return 0
            return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.video_path_input = QLineEdit(str(self.video_path))
        self.browse_video_button = QPushButton("参照")
        self.browse_video_button.clicked.connect(self.select_video)
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_path_input, stretch=1)
        video_layout.addWidget(self.browse_video_button)

        self.pose_combo = QComboBox()
        self.pose_combo.addItems(list(POSE_BACKENDS))
        pose_model = config_text(self.config, "pose_model", "yolo11")
        self.pose_combo.setCurrentText(pose_model if pose_model in POSE_BACKENDS else "yolo11")
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(list(CENSOR_EFFECTS))
        effect = config_text(self.config, "effect", "mosaic")
        self.effect_combo.setCurrentText(effect if effect in CENSOR_EFFECTS else "mosaic")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(config_float(self.config, "confidence", 0.03, 0.0, 1.0))
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(1, 10000)
        self.intensity_spin.setValue(config_int(self.config, "intensity", 15, 1, 10000))
        self.detect_every_spin = QSpinBox()
        self.detect_every_spin.setRange(1, 10000)
        self.detect_every_spin.setValue(config_int(self.config, "detect_every", 1, 1, 10000))
        self.interpolate_gap_spin = QSpinBox()
        self.interpolate_gap_spin.setRange(0, 10000)
        self.interpolate_gap_spin.setValue(config_int(self.config, "interpolate_gap", 10, 0, 10000))
        self.no_crotch_check = QCheckBox("股間領域フィルタを無効化")
        self.no_crotch_check.setChecked(config_bool(self.config, "no_crotch", False))
        self.skip_no_person_check = QCheckBox("人物がいないフレームは検出をスキップ")
        self.skip_no_person_check.setChecked(config_bool(self.config, "skip_no_person", False))
        self.all_frames_check = QCheckBox("全フレーム")
        self.all_frames_check.setChecked(config_bool(self.config, "all_frames", True))
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        max_frame = max(0, self.total_frames - 1)
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(0, max_frame)
            spin.setEnabled(not self.all_frames_check.isChecked())
        self.start_spin.setValue(config_int(self.config, "start_frame", 0, 0, max_frame))
        self.end_spin.setValue(config_int(self.config, "end_frame", max_frame, 0, max_frame))
        self.yolo_model_input = QLineEdit()
        self.yolo_model_input.setText(config_text(self.config, "yolo_nsfw_model", ""))
        self.yolo_confidence_spin = QDoubleSpinBox()
        self.yolo_confidence_spin.setRange(0.0, 1.0)
        self.yolo_confidence_spin.setDecimals(3)
        self.yolo_confidence_spin.setSingleStep(0.01)
        self.yolo_confidence_spin.setSpecialValueText("自動")
        self.yolo_confidence_spin.setValue(config_float(self.config, "yolo_confidence", 0.0, 0.0, 1.0))
        browse_yolo = QPushButton("参照")
        browse_yolo.clicked.connect(self.select_yolo_model)
        yolo_layout = QHBoxLayout()
        yolo_layout.addWidget(self.yolo_model_input, stretch=1)
        yolo_layout.addWidget(browse_yolo)
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(self.start_spin)
        frame_layout.addWidget(QLabel("-"))
        frame_layout.addWidget(self.end_spin)

        form.addRow("動画ファイル", video_layout)
        form.addRow("pose_model", self.pose_combo)
        form.addRow("frames", frame_layout)
        form.addRow("", self.all_frames_check)
        form.addRow("confidence ※検出の信頼度しきい値", self.confidence_spin)
        form.addRow("intensity ※モザイク/ぼかしの強度", self.intensity_spin)
        form.addRow("effect", self.effect_combo)
        form.addRow("detect_every ※何フレームごとにAI検出するか", self.detect_every_spin)
        form.addRow("interpolate_gap ※検出漏れを補間する最大フレーム数", self.interpolate_gap_spin)
        form.addRow("yolo_nsfw_model ※未選択時はNudeNetを使用", yolo_layout)
        form.addRow("yolo_confidence", self.yolo_confidence_spin)
        form.addRow("", self.no_crotch_check)
        form.addRow("", self.skip_no_person_check)
        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("未実行")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        self.buttons = QDialogButtonBox()
        self.start_button = self.buttons.addButton("レシピ生成", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = self.buttons.addButton("キャンセル", QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(self.buttons)

        self.all_frames_check.stateChanged.connect(self.update_frame_enabled)
        self.video_path_input.editingFinished.connect(self.update_video_path_from_input)
        self.start_button.clicked.connect(self.start_pre_create)
        self.cancel_button.clicked.connect(self.cancel_or_reject)

    def update_frame_enabled(self) -> None:
        enabled = not self.all_frames_check.isChecked()
        self.start_spin.setEnabled(enabled)
        self.end_spin.setEnabled(enabled)

    def select_yolo_model(self) -> None:
        log_user_action("YOLO NSFWモデル参照")
        selected, _ = QFileDialog.getOpenFileName(self, "YOLO NSFWモデルを選択", "", "Model (*.pt);;All files (*)")
        if selected:
            self.yolo_model_input.setText(selected)
            log_user_action("YOLO NSFWモデル選択", path=selected)

    def select_video(self) -> None:
        log_user_action("動画ファイル参照")
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "動画ファイルを選択",
            str(self.video_path.parent),
            "Video (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All files (*)",
        )
        if selected:
            self.video_path_input.setText(selected)
            self.update_video_path_from_input()

    def update_video_path_from_input(self) -> None:
        video_path = Path(self.video_path_input.text().strip()).expanduser()
        if video_path == self.video_path:
            return True
        if not video_path.is_file():
            APP_LOGGER.warning("動画ファイルが見つかりません: %s", video_path)
            QMessageBox.warning(self, "Error", f"動画ファイルが見つかりません: {video_path}")
            self.video_path_input.setText(str(self.video_path))
            return False
        self.video_path = video_path
        self.total_frames = self.detect_total_frames()
        log_user_action("動画ファイル変更", path=video_path, total_frames=self.total_frames)
        max_frame = max(0, self.total_frames - 1)
        for spin in (self.start_spin, self.end_spin):
            spin.setMaximum(max_frame)
        self.start_spin.setValue(0)
        self.end_spin.setValue(max_frame)
        return True

    def output_paths(self) -> tuple[Path, Path]:
        range_suffix = ""
        frame_range = self.frame_range()
        if frame_range is not None:
            range_suffix = f"_frames{frame_range[0]}-{frame_range[1]}"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self.video_path.stem
        csv_path = self.video_path.with_name(f"{stem}{range_suffix}_pre_{stamp}.csv")
        log_path = self.video_path.with_name(f"{stem}{range_suffix}_pre_{stamp}_log.txt")
        return csv_path, log_path

    def frame_range(self) -> tuple[int, int] | None:
        if self.all_frames_check.isChecked():
            return None
        return self.start_spin.value(), self.end_spin.value()

    def current_settings(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "pose_model": self.pose_combo.currentText(),
            "all_frames": self.all_frames_check.isChecked(),
            "start_frame": self.start_spin.value(),
            "end_frame": self.end_spin.value(),
            "confidence": self.confidence_spin.value(),
            "intensity": self.intensity_spin.value(),
            "effect": self.effect_combo.currentText(),
            "detect_every": self.detect_every_spin.value(),
            "interpolate_gap": self.interpolate_gap_spin.value(),
            "yolo_nsfw_model": self.yolo_model_input.text().strip(),
            "yolo_confidence": self.yolo_confidence_spin.value(),
            "no_crotch": self.no_crotch_check.isChecked(),
            "skip_no_person": self.skip_no_person_check.isChecked(),
        }

    def start_pre_create(self) -> None:
        if not self.update_video_path_from_input():
            return
        frame_range = self.frame_range()
        if frame_range is not None and frame_range[1] < frame_range[0]:
            APP_LOGGER.warning("フレーム範囲が不正です: range=%s", frame_range)
            QMessageBox.warning(self, "Error", "フレーム範囲が不正です")
            return
        settings = self.current_settings()
        log_user_action("レシピ生成開始", **settings)
        try:
            save_recipe_config(settings)
        except Exception:
            APP_LOGGER.exception("設定ファイルを書き込めません: %s", CONFIG_PATH)
            QMessageBox.critical(self, "Error", f"設定ファイルを書き込めません: {CONFIG_PATH}")
            return
        csv_path, log_path = self.output_paths()
        yolo_model = self.yolo_model_input.text().strip() or None
        yolo_conf = None if self.yolo_confidence_spin.value() <= 0 else self.yolo_confidence_spin.value()
        options = {
            "pose_model": self.pose_combo.currentText(),
            "frame_range": frame_range,
            "confidence": self.confidence_spin.value(),
            "intensity": self.intensity_spin.value(),
            "effect": self.effect_combo.currentText(),
            "no_crotch": self.no_crotch_check.isChecked(),
            "detect_every": self.detect_every_spin.value(),
            "interpolate_gap": self.interpolate_gap_spin.value(),
            "yolo_nsfw_model": yolo_model,
            "yolo_confidence": yolo_conf,
            "skip_no_person": self.skip_no_person_check.isChecked(),
        }
        self.worker_failed = False
        self.result_csv = None
        self.set_running(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("AIレシピを作成中...")
        self.thread = QThread(self)
        self.worker = PreCreateWorker(self.video_path, csv_path, log_path, options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.pre_finished)
        self.worker.failed.connect(self.pre_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.pre_thread_finished)
        self.thread.start()

    def set_running(self, running: bool) -> None:
        self.running = running
        for widget in (
            self.pose_combo, self.effect_combo, self.confidence_spin, self.intensity_spin,
            self.detect_every_spin, self.interpolate_gap_spin, self.no_crotch_check,
            self.skip_no_person_check, self.all_frames_check, self.start_spin,
            self.end_spin, self.video_path_input, self.browse_video_button,
            self.yolo_model_input, self.yolo_confidence_spin, self.start_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("キャンセル" if running else "キャンセル")
        if not running:
            self.update_frame_enabled()

    def update_progress(self, current: int, total: int, elapsed: float) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        self.progress_label.setText(progress_text(current, total, elapsed))

    def pre_finished(self, csv_path: str) -> None:
        self.result_csv = Path(csv_path)
        self.progress_label.setText(f"完了: {csv_path}")
        log_user_action("レシピ生成完了", csv_path=csv_path)

    def pre_failed(self, message: str) -> None:
        self.worker_failed = True
        self.set_running(False)
        if "キャンセル" in message:
            APP_LOGGER.info("レシピ生成キャンセル表示: %s", message)
            QMessageBox.information(self, "キャンセル", message)
            self.progress_label.setText("キャンセル")
        else:
            APP_LOGGER.error("レシピ生成失敗表示: %s", message)
            QMessageBox.critical(self, "Error", message)
            self.progress_label.setText("失敗")

    def pre_thread_finished(self) -> None:
        self.running = False
        if self.result_csv is not None and not self.worker_failed:
            self.accept()

    def cancel_or_reject(self) -> None:
        if self.running:
            log_user_action("レシピ生成キャンセル要求")
            if self.worker:
                self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("キャンセル中...")
            return
        log_user_action("レシピ生成ダイアログを閉じる")
        self.reject()

    def reject(self) -> None:
        if self.running:
            self.cancel_or_reject()
            return
        super().reject()


class PostProgressDialog(QDialog):
    def __init__(self, csv_path: Path, output_path: Path, log_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.csv_path = csv_path
        self.output_path = output_path
        self.log_path = log_path
        self.thread: QThread | None = None
        self.worker: PostWorker | None = None
        self.finished_output: str | None = None
        self.failed_message: str | None = None
        self.running = False
        self.setWindowTitle("エンコード")
        self.resize(460, 140)
        layout = QVBoxLayout(self)
        self.label = QLabel("エンコード準備中...")
        self.progress_bar = QProgressBar()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

    def start(self) -> None:
        self.running = True
        log_user_action("エンコードダイアログ開始", csv_path=self.csv_path, output_path=self.output_path)
        self.thread = QThread(self)
        self.worker = PostWorker(self.csv_path, self.output_path, self.log_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.post_finished)
        self.worker.failed.connect(self.post_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.post_thread_finished)
        self.thread.start()

    def update_progress(self, current: int, total: int, elapsed: float) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        self.label.setText(progress_text(current, total, elapsed))

    def post_finished(self, output_path: str) -> None:
        self.finished_output = output_path

    def post_failed(self, message: str) -> None:
        self.failed_message = message

    def post_thread_finished(self) -> None:
        self.running = False
        if self.failed_message:
            APP_LOGGER.error("エンコード失敗表示: %s", self.failed_message)
            QMessageBox.critical(self, "Error", self.failed_message)
            self.reject()
            return
        if self.finished_output:
            log_user_action("エンコード完了", output_path=self.finished_output)
            QMessageBox.information(self, "完了", f"エンコードが完了しました。\n{self.finished_output}")
            self.accept()

    def reject(self) -> None:
        if self.running:
            APP_LOGGER.warning("エンコード中に閉じる操作が行われました")
            QMessageBox.information(self, "処理中", "エンコード中は閉じられません。")
            return
        super().reject()


@dataclass
class CsvData:
    meta: list[list[str]]
    fieldnames: list[str]
    rows: list[dict[str, str]]

    @property
    def meta_dict(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self.meta if len(row) >= 2}


def read_pre_csv(path: Path) -> CsvData:
    meta: list[list[str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "frame_no":
                return CsvData(meta=meta, fieldnames=row, rows=list(csv.DictReader(f, fieldnames=row)))
            meta.append(row)
    raise RuntimeError("frame_no ヘッダ行が見つかりません")


def write_pre_csv(path: Path, data: CsvData) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data.meta)
        dict_writer = csv.DictWriter(f, fieldnames=data.fieldnames, extrasaction="ignore")
        dict_writer.writeheader()
        for row in data.rows:
            dict_writer.writerow(row)


def set_meta_value(data: CsvData, key: str, value: str) -> None:
    for row in data.meta:
        if row and row[0] == key:
            if len(row) >= 2:
                row[1] = value
            else:
                row.append(value)
            return
    data.meta.append([key, value])


def source_video_path(data: CsvData, csv_path: Path) -> Path:
    source_video = data.meta_dict.get("source_video", "").strip()
    if not source_video:
        raise RuntimeError("CSVに source_video がありません")
    video_path = Path(source_video).expanduser()
    candidates = [video_path]
    if not video_path.is_absolute():
        candidates.insert(0, csv_path.parent / video_path)
    candidates.append(csv_path.parent / video_path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return video_path


def format_frame_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{start}-{end}" for start, end in ranges)


def fit_button_to_text(button: QPushButton) -> None:
    width = button.fontMetrics().horizontalAdvance(button.text()) + 28
    button.setMinimumWidth(width)
    button.setMaximumWidth(width)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def is_on(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "t", "on", "yes", "y"}


def true_false(value: str | None) -> str:
    return "T" if is_on(value) else "F"


def enabled_mosaic_count(row: dict[str, str]) -> int:
    return sum(is_on(row.get(f"mosaic{slot}_on")) for slot in range(1, MAX_MOSAICS + 1))


def set_blank_crotch(row: dict[str, str], slot: int) -> None:
    row[f"mosaic{slot}_crotch_no"] = ""
    row[f"mosaic{slot}_crotch_center"] = ""


def get_rect(row: dict[str, str], slot: int) -> QRect | None:
    try:
        x1 = int(float(row.get(f"mosaic{slot}_x1", "")))
        y1 = int(float(row.get(f"mosaic{slot}_y1", "")))
        x2 = int(float(row.get(f"mosaic{slot}_x2", "")))
        y2 = int(float(row.get(f"mosaic{slot}_y2", "")))
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return QRect(x1, y1, x2 - x1, y2 - y1)


def set_rect(row: dict[str, str], slot: int, rect: QRect, on: bool = True) -> None:
    rect = rect.normalized()
    row[f"mosaic{slot}_on"] = "1" if on else "0"
    if not row.get(f"mosaic{slot}_type"):
        row[f"mosaic{slot}_type"] = "manual"
    row[f"mosaic{slot}_score"] = ""
    row[f"mosaic{slot}_x1"] = str(rect.left())
    row[f"mosaic{slot}_y1"] = str(rect.top())
    row[f"mosaic{slot}_x2"] = str(rect.right() + 1)
    row[f"mosaic{slot}_y2"] = str(rect.bottom() + 1)
    set_blank_crotch(row, slot)


def clamp_rect(rect: QRect, width: int, height: int) -> QRect:
    rect = rect.normalized()
    left = max(0, min(width - 1, rect.left()))
    top = max(0, min(height - 1, rect.top()))
    right = max(left + 1, min(width, rect.right() + 1))
    bottom = max(top + 1, min(height, rect.bottom() + 1))
    return QRect(left, top, right - left, bottom - top)


def rect_to_xywh(rect: QRect) -> tuple[int, int, int, int]:
    return rect.left(), rect.top(), rect.width(), rect.height()


def xywh_to_rect(x: int, y: int, width: int, height: int) -> QRect:
    return QRect(x, y, max(1, width), max(1, height))


def track_rect_template(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    rect: QRect,
    search_scale: float = 2.0,
) -> tuple[QRect, float] | None:
    rect = rect.normalized()
    x, y, w, h = rect_to_xywh(rect)
    if w < 4 or h < 4:
        return None
    prev_h, prev_w = prev_frame.shape[:2]
    next_h, next_w = next_frame.shape[:2]
    if x < 0 or y < 0 or x + w > prev_w or y + h > prev_h:
        return None

    template = cv2.cvtColor(prev_frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    if template.size == 0:
        return None
    margin_x = max(12, round(w * search_scale))
    margin_y = max(12, round(h * search_scale))
    sx1 = max(0, x - margin_x)
    sy1 = max(0, y - margin_y)
    sx2 = min(next_w, x + w + margin_x)
    sy2 = min(next_h, y + h + margin_y)
    search = cv2.cvtColor(next_frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
    if search.size == 0:
        return None

    best: tuple[float, QRect] | None = None
    for scale in (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3):
        tw = max(4, round(w * scale))
        th = max(4, round(h * scale))
        if tw > search.shape[1] or th > search.shape[0]:
            continue
        scaled_template = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score):
            continue
        candidate = xywh_to_rect(sx1 + loc[0], sy1 + loc[1], tw, th)
        if best is None or score > best[0]:
            best = (float(score), candidate)

    if best is None:
        return None
    return best[1], best[0]


def refine_rect_with_optical_flow(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    rect: QRect,
    allow_scale: bool = True,
) -> QRect | None:
    rect = rect.normalized()
    x, y, w, h = rect_to_xywh(rect)
    prev_h, prev_w = prev_frame.shape[:2]
    if x < 0 or y < 0 or x + w > prev_w or y + h > prev_h:
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(prev_gray.shape, dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=max(3, min(w, h) // 10),
        mask=mask,
    )
    if points is None or len(points) < 4:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, points, None)
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    if int(valid.sum()) < 4:
        return None
    prev_valid = points[valid].reshape(-1, 2)
    next_valid = next_points[valid].reshape(-1, 2)
    deltas = next_valid - prev_valid
    dx, dy = np.median(deltas, axis=0)
    if not allow_scale:
        return QRect(round(x + float(dx)), round(y + float(dy)), w, h)

    prev_center = np.median(prev_valid, axis=0)
    next_center = np.median(next_valid, axis=0)
    prev_dist = np.linalg.norm(prev_valid - prev_center, axis=1)
    next_dist = np.linalg.norm(next_valid - next_center, axis=1)
    valid_dist = prev_dist > 1.0
    if int(valid_dist.sum()) < 4:
        scale = 1.0
    else:
        scale = float(np.median(next_dist[valid_dist] / prev_dist[valid_dist]))
    scale = max(0.6, min(1.8, scale))
    new_w = max(4, round(w * scale))
    new_h = max(4, round(h * scale))
    return QRect(
        round(x + float(dx) + (w - new_w) / 2),
        round(y + float(dy) + (h - new_h) / 2),
        new_w,
        new_h,
    )


def keep_original_size(candidate: QRect, original: QRect) -> QRect:
    center = candidate.center()
    left = round(center.x() - original.width() / 2)
    top = round(center.y() - original.height() / 2)
    return QRect(left, top, original.width(), original.height())


def track_rect(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    rect: QRect,
    allow_scale: bool = True,
) -> tuple[QRect, float, str] | None:
    template_result = track_rect_template(prev_frame, next_frame, rect)
    flow_rect = refine_rect_with_optical_flow(prev_frame, next_frame, rect, allow_scale=allow_scale)
    if template_result is None and flow_rect is None:
        return None
    if template_result is None and flow_rect is not None:
        return flow_rect, 0.0, "flow"
    template_rect, score = template_result
    if flow_rect is None:
        return template_rect, score, "template"

    dx = abs(template_rect.center().x() - flow_rect.center().x())
    dy = abs(template_rect.center().y() - flow_rect.center().y())
    tolerance = max(8, min(rect.width(), rect.height()) // 2)
    if dx <= tolerance and dy <= tolerance:
        result_rect = template_rect if allow_scale else keep_original_size(template_rect, rect)
        return result_rect, score, "template+flow" if allow_scale else "template+flow-fixed"
    if score >= 0.55:
        result_rect = template_rect if allow_scale else keep_original_size(template_rect, rect)
        return result_rect, score, "template" if allow_scale else "template-fixed"
    return flow_rect, score, "flow"


def interpolate_rect(start: QRect, end: QRect, t: float) -> QRect:
    x = round(start.left() + (end.left() - start.left()) * t)
    y = round(start.top() + (end.top() - start.top()) * t)
    width = round(start.width() + (end.width() - start.width()) * t)
    height = round(start.height() + (end.height() - start.height()) * t)
    return QRect(x, y, max(1, width), max(1, height))


def apply_preview_effect(
    frame: np.ndarray,
    rect: QRect,
    intensity: int,
    effect: str,
) -> None:
    x1, y1 = rect.left(), rect.top()
    x2, y2 = rect.right() + 1, rect.bottom() + 1
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    if effect == "blur":
        kernel_size = intensity if intensity % 2 == 1 else intensity + 1
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        return
    small = cv2.resize(
        roi,
        (max(1, w // intensity), max(1, h // intensity)),
        interpolation=cv2.INTER_LINEAR,
    )
    frame[y1:y2, x1:x2] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST
    )


def draw_pose_overlay(
    frame: np.ndarray,
    crotch_boxes: list[tuple[int, int, int, int]],
    pose_keypoints: list[np.ndarray],
    draw_crotch: bool,
    draw_skeleton: bool,
) -> None:
    if draw_skeleton:
        for kps in pose_keypoints:
            for a, b in SKELETON_EDGES:
                if a >= len(kps) or b >= len(kps):
                    continue
                if kps[a][2] < POSE_OVERLAY_MIN_SCORE or kps[b][2] < POSE_OVERLAY_MIN_SCORE:
                    continue
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, pa, pb, (0, 210, 255), 2, cv2.LINE_AA)
            for kp in kps:
                if kp[2] < POSE_OVERLAY_MIN_SCORE:
                    continue
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 80, 40), -1, cv2.LINE_AA)
    if draw_crotch:
        for idx, (x1, y1, x2, y2) in enumerate(crotch_boxes, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                frame,
                f"crotch{idx}",
                (x1 + 4, max(16, y1 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )


class SequenceSlider(QSlider):
    def __init__(self, frame_numbers: list[int]) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.frame_numbers = frame_numbers
        self.keep_ranges: list[tuple[int, int]] = []
        self.keep_start_marker: int | None = None
        self.keep_end_marker: int | None = None
        self.setMinimumHeight(42)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 8px;
                background: transparent;
            }
            QSlider::handle:horizontal {
                width: 5px;
                margin: -7px 0;
                background: #0068c9;
                border: 1px solid #004b91;
            }
            """
        )

    def set_keep_ranges(self, keep_ranges: list[tuple[int, int]]) -> None:
        self.keep_ranges = keep_ranges
        self.update()

    def set_keep_markers(self, start_frame: int | None, end_frame: int | None) -> None:
        self.keep_start_marker = start_frame
        self.keep_end_marker = end_frame
        self.update()

    def tick_index(self, frame_no: int) -> int:
        index = bisect.bisect_left(self.frame_numbers, frame_no)
        if index >= len(self.frame_numbers):
            return len(self.frame_numbers) - 1
        if index > 0 and frame_no - self.frame_numbers[index - 1] < self.frame_numbers[index] - frame_no:
            return index - 1
        return index

    def paintEvent(self, event) -> None:
        self.paint_trim_ranges()
        super().paintEvent(event)
        if not self.frame_numbers:
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_max = max(0, self.width() - handle.width())
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#505050")))
        first_frame = self.frame_numbers[0]
        last_frame = self.frame_numbers[-1]
        for frame_no in range(first_frame, last_frame + 1, 100):
            position = QStyle.sliderPositionFromValue(
                self.minimum(),
                self.maximum(),
                self.tick_index(frame_no),
                slider_max,
                upsideDown=option.upsideDown,
            )
            x = position + handle.width() // 2
            painter.drawLine(x, 24, x, 29)
            label_x = max(0, min(self.width() - 80, x - 40))
            alignment = Qt.AlignmentFlag.AlignHCenter
            if x < 40:
                alignment = Qt.AlignmentFlag.AlignLeft
            elif x > self.width() - 40:
                alignment = Qt.AlignmentFlag.AlignRight
            painter.drawText(label_x, 29, 80, 13, alignment, str(frame_no))
        self.paint_keep_markers(painter, option, handle)

    def paint_trim_ranges(self) -> None:
        if not self.frame_numbers:
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_max = max(1, self.width() - handle.width())
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        y = groove.center().y() - 5
        height = 10
        left_offset = handle.width() // 2
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        first_index = self.minimum()
        last_index = self.maximum()
        if last_index <= first_index:
            return

        def index_to_x(index: int) -> int:
            position = QStyle.sliderPositionFromValue(
                first_index,
                last_index,
                max(first_index, min(last_index, index)),
                slider_max,
                upsideDown=option.upsideDown,
            )
            return position + left_offset

        full_left = index_to_x(first_index)
        full_right = index_to_x(last_index)
        if self.keep_ranges:
            painter.setBrush(QColor(230, 70, 70, 110))
            painter.drawRect(full_left, y, max(1, full_right - full_left), height)
            painter.setBrush(QColor(60, 130, 240, 120))
            for start_frame, end_frame in self.keep_ranges:
                start_index = self.tick_index(start_frame)
                end_index = self.tick_index(end_frame)
                left = index_to_x(start_index)
                right = index_to_x(end_index)
                painter.drawRect(left, y, max(2, right - left), height)
        else:
            painter.setBrush(QColor(60, 130, 240, 100))
            painter.drawRect(full_left, y, max(1, full_right - full_left), height)

    def paint_keep_markers(self, painter: QPainter, option: QStyleOptionSlider, handle: QRect) -> None:
        if not self.frame_numbers:
            return
        slider_max = max(1, self.width() - handle.width())
        left_offset = handle.width() // 2

        def marker_x(frame_no: int) -> int:
            index = self.tick_index(frame_no)
            position = QStyle.sliderPositionFromValue(
                self.minimum(),
                self.maximum(),
                max(self.minimum(), min(self.maximum(), index)),
                slider_max,
                upsideDown=option.upsideDown,
            )
            return position + left_offset

        for label, frame_no, color in (
            ("S", self.keep_start_marker, QColor(30, 150, 70)),
            ("E", self.keep_end_marker, QColor(220, 120, 20)),
        ):
            if frame_no is None:
                continue
            x = marker_x(frame_no)
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, 14, x, 29)
            painter.setPen(color)
            painter.drawText(max(0, x - 8), 0, 16, 12, Qt.AlignmentFlag.AlignHCenter, label)

    def mousePressEvent(self, event) -> None:
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        if event.button() == Qt.MouseButton.LeftButton and not handle.contains(event.position().toPoint()):
            slider_max = max(0, self.width() - handle.width())
            position = round(event.position().x() - handle.width() / 2)
            self.setValue(
                QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    position,
                    slider_max,
                    upsideDown=option.upsideDown,
                )
            )
            event.accept()
            return
        super().mousePressEvent(event)


class CanvasScrollArea(QScrollArea):
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        canvas = self.widget()
        if canvas is not None and hasattr(canvas, "update_canvas_size"):
            canvas.update_canvas_size()


class FrameCanvas(QWidget):
    def __init__(self, parent: "EditorWindow") -> None:
        super().__init__()
        self.parent_window = parent
        self.pixmap: QPixmap | None = None
        self.image_size = None
        self.drag_mode: str | None = None
        self.drag_start_img = QPoint()
        self.drag_start_rect = QRect()
        self.pan_start_pos = QPoint()
        self.pan_start_scroll = QPoint()
        self.zoom_factor = 1.0
        self.scroll_area: QScrollArea | None = None
        self.resize(640, 360)
        self.setMouseTracking(True)

    def set_scroll_area(self, scroll_area: QScrollArea) -> None:
        self.scroll_area = scroll_area
        self.update_canvas_size()

    def set_frame(self, pixmap: QPixmap, image_size) -> None:
        self.pixmap = pixmap
        self.image_size = image_size
        self.update_canvas_size()
        self.update()

    def viewport_size(self):
        if self.scroll_area is not None:
            return self.scroll_area.viewport().size()
        return self.size()

    def update_canvas_size(self) -> None:
        if self.scroll_area is None:
            return
        viewport = self.viewport_size()
        width = viewport.width()
        height = viewport.height()
        if self.pixmap:
            scaled = self.pixmap.size()
            scaled.scale(viewport, Qt.AspectRatioMode.KeepAspectRatio)
            width = max(width, round(scaled.width() * self.zoom_factor))
            height = max(height, round(scaled.height() * self.zoom_factor))
        self.resize(width, height)
        self.update()

    def image_rect_on_widget(self) -> QRectF:
        if not self.pixmap:
            return QRectF()
        scaled = self.pixmap.size()
        scaled.scale(self.viewport_size(), Qt.AspectRatioMode.KeepAspectRatio)
        width = scaled.width() * self.zoom_factor
        height = scaled.height() * self.zoom_factor
        x = (self.width() - width) / 2
        y = (self.height() - height) / 2
        return QRectF(x, y, width, height)

    def widget_to_image(self, pos: QPoint) -> QPoint:
        image_rect = self.image_rect_on_widget()
        if image_rect.isEmpty() or not self.pixmap:
            return QPoint()
        x = int((pos.x() - image_rect.x()) * self.pixmap.width() / image_rect.width())
        y = int((pos.y() - image_rect.y()) * self.pixmap.height() / image_rect.height())
        return QPoint(max(0, min(self.pixmap.width() - 1, x)), max(0, min(self.pixmap.height() - 1, y)))

    def image_to_widget_rect(self, rect: QRect) -> QRectF:
        image_rect = self.image_rect_on_widget()
        if image_rect.isEmpty() or not self.pixmap:
            return QRectF()
        sx = image_rect.width() / self.pixmap.width()
        sy = image_rect.height() / self.pixmap.height()
        return QRectF(
            image_rect.x() + rect.left() * sx,
            image_rect.y() + rect.top() * sy,
            rect.width() * sx,
            rect.height() * sy,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(22, 24, 28))
        if not self.pixmap:
            return
        target = self.image_rect_on_widget()
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))

        current = self.parent_window.current_row()
        selected = self.parent_window.selected_slot
        for slot in range(1, MAX_MOSAICS + 1):
            rect = get_rect(current, slot)
            if rect is None:
                continue
            if not is_on(current.get(f"mosaic{slot}_on")) and slot != selected:
                continue
            wrect = self.image_to_widget_rect(rect)
            if slot == selected:
                pen = QPen(QColor(255, 210, 60), 3)
            elif is_on(current.get(f"mosaic{slot}_on")):
                pen = QPen(QColor(70, 220, 110), 2)
            else:
                pen = QPen(QColor(180, 180, 180), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wrect)
            painter.drawText(wrect.topLeft() + QPointF(4, -4), f"mosaic{slot}")
            if slot == selected:
                painter.setBrush(QColor(255, 210, 60))
                for handle in self.handles_for(wrect):
                    painter.drawRect(handle)

    def handles_for(self, rect: QRectF) -> list[QRectF]:
        half = HANDLE_SIZE / 2
        points = [
            rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight(),
            QPoint(int(rect.center().x()), int(rect.top())),
            QPoint(int(rect.center().x()), int(rect.bottom())),
            QPoint(int(rect.left()), int(rect.center().y())),
            QPoint(int(rect.right()), int(rect.center().y())),
        ]
        return [QRectF(p.x() - half, p.y() - half, HANDLE_SIZE, HANDLE_SIZE) for p in points]

    def hit_mode(self, pos: QPoint, rect: QRect) -> str | None:
        wrect = self.image_to_widget_rect(rect)
        pos_f = QPointF(pos)
        if wrect.adjusted(-HANDLE_SIZE, -HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE).contains(pos_f):
            margin = HANDLE_SIZE * 1.5
            left = abs(pos.x() - wrect.left()) <= margin
            right = abs(pos.x() - wrect.right()) <= margin
            top = abs(pos.y() - wrect.top()) <= margin
            bottom = abs(pos.y() - wrect.bottom()) <= margin
            if left and top:
                return "resize_tl"
            if right and top:
                return "resize_tr"
            if left and bottom:
                return "resize_bl"
            if right and bottom:
                return "resize_br"
            if wrect.contains(pos_f):
                return "move"
        return None

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.scroll_area is not None
        ):
            self.drag_mode = "pan"
            self.pan_start_pos = event.globalPosition().toPoint()
            self.pan_start_scroll = QPoint(
                self.scroll_area.horizontalScrollBar().value(),
                self.scroll_area.verticalScrollBar().value(),
            )
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            row = self.parent_window.current_row()
            slot = self.parent_window.selected_slot
            img_pos = self.widget_to_image(event.pos())
            self.drag_mode = "create"
            self.drag_start_img = img_pos
            self.drag_start_rect = QRect(img_pos, img_pos)
            set_rect(row, slot, QRect(img_pos, img_pos), on=True)
            self.parent_window.mark_dirty()
            self.parent_window.refresh_mosaic_table()
            self.update()
            log_user_action("モザイク枠作成開始", frame=self.parent_window.current_frame_no(), slot=slot, x=img_pos.x(), y=img_pos.y())
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        row = self.parent_window.current_row()
        img_pos = self.widget_to_image(event.pos())
        for candidate_slot in range(1, MAX_MOSAICS + 1):
            rect = get_rect(row, candidate_slot)
            if rect is None:
                continue
            if not is_on(row.get(f"mosaic{candidate_slot}_on")) and candidate_slot != self.parent_window.selected_slot:
                continue
            mode = self.hit_mode(event.pos(), rect)
            if mode:
                self.parent_window.selected_slot = candidate_slot
                self.parent_window.refresh_mosaic_table()
                self.drag_mode = mode
                self.drag_start_img = img_pos
                self.drag_start_rect = QRect(rect)
                return

    def mouseMoveEvent(self, event) -> None:
        if not self.drag_mode:
            return
        if self.drag_mode == "pan":
            delta = event.globalPosition().toPoint() - self.pan_start_pos
            self.scroll_area.horizontalScrollBar().setValue(self.pan_start_scroll.x() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(self.pan_start_scroll.y() - delta.y())
            event.accept()
            return
        row = self.parent_window.current_row()
        slot = self.parent_window.selected_slot
        pos = self.widget_to_image(event.pos())
        rect = QRect(self.drag_start_rect)
        dx = pos.x() - self.drag_start_img.x()
        dy = pos.y() - self.drag_start_img.y()
        if self.drag_mode == "move":
            rect.translate(dx, dy)
        elif self.drag_mode == "resize_tl":
            rect.setTopLeft(pos)
        elif self.drag_mode == "resize_tr":
            rect.setTopRight(pos)
        elif self.drag_mode == "resize_bl":
            rect.setBottomLeft(pos)
        elif self.drag_mode == "resize_br":
            rect.setBottomRight(pos)
        else:
            rect = QRect(self.drag_start_img, pos)
        rect = rect.normalized()
        if rect.width() >= 2 and rect.height() >= 2:
            rect = clamp_rect(rect, self.parent_window.image_width, self.parent_window.image_height)
            set_rect(row, slot, rect, on=True)
            self.parent_window.mark_dirty()
            self.parent_window.refresh_mosaic_table()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        released_mode = self.drag_mode
        released_slot = self.parent_window.selected_slot
        if self.drag_mode == "pan":
            self.unsetCursor()
        elif self.drag_mode:
            rect = get_rect(self.parent_window.current_row(), released_slot)
            if rect is not None:
                log_user_action(
                    "モザイク枠編集完了",
                    frame=self.parent_window.current_frame_no(),
                    slot=released_slot,
                    mode=released_mode,
                    x1=rect.left(),
                    y1=rect.top(),
                    x2=rect.right() + 1,
                    y2=rect.bottom() + 1,
                )
        self.drag_mode = None

    def wheelEvent(self, event) -> None:
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.zoom_factor = min(MAX_ZOOM, self.zoom_factor * ZOOM_STEP)
        elif event.angleDelta().y() < 0:
            self.zoom_factor = max(MIN_ZOOM, self.zoom_factor / ZOOM_STEP)
        horizontal_bar = self.scroll_area.horizontalScrollBar() if self.scroll_area else None
        vertical_bar = self.scroll_area.verticalScrollBar() if self.scroll_area else None
        horizontal_ratio = horizontal_bar.value() / horizontal_bar.maximum() if horizontal_bar and horizontal_bar.maximum() else 0.5
        vertical_ratio = vertical_bar.value() / vertical_bar.maximum() if vertical_bar and vertical_bar.maximum() else 0.5
        self.update_canvas_size()
        if horizontal_bar is not None:
            horizontal_bar.setValue(round(horizontal_bar.maximum() * horizontal_ratio))
        if vertical_bar is not None:
            vertical_bar.setValue(round(vertical_bar.maximum() * vertical_ratio))
        self.parent_window.update_zoom_label(self.zoom_factor)
        log_user_action("表示ズーム変更", zoom=round(self.zoom_factor, 3))
        event.accept()


class EditorWindow(QMainWindow):
    def __init__(self, csv_path: Path | None = None) -> None:
        super().__init__()
        APP_LOGGER.info("編集ウィンドウ初期化: csv_path=%s", csv_path)
        self.csv_path: Path | None = None
        self.dirty = False
        self.editor_windows: list[EditorWindow] = []
        self.cap = None
        self.setWindowTitle("pre CSV Editor")
        self.resize(1500, 980)
        if csv_path is None:
            self.build_empty_ui()
        else:
            self.load_csv(csv_path)

    def build_empty_ui(self) -> None:
        self.menuBar().clear()
        self.build_menu()
        label = QLabel("レシピを開くかレシピ生成を実行してください")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(label)

    def load_csv(self, csv_path: Path) -> None:
        APP_LOGGER.info("CSV読込開始: %s", csv_path)
        if hasattr(self, "playback_active") and self.playback_active:
            self.stop_preview_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        old_central = self.centralWidget()
        if old_central is not None:
            old_central.deleteLater()
        self.menuBar().clear()
        self.csv_path = csv_path
        self.data = read_pre_csv(csv_path)
        self.original_rows = [dict(row) for row in self.data.rows]
        self.video_path = source_video_path(self.data, self.csv_path)
        if not self.video_path.is_file():
            APP_LOGGER.error("元動画が見つかりません: csv=%s video=%s", self.csv_path, self.video_path)
            raise RuntimeError(f"元動画が見つかりません: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            APP_LOGGER.error("動画を開けません: csv=%s video=%s", self.csv_path, self.video_path)
            raise RuntimeError(f"動画を開けません: {self.video_path}")
        self.current_index = 0
        self.selected_slot = 1
        self.dirty = False
        self.display_cap_pos = None
        self.image_width = 0
        self.image_height = 0
        self.source_frame: np.ndarray | None = None
        self.trace_slots: set[int] = set()
        self.trace_scale_slots: set[int] = set()
        self.trace_ranges: dict[int, tuple[int, int]] = {}
        self.auto_track_anchors: dict[int, tuple[int, QRect, np.ndarray, str]] = {}
        self.pose_model_bundle = None
        self.pose_model_error: str | None = None
        self.pose_overlay_cache: dict[int, tuple[list[tuple[int, int, int, int]], list[np.ndarray]]] = {}
        self.keep_range_start: int | None = None
        self.keep_range_end: int | None = None
        self.display_cap_pos: int | None = None
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_next_frame)
        self.playback_index: int | None = None
        self.playback_cap_pos: int | None = None
        self.playback_active = False

        self.setWindowTitle(f"pre CSV Editor - {csv_path.name}")
        self.build_ui(show_load_progress=True)
        self.load_frame()
        APP_LOGGER.info("CSV読込完了: csv=%s rows=%s video=%s", csv_path, len(self.data.rows), self.video_path)

    def build_ui(self, show_load_progress: bool = False) -> None:
        self.build_menu()

        prev_frame_action = QAction("前のフレーム", self)
        prev_frame_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        prev_frame_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        prev_frame_action.triggered.connect(self.prev_frame)
        self.addAction(prev_frame_action)

        next_frame_action = QAction("次のフレーム", self)
        next_frame_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        next_frame_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        next_frame_action.triggered.connect(self.next_frame)
        self.addAction(next_frame_action)

        prev_row_action = QAction("前のCSV行", self)
        prev_row_action.setShortcut(QKeySequence(Qt.Key.Key_Up))
        prev_row_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        prev_row_action.triggered.connect(self.prev_frame)
        self.addAction(prev_row_action)

        next_row_action = QAction("次のCSV行", self)
        next_row_action.setShortcut(QKeySequence(Qt.Key.Key_Down))
        next_row_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        next_row_action.triggered.connect(self.next_frame)
        self.addAction(next_row_action)

        prev_page_action = QAction("前のCSV行 (PgUp)", self)
        prev_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageUp))
        prev_page_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        prev_page_action.triggered.connect(self.prev_frame)
        self.addAction(prev_page_action)

        next_page_action = QAction("次のCSV行 (PgDn)", self)
        next_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageDown))
        next_page_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        next_page_action.triggered.connect(self.next_frame)
        self.addAction(next_page_action)

        delete_mosaic_action = QAction("選択モザイクを削除", self)
        delete_mosaic_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        delete_mosaic_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        delete_mosaic_action.triggered.connect(self.disable_selected)
        self.addAction(delete_mosaic_action)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.info_label = QLabel(f"CSV: {self.csv_path}\nVideo: {self.video_path}")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)
        self.canvas = FrameCanvas(self)
        self.canvas_scroll_area = CanvasScrollArea()
        self.canvas_scroll_area.setWidgetResizable(False)
        self.canvas_scroll_area.setWidget(self.canvas)
        self.canvas.set_scroll_area(self.canvas_scroll_area)
        left_layout.addWidget(self.canvas_scroll_area, stretch=1)
        self.frame_label = QLabel()
        self.zoom_label = QLabel("100%")
        self.preview_checkbox = QCheckBox("mosaicプレビュー")
        self.preview_checkbox.setChecked(True)
        self.crotch_overlay_checkbox = QCheckBox("Crotch")
        self.crotch_overlay_checkbox.setChecked(True)
        self.skeleton_overlay_checkbox = QCheckBox("Skeleton")
        self.skeleton_overlay_checkbox.setChecked(True)
        frame_status_layout = QHBoxLayout()
        frame_status_layout.addWidget(self.frame_label)
        frame_status_layout.addWidget(self.zoom_label)
        frame_status_layout.addWidget(self.preview_checkbox)
        frame_status_layout.addWidget(self.crotch_overlay_checkbox)
        frame_status_layout.addWidget(self.skeleton_overlay_checkbox)
        frame_status_layout.addStretch()
        left_layout.addLayout(frame_status_layout)
        self.keep_start_button = QPushButton("残す開始")
        self.keep_end_button = QPushButton("残す終了")
        self.keep_add_button = QPushButton("残す範囲決定")
        self.keep_list_button = QPushButton("残す範囲一覧")
        self.keep_clear_button = QPushButton("残す範囲クリア")
        self.preview_play_button = QPushButton("仮再生")
        for button in (
            self.keep_start_button,
            self.keep_end_button,
            self.keep_add_button,
            self.keep_list_button,
            self.keep_clear_button,
            self.preview_play_button,
        ):
            fit_button_to_text(button)
        trimming_group = QGroupBox("トリミング")
        trimming_group_layout = QHBoxLayout(trimming_group)
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(self.keep_start_button)
        trim_layout.addWidget(self.keep_end_button)
        trim_layout.addWidget(self.keep_add_button)
        trim_layout.addWidget(self.keep_list_button)
        trim_layout.addWidget(self.keep_clear_button)
        trim_layout.addWidget(self.preview_play_button)
        trim_layout.addStretch()
        trimming_group_layout.addLayout(trim_layout)
        left_layout.addWidget(trimming_group)
        frame_numbers = [int(row["frame_no"]) for row in self.data.rows]
        self.sequence_slider = SequenceSlider(frame_numbers)
        self.sequence_slider.setRange(0, max(0, len(self.data.rows) - 1))
        self.sequence_slider.set_keep_ranges(self.keep_ranges())
        left_layout.addWidget(self.sequence_slider)

        nav = QHBoxLayout()
        self.skip_frame_input = QLineEdit("10")
        self.skip_frame_input.setPlaceholderText("skip")
        skip_input_width = self.skip_frame_input.fontMetrics().horizontalAdvance("000000") + 24
        self.skip_frame_input.setFixedWidth(skip_input_width)
        self.prev_skip_button = QPushButton("前へスキップ")
        self.prev_button = QPushButton("前へ")
        self.next_button = QPushButton("次へ")
        self.next_skip_button = QPushButton("次へスキップ")
        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("frame_no")
        frame_input_width = self.frame_input.fontMetrics().horizontalAdvance("0000000000") + 24
        self.frame_input.setFixedWidth(frame_input_width)
        self.go_button = QPushButton("移動")
        nav.addWidget(QLabel("skip frame:"))
        nav.addWidget(self.skip_frame_input)
        nav.addWidget(self.prev_skip_button)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        nav.addWidget(self.next_skip_button)
        nav.addWidget(self.go_button)
        nav.addWidget(QLabel("Frame:"))
        nav.addWidget(self.frame_input)
        nav.addStretch()
        left_layout.addLayout(nav)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        meta = self.data.meta_dict
        meta_layout = QFormLayout()
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["mosaic", "blur"])
        self.effect_combo.setCurrentText(meta.get("effect", "mosaic"))
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(1, 2147483647)
        self.intensity_spin.setFixedWidth(70)
        try:
            intensity = int(meta.get("intensity", "15"))
        except ValueError:
            intensity = 15
        self.intensity_spin.setValue(max(1, intensity))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(1, max(DEFAULT_INTENSITY_SLIDER_MAX, intensity))
        self.intensity_slider.setValue(max(1, intensity))
        intensity_layout = QHBoxLayout()
        intensity_layout.setContentsMargins(0, 0, 0, 0)
        intensity_layout.addWidget(self.intensity_spin)
        intensity_layout.addWidget(self.intensity_slider, stretch=1)
        meta_layout.addRow("intensity", intensity_layout)
        meta_layout.addRow("effect", self.effect_combo)
        for key in ("confidence", "pose_model", "yolo_nsfw_model", "interpolate_gap", "no_crotch", "skip_no_person"):
            meta_layout.addRow(key, QLabel(meta.get(key, "")))
        right_layout.addLayout(meta_layout)

        self.frame_table = QTableWidget(len(self.data.rows), 7)
        self.frame_table.setHorizontalHeaderLabels(
            ["frame_no", "modify", "Keep", "Persons", "Mosaic", "Crotch", "comment"]
        )
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        self.frame_table.verticalHeader().setVisible(False)
        right_layout.addWidget(QLabel("CSV行"))
        right_layout.addWidget(self.frame_table, stretch=1)

        self.type_input = QLineEdit()
        self.type_input.setPlaceholderText("mosaic type")
        self.auto_track_status = QLabel("")
        self.auto_track_status.setWordWrap(True)
        self.create_from_nearest_button = QPushButton("直近枠から作成")
        self.save_button = QPushButton("保存")
        self.encode_button = QPushButton("エンコード")
        self.restore_frame_button = QPushButton("選択したフレームを元に戻す(複数可能)")
        right_layout.addWidget(QLabel("Type"))
        right_layout.addWidget(self.type_input)
        right_layout.addWidget(self.auto_track_status)
        right_layout.addWidget(self.create_from_nearest_button)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.encode_button)
        right_layout.addWidget(self.restore_frame_button)

        self.mosaic_table = QTableWidget(DEFAULT_VISIBLE_MOSAICS, 14)
        self.mosaic_table.setHorizontalHeaderLabels(
            [
                "mosaic", "Trace", "T scale", "Start", "End", "type", "score",
                "w", "h", "x1", "y1", "x2", "y2", "comment",
            ]
        )
        self.mosaic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mosaic_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.mosaic_table, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([1100, 400])

        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.prev_skip_button.clicked.connect(self.prev_skip_frame)
        self.next_skip_button.clicked.connect(self.next_skip_frame)
        self.go_button.clicked.connect(self.go_to_frame)
        self.sequence_slider.valueChanged.connect(self.select_sequence_frame)
        self.save_button.clicked.connect(self.save_with_confirm)
        self.encode_button.clicked.connect(self.encode_post)
        self.restore_frame_button.clicked.connect(self.restore_current_frame)
        self.create_from_nearest_button.clicked.connect(self.create_from_nearest)
        self.keep_start_button.clicked.connect(self.set_keep_range_start)
        self.keep_end_button.clicked.connect(self.set_keep_range_end)
        self.keep_add_button.clicked.connect(self.add_current_selection_keep_range)
        self.keep_list_button.clicked.connect(self.show_keep_ranges)
        self.keep_clear_button.clicked.connect(self.clear_keep_ranges)
        self.preview_play_button.clicked.connect(self.toggle_preview_playback)
        self.preview_checkbox.stateChanged.connect(self.refresh_canvas_frame)
        self.crotch_overlay_checkbox.stateChanged.connect(self.refresh_canvas_frame)
        self.skeleton_overlay_checkbox.stateChanged.connect(self.refresh_canvas_frame)
        self.effect_combo.currentTextChanged.connect(self.update_preview_meta)
        self.intensity_slider.valueChanged.connect(self.intensity_spin.setValue)
        self.intensity_spin.valueChanged.connect(self.sync_intensity_slider)
        self.intensity_spin.valueChanged.connect(self.update_preview_meta)
        self.type_input.editingFinished.connect(self.update_selected_type)
        self.mosaic_table.cellClicked.connect(self.select_mosaic)
        self.mosaic_table.itemChanged.connect(self.update_mosaic_from_table)
        self.frame_table.cellClicked.connect(self.select_frame_row)
        self.frame_table.itemSelectionChanged.connect(self.select_selected_frame_row)
        self.frame_table.itemChanged.connect(self.update_frame_from_table)
        self.populate_frame_table(
            show_progress=show_load_progress,
            progress_message="レシピファイルを読み込み中...",
        )

    def build_menu(self) -> None:
        menu = self.menuBar().addMenu("メニュー")
        version_action = QAction(f"version {APP_VERSION}", self)
        version_action.setEnabled(False)
        self.menuBar().addAction(version_action)

        create_action = QAction("レシピ生成", self)
        create_action.triggered.connect(self.create_recipe_from_menu)
        menu.addAction(create_action)

        open_action = QAction("レシピを開く", self)
        open_action.triggered.connect(self.open_recipe_from_menu)
        menu.addAction(open_action)

        save_action = QAction("レシピを保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_with_confirm)
        save_action.setEnabled(self.csv_path is not None)
        menu.addAction(save_action)

        encode_action = QAction("エンコード", self)
        encode_action.triggered.connect(self.encode_post)
        encode_action.setEnabled(self.csv_path is not None)
        menu.addAction(encode_action)

        menu.addSeparator()

        exit_action = QAction("終了", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def create_recipe_from_menu(self) -> None:
        log_user_action("メニュー レシピ生成")
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "動画ファイルを選択",
            "",
            "Video (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All files (*)",
        )
        if not selected:
            log_user_action("レシピ生成 動画選択キャンセル")
            return
        log_user_action("レシピ生成 動画選択", path=selected)
        dialog = PreCreateDialog(Path(selected))
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_csv is not None:
            self.open_editor_window(dialog.result_csv)

    def open_recipe_from_menu(self) -> None:
        log_user_action("メニュー レシピを開く")
        selected, _ = QFileDialog.getOpenFileName(self, "_pre.csv を選択", "", "CSV (*.csv)")
        if selected:
            log_user_action("レシピ選択", path=selected)
            self.open_editor_window(Path(selected))

    def open_editor_window(self, csv_path: Path) -> None:
        log_user_action("レシピを開く", csv_path=csv_path)
        if self.csv_path is None:
            try:
                self.load_csv(csv_path)
            except Exception as exc:
                APP_LOGGER.exception("レシピを開けません: %s", csv_path)
                QMessageBox.critical(self, "Error", str(exc))
            return
        try:
            window = EditorWindow(csv_path)
        except Exception as exc:
            APP_LOGGER.exception("別ウィンドウでレシピを開けません: %s", csv_path)
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.editor_windows.append(window)
        window.destroyed.connect(lambda _obj=None, w=window: self.forget_editor_window(w))
        window.show()

    def forget_editor_window(self, window: "EditorWindow") -> None:
        if window in self.editor_windows:
            self.editor_windows.remove(window)

    def current_row(self) -> dict[str, str]:
        return self.data.rows[self.current_index]

    def update_zoom_label(self, zoom_factor: float) -> None:
        self.zoom_label.setText(f"{round(zoom_factor * 100)}%")

    def mark_dirty(self) -> None:
        self.dirty = True
        if not self.windowTitle().endswith("*"):
            self.setWindowTitle(self.windowTitle() + " *")

    def load_frame(self) -> None:
        frame_no_text = self.current_row().get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            APP_LOGGER.warning("frame_no が不正です: index=%s value=%s", self.current_index, frame_no_text)
            QMessageBox.warning(self, "Error", f"frame_no が不正です: {frame_no_text}")
            return
        frame = self.read_frame_number(frame_no, prefer_sequential=True)
        if frame is None:
            APP_LOGGER.warning("フレームを読めません: frame_no=%s video=%s", frame_no, self.video_path)
            QMessageBox.warning(self, "Error", f"フレームを読めません: frame_no={frame_no}")
            return
        self.source_frame = frame
        h, w = frame.shape[:2]
        self.image_width = w
        self.image_height = h
        self.refresh_canvas_frame()
        self.frame_label.setText(f"{self.current_index + 1}/{len(self.data.rows)}")
        self.frame_input.setText(str(frame_no))
        self.sequence_slider.blockSignals(True)
        self.sequence_slider.setValue(self.current_index)
        self.sequence_slider.blockSignals(False)
        if get_rect(self.current_row(), self.selected_slot) is None:
            self.populate_selected_from_nearest(on=False)
        self.select_current_frame_row()
        self.refresh_mosaic_table()

    def read_frame_number(self, frame_no: int, prefer_sequential: bool) -> np.ndarray | None:
        if self.cap is None:
            return None
        can_read_forward = (
            prefer_sequential
            and self.display_cap_pos is not None
            and frame_no >= self.display_cap_pos
            and frame_no - self.display_cap_pos <= DISPLAY_SEQUENTIAL_MAX_SKIP
        )
        if not can_read_forward:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            self.display_cap_pos = frame_no

        frame = None
        while self.display_cap_pos is not None and self.display_cap_pos <= frame_no:
            ok, frame = self.cap.read()
            if not ok:
                self.display_cap_pos = None
                return None
            self.display_cap_pos += 1
        return frame

    def refresh_canvas_frame(self, *args) -> None:
        if self.source_frame is None:
            return
        frame = self.source_frame.copy()
        if self.preview_checkbox.isChecked():
            meta = self.data.meta_dict
            try:
                intensity = max(1, int(meta.get("intensity", "15")))
            except ValueError:
                intensity = 15
            effect = meta.get("effect", "mosaic")
            for slot in range(1, MAX_MOSAICS + 1):
                if not is_on(self.current_row().get(f"mosaic{slot}_on")):
                    continue
                rect = get_rect(self.current_row(), slot)
                if rect is not None:
                    apply_preview_effect(frame, rect, intensity, effect)
        draw_crotch = self.crotch_overlay_checkbox.isChecked()
        draw_skeleton = self.skeleton_overlay_checkbox.isChecked()
        if draw_crotch or draw_skeleton:
            overlay = self.current_pose_overlay()
            if overlay is not None:
                crotch_boxes, pose_keypoints = overlay
                draw_pose_overlay(frame, crotch_boxes, pose_keypoints, draw_crotch, draw_skeleton)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(QPixmap.fromImage(qimg), (w, h))

    def render_recipe_frame(self, frame: np.ndarray, row: dict[str, str]) -> np.ndarray:
        result = frame.copy()
        meta = self.data.meta_dict
        try:
            intensity = max(1, int(meta.get("intensity", "15")))
        except ValueError:
            intensity = 15
        effect = meta.get("effect", "mosaic")
        for slot in range(1, MAX_MOSAICS + 1):
            if not is_on(row.get(f"mosaic{slot}_on")):
                continue
            rect = get_rect(row, slot)
            if rect is not None:
                apply_preview_effect(result, rect, intensity, effect)
        return result

    def playback_frame_at_index(self, index: int) -> np.ndarray | None:
        if self.cap is None:
            return None
        frame_no_text = self.data.rows[index].get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            return None
        if self.playback_cap_pos != frame_no:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.display_cap_pos = None
        ok, frame = self.cap.read()
        if not ok:
            self.playback_cap_pos = None
            return None
        self.playback_cap_pos = frame_no + 1
        return frame

    def show_playback_frame(self, index: int) -> bool:
        frame = self.playback_frame_at_index(index)
        if frame is None:
            return False
        row = self.data.rows[index]
        rendered = self.render_recipe_frame(frame, row)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(QPixmap.fromImage(qimg), (w, h))
        self.current_index = index
        self.frame_label.setText(f"{self.current_index + 1}/{len(self.data.rows)}")
        self.frame_input.setText(row.get("frame_no", ""))
        self.sequence_slider.blockSignals(True)
        self.sequence_slider.setValue(self.current_index)
        self.sequence_slider.blockSignals(False)
        return True

    def next_kept_index(self, start_index: int) -> int | None:
        keep_ranges = self.keep_ranges()
        for idx in range(max(0, start_index), len(self.data.rows)):
            if self.frame_is_kept(self.data.rows[idx], keep_ranges):
                return idx
        return None

    def toggle_preview_playback(self) -> None:
        if self.playback_active:
            log_user_action("仮再生停止", frame=self.current_frame_no())
            self.stop_preview_playback()
            return
        start_index = self.next_kept_index(self.current_index)
        if start_index is None:
            APP_LOGGER.info("仮再生不可: 再生できる残す範囲がありません")
            QMessageBox.information(self, "仮再生", "再生できる残す範囲がありません。")
            return
        log_user_action("仮再生開始", frame=self.data.rows[start_index].get("frame_no", ""))
        self.playback_active = True
        self.playback_index = start_index
        self.playback_cap_pos = None
        self.display_cap_pos = None
        self.preview_play_button.setText("停止")
        fit_button_to_text(self.preview_play_button)
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 30
        interval_ms = max(1, round(1000 / fps)) if fps and fps > 0 else 33
        self.playback_timer.start(interval_ms)
        self.playback_next_frame()

    def stop_preview_playback(self) -> None:
        if not self.playback_active:
            return
        self.playback_timer.stop()
        self.playback_active = False
        self.playback_index = None
        self.playback_cap_pos = None
        self.display_cap_pos = None
        self.preview_play_button.setText("仮再生")
        fit_button_to_text(self.preview_play_button)
        self.load_frame()

    def playback_next_frame(self) -> None:
        if self.playback_index is None:
            self.stop_preview_playback()
            return
        index = self.next_kept_index(self.playback_index)
        if index is None:
            self.stop_preview_playback()
            return
        if not self.show_playback_frame(index):
            self.stop_preview_playback()
            return
        self.playback_index = index + 1

    def current_pose_overlay(self) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]] | None:
        if self.source_frame is None:
            return None
        frame_no_text = self.current_row().get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            return None
        if frame_no in self.pose_overlay_cache:
            return self.pose_overlay_cache[frame_no]
        bundle = self.ensure_pose_model()
        if bundle is None:
            return None
        try:
            overlay = get_crotch_boxes(self.source_frame, bundle)
        except Exception as exc:
            self.pose_model_error = str(exc)
            self.auto_track_status.setText(f"Pose overlay error: {exc}")
            APP_LOGGER.exception("Pose overlay error: frame=%s", frame_no)
            return None
        self.pose_overlay_cache[frame_no] = overlay
        self.update_frame_table_row(self.current_index, self.current_row())
        return overlay

    def ensure_pose_model(self):
        if self.pose_model_bundle is not None:
            return self.pose_model_bundle
        if self.pose_model_error:
            return None
        backend = self.data.meta_dict.get("pose_model", "yolo8") or "yolo8"
        try:
            self.pose_model_bundle = load_pose_model(backend)
        except Exception as exc:
            self.pose_model_error = str(exc)
            self.auto_track_status.setText(f"Pose model load error: {exc}")
            APP_LOGGER.exception("Pose model load error: backend=%s", backend)
            return None
        return self.pose_model_bundle

    def update_preview_meta(self, *args) -> None:
        set_meta_value(self.data, "effect", self.effect_combo.currentText())
        set_meta_value(self.data, "intensity", str(self.intensity_spin.value()))
        log_user_action("プレビュー設定変更", effect=self.effect_combo.currentText(), intensity=self.intensity_spin.value())
        self.mark_dirty()
        self.refresh_canvas_frame()

    def sync_intensity_slider(self, intensity: int) -> None:
        if intensity > self.intensity_slider.maximum():
            self.intensity_slider.setMaximum(intensity)
        self.intensity_slider.setValue(intensity)

    def populate_frame_table(
        self,
        show_progress: bool = False,
        progress_message: str = "トリミング範囲を反映中...",
    ) -> None:
        keep_ranges = self.keep_ranges()
        progress = None
        start_time = time.monotonic()
        total = len(self.data.rows)
        if show_progress:
            progress = QProgressDialog(progress_message, "", 0, max(1, total), self)
            progress.setWindowTitle("処理中")
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        self.frame_table.blockSignals(True)
        try:
            for idx, row in enumerate(self.data.rows):
                self.update_frame_table_row(idx, row, keep_ranges)
                if progress is not None and (idx % 20 == 0 or idx + 1 == total):
                    done = idx + 1
                    elapsed = time.monotonic() - start_time
                    percent = int(done * 100 / max(1, total))
                    remaining = elapsed * (total - done) / done if done else 0
                    progress.setLabelText(
                        f"{progress_message} {percent}%  残り "
                        f"{time.strftime('%H:%M:%S', time.gmtime(max(0, remaining)))}"
                    )
                    progress.setValue(done)
                    QApplication.processEvents()
        finally:
            self.frame_table.blockSignals(False)
            if progress is not None:
                progress.setValue(max(1, total))

    def row_modified(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.original_rows):
            return False
        return self.data.rows[idx] != self.original_rows[idx]

    def keep_ranges(self) -> list[tuple[int, int]]:
        try:
            return parse_frame_ranges(self.data.meta_dict.get("keep_ranges", ""))
        except ValueError:
            return []

    def frame_is_kept(self, row: dict[str, str], keep_ranges: list[tuple[int, int]] | None = None) -> bool:
        try:
            frame_no = int(row.get("frame_no", ""))
        except ValueError:
            return True
        return frame_in_ranges(frame_no, self.keep_ranges() if keep_ranges is None else keep_ranges)

    def set_keep_ranges(self, ranges: list[tuple[int, int]], show_progress: bool = False) -> None:
        normalized = normalize_frame_ranges(ranges)
        set_meta_value(self.data, "keep_ranges", format_frame_ranges(normalized))
        log_user_action("残す範囲更新", ranges=format_frame_ranges(normalized) or "all")
        self.mark_dirty()
        self.sequence_slider.set_keep_ranges(self.keep_ranges())
        self.populate_frame_table(show_progress=show_progress)

    def update_frame_table_row(
        self,
        idx: int,
        row: dict[str, str],
        keep_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        mosaic_count = enabled_mosaic_count(row)
        nsfw_detection_count = row.get("nsfw_detection_count") or "?"
        persons = self.person_count_for_row(row)
        kept = self.frame_is_kept(row, keep_ranges)
        values = [
            row.get("frame_no", ""),
            "T" if self.row_modified(idx) else "F",
            "keep" if kept else "cut",
            persons,
            f"{mosaic_count}/{nsfw_detection_count}",
            "yes" if is_on(row.get("crotch_detected")) else "none",
            row.get("comment", ""),
        ]
        if not kept:
            background = QColor(205, 205, 205)
        else:
            background = QColor(255, 220, 230) if mosaic_count else QColor(232, 232, 232)
        self.frame_table.blockSignals(True)
        try:
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(background)
                if col != 6:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.frame_table.setItem(idx, col, item)
        finally:
            self.frame_table.blockSignals(False)

    def person_count_for_row(self, row: dict[str, str]) -> str:
        try:
            frame_no = int(row.get("frame_no", ""))
        except ValueError:
            return "?"
        overlay = self.pose_overlay_cache.get(frame_no)
        if overlay is None:
            return "?"
        _, pose_keypoints = overlay
        return str(len(pose_keypoints))

    def select_current_frame_row(self) -> None:
        self.frame_table.blockSignals(True)
        self.frame_table.selectRow(self.current_index)
        self.frame_table.scrollToItem(self.frame_table.item(self.current_index, 0))
        self.frame_table.blockSignals(False)

    def refresh_mosaic_table(self) -> None:
        row = self.current_row()
        visible_slots = self.visible_mosaic_slots(row)
        self.mosaic_table.blockSignals(True)
        self.mosaic_table.setRowCount(len(visible_slots))
        for table_row, slot in enumerate(visible_slots):
            self.mosaic_table.setVerticalHeaderItem(table_row, QTableWidgetItem(str(slot)))
            rect = get_rect(row, slot)
            values = [
                true_false(row.get(f"mosaic{slot}_on")),
                "T" if slot in self.trace_slots else "F",
                "T" if self.trace_scale_enabled(slot) else "F",
                str(self.trace_range_for_slot(slot)[0]),
                str(self.trace_range_for_slot(slot)[1]),
                row.get(f"mosaic{slot}_type", ""),
                row.get(f"mosaic{slot}_score", ""),
                str(rect.width()) if rect is not None else "",
                str(rect.height()) if rect is not None else "",
                row.get(f"mosaic{slot}_x1", ""),
                row.get(f"mosaic{slot}_y1", ""),
                row.get(f"mosaic{slot}_x2", ""),
                row.get(f"mosaic{slot}_y2", ""),
                row.get("comment", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 6, 7, 8):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if slot == self.selected_slot:
                    item.setBackground(QColor(255, 245, 200))
                self.mosaic_table.setItem(table_row, col, item)
        self.mosaic_table.blockSignals(False)
        selected_row = visible_slots.index(self.selected_slot) if self.selected_slot in visible_slots else 0
        self.mosaic_table.selectRow(selected_row)
        self.mosaic_table.scrollToItem(self.mosaic_table.item(selected_row, 0))
        self.type_input.setText(row.get(f"mosaic{self.selected_slot}_type", ""))
        self.update_frame_table_row(self.current_index, row)
        self.refresh_canvas_frame()
        self.canvas.update()

    def visible_mosaic_slots(self, row: dict[str, str]) -> list[int]:
        visible = list(range(1, DEFAULT_VISIBLE_MOSAICS + 1))
        for slot in range(DEFAULT_VISIBLE_MOSAICS + 1, MAX_MOSAICS + 1):
            if (
                is_on(row.get(f"mosaic{slot}_on"))
                or get_rect(row, slot) is not None
                or slot in self.trace_slots
            ):
                visible.append(slot)
        if self.selected_slot not in visible:
            visible.append(self.selected_slot)
        return visible

    def current_frame_no(self) -> int:
        try:
            return int(self.current_row().get("frame_no", "0"))
        except ValueError:
            return 0

    def trace_range_for_slot(self, slot: int) -> tuple[int, int]:
        frame_no = self.current_frame_no()
        return self.trace_ranges.get(slot, (frame_no, frame_no))

    def trace_scale_enabled(self, slot: int) -> bool:
        return slot in self.trace_scale_slots

    def frame_index_for_no(self, frame_no: int) -> int | None:
        for idx, row in enumerate(self.data.rows):
            try:
                if int(row.get("frame_no", "")) == frame_no:
                    return idx
            except ValueError:
                continue
        return None

    def update_mosaic_from_table(self, item: QTableWidgetItem) -> None:
        header_item = self.mosaic_table.verticalHeaderItem(item.row())
        if header_item is None:
            return
        try:
            slot = int(header_item.text())
        except ValueError:
            return
        col = item.column()
        if col == 1:
            if is_on(item.text()):
                self.trace_slots.add(slot)
                self.selected_slot = slot
                self.refresh_mosaic_table()
                log_user_action("追跡開始", frame=self.current_frame_no(), slot=slot, source="table")
                self.trace_slot_range(slot)
            else:
                self.trace_slots.discard(slot)
                self.auto_track_anchors.pop(slot, None)
                self.selected_slot = slot
                log_user_action("追跡解除", frame=self.current_frame_no(), slot=slot, source="table")
                self.refresh_mosaic_table()
            return
        if col == 2:
            if is_on(item.text()):
                self.trace_scale_slots.add(slot)
            else:
                self.trace_scale_slots.discard(slot)
            self.selected_slot = slot
            log_user_action("追跡スケール設定変更", frame=self.current_frame_no(), slot=slot, enabled=slot in self.trace_scale_slots)
            self.refresh_mosaic_table()
            return
        if col in (3, 4):
            current_start, current_end = self.trace_range_for_slot(slot)
            try:
                value = int(item.text().strip())
            except ValueError:
                self.auto_track_status.setText(f"mosaic{slot}: Start/End は frame_no の数値で入力してください")
                self.refresh_mosaic_table()
                return
            if col == 3:
                self.trace_ranges[slot] = (value, current_end)
            else:
                self.trace_ranges[slot] = (current_start, value)
            self.selected_slot = slot
            log_user_action("追跡範囲変更", frame=self.current_frame_no(), slot=slot, start=self.trace_ranges[slot][0], end=self.trace_ranges[slot][1])
            self.refresh_mosaic_table()
            return
        keys = {
            0: f"mosaic{slot}_on",
            5: f"mosaic{slot}_type",
            9: f"mosaic{slot}_x1",
            10: f"mosaic{slot}_y1",
            11: f"mosaic{slot}_x2",
            12: f"mosaic{slot}_y2",
        }
        if col == 13:
            self.current_row()["comment"] = item.text().strip()
            log_user_action("モザイクコメント変更", frame=self.current_frame_no(), slot=slot)
            self.mark_dirty()
            self.update_frame_table_row(self.current_index, self.current_row())
            self.refresh_mosaic_table()
            return
        key = keys.get(col)
        if not key:
            return
        value = true_false(item.text()) if col == 0 else item.text().strip()
        self.current_row()[key] = value
        if col >= 9:
            self.current_row()[f"mosaic{slot}_score"] = ""
            set_blank_crotch(self.current_row(), slot)
        self.mark_dirty()
        self.selected_slot = slot
        log_user_action("モザイク表編集", frame=self.current_frame_no(), slot=slot, key=key, value=value)
        self.refresh_mosaic_table()

    def update_frame_from_table(self, item: QTableWidgetItem) -> None:
        if item.column() != 6:
            return
        self.data.rows[item.row()]["comment"] = item.text().strip()
        log_user_action("フレームコメント変更", row=item.row(), frame=self.data.rows[item.row()].get("frame_no", ""))
        self.mark_dirty()
        self.update_frame_table_row(item.row(), self.data.rows[item.row()])

    def select_frame_row(self, row: int, col: int) -> None:
        if 0 <= row < len(self.data.rows):
            self.move_to_index(row)

    def select_selected_frame_row(self) -> None:
        selected = self.frame_table.selectedIndexes()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self.data.rows) and row != self.current_index:
            self.move_to_index(row)

    def select_sequence_frame(self, index: int) -> None:
        if 0 <= index < len(self.data.rows) and index != self.current_index:
            self.move_to_index(index)

    def select_mosaic(self, row: int, col: int) -> None:
        header_item = self.mosaic_table.verticalHeaderItem(row)
        if header_item is None:
            return
        try:
            self.selected_slot = int(header_item.text())
        except ValueError:
            return
        if col == 0:
            key = f"mosaic{self.selected_slot}_on"
            on = not is_on(self.current_row().get(key))
            self.current_row()[key] = "1" if on else "0"
            if on and get_rect(self.current_row(), self.selected_slot) is None:
                self.populate_selected_from_nearest(on=True)
            self.mark_dirty()
            log_user_action("モザイク有効切替", frame=self.current_frame_no(), slot=self.selected_slot, enabled=on)
            self.refresh_mosaic_table()
            return
        if col == 1:
            if self.selected_slot in self.trace_slots:
                self.trace_slots.discard(self.selected_slot)
                self.auto_track_anchors.pop(self.selected_slot, None)
                log_user_action("追跡解除", frame=self.current_frame_no(), slot=self.selected_slot, source="click")
                self.refresh_mosaic_table()
            else:
                self.trace_slots.add(self.selected_slot)
                log_user_action("追跡開始", frame=self.current_frame_no(), slot=self.selected_slot, source="click")
                self.refresh_mosaic_table()
                self.trace_slot_range(self.selected_slot)
            return
        if col == 2:
            if self.selected_slot in self.trace_scale_slots:
                self.trace_scale_slots.discard(self.selected_slot)
            else:
                self.trace_scale_slots.add(self.selected_slot)
            log_user_action("追跡スケール設定変更", frame=self.current_frame_no(), slot=self.selected_slot, enabled=self.selected_slot in self.trace_scale_slots)
            self.refresh_mosaic_table()
            return
        selected_rect = get_rect(self.current_row(), self.selected_slot)
        if selected_rect is None:
            self.populate_selected_from_nearest(on=False)
        self.refresh_mosaic_table()

    def current_frame_no(self) -> int | None:
        try:
            return int(self.current_row().get("frame_no", ""))
        except ValueError:
            return None

    def set_keep_range_start(self) -> None:
        frame_no = self.current_frame_no()
        if frame_no is None:
            APP_LOGGER.warning("残す開始を設定できません: 現在フレーム番号が不正")
            QMessageBox.warning(self, "Error", "現在フレーム番号が不正です")
            return
        self.keep_range_start = frame_no
        log_user_action("残す開始設定", frame=frame_no)
        self.sequence_slider.set_keep_markers(self.keep_range_start, self.keep_range_end)
        self.auto_track_status.setText(f"残す開始: frame {frame_no}")

    def set_keep_range_end(self) -> None:
        frame_no = self.current_frame_no()
        if frame_no is None:
            APP_LOGGER.warning("残す終了を設定できません: 現在フレーム番号が不正")
            QMessageBox.warning(self, "Error", "現在フレーム番号が不正です")
            return
        self.keep_range_end = frame_no
        log_user_action("残す終了設定", frame=frame_no)
        self.sequence_slider.set_keep_markers(self.keep_range_start, self.keep_range_end)
        self.auto_track_status.setText(f"残す終了: frame {frame_no}")

    def add_current_selection_keep_range(self) -> None:
        if self.keep_range_start is None or self.keep_range_end is None:
            APP_LOGGER.info("残す範囲追加失敗: 開始または終了が未指定")
            QMessageBox.information(self, "範囲未指定", "残す開始と残す終了を指定してください。")
            return
        start = min(self.keep_range_start, self.keep_range_end)
        end = max(self.keep_range_start, self.keep_range_end)
        ranges = self.keep_ranges()
        ranges.append((start, end))
        log_user_action("残す範囲追加", start=start, end=end)
        self.set_keep_ranges(ranges, show_progress=True)
        self.auto_track_status.setText(f"残す範囲を追加: {start}-{end}")
        self.keep_range_start = None
        self.keep_range_end = None
        self.sequence_slider.set_keep_markers(None, None)

    def show_keep_ranges(self) -> None:
        ranges = self.keep_ranges()
        text = format_frame_ranges(ranges) if ranges else "全フレームを残します。"
        QMessageBox.information(self, "残す範囲一覧", text)

    def clear_keep_ranges(self) -> None:
        if QMessageBox.question(self, "確認", "残す範囲をクリアして全フレームを残しますか？") != QMessageBox.StandardButton.Yes:
            log_user_action("残す範囲クリア キャンセル")
            return
        self.keep_range_start = None
        self.keep_range_end = None
        self.sequence_slider.set_keep_markers(None, None)
        self.set_keep_ranges([])
        log_user_action("残す範囲クリア")
        self.auto_track_status.setText("残す範囲をクリアしました")

    def update_selected_type(self, *args) -> None:
        self.current_row()[f"mosaic{self.selected_slot}_type"] = self.type_input.text()
        log_user_action("選択モザイクtype変更", frame=self.current_frame_no(), slot=self.selected_slot, value=self.type_input.text())
        self.mark_dirty()
        self.refresh_mosaic_table()

    def disable_selected(self, *args) -> None:
        self.current_row()[f"mosaic{self.selected_slot}_on"] = "0"
        set_blank_crotch(self.current_row(), self.selected_slot)
        log_user_action("選択モザイク削除", frame=self.current_frame_no(), slot=self.selected_slot)
        self.mark_dirty()
        self.refresh_mosaic_table()

    def restore_current_frame(self, *args) -> None:
        rows = sorted({index.row() for index in self.frame_table.selectedIndexes()})
        if not rows:
            rows = [self.current_index]
        log_user_action("フレーム復元", rows=rows)
        for row_index in rows:
            if 0 <= row_index < len(self.original_rows):
                self.data.rows[row_index] = dict(self.original_rows[row_index])
                self.update_frame_table_row(row_index, self.data.rows[row_index])
        self.mark_dirty()
        self.load_frame()
        self.refresh_mosaic_table()

    def populate_selected_from_nearest(self, on: bool) -> bool:
        row = self.current_row()
        if get_rect(row, self.selected_slot) is not None:
            return True
        candidate = self.nearest_rect(self.selected_slot) or self.nearest_any_rect()
        if candidate is None:
            return False
        rect, label = candidate
        rect = clamp_rect(rect, self.image_width, self.image_height)
        set_rect(row, self.selected_slot, rect, on=on)
        if label:
            row[f"mosaic{self.selected_slot}_type"] = label
        return True

    def create_from_nearest(self, *args) -> None:
        if self.populate_selected_from_nearest(on=True):
            log_user_action("直近枠から作成", frame=self.current_frame_no(), slot=self.selected_slot)
            self.mark_dirty()
            self.refresh_mosaic_table()

    def nearest_rect(self, slot: int) -> tuple[QRect, str] | None:
        for idx in range(self.current_index - 1, -1, -1):
            row = self.data.rows[idx]
            rect = get_rect(row, slot)
            if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                return QRect(rect), row.get(f"mosaic{slot}_type", "")
        for idx in range(self.current_index + 1, len(self.data.rows)):
            row = self.data.rows[idx]
            rect = get_rect(row, slot)
            if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                return QRect(rect), row.get(f"mosaic{slot}_type", "")
        return None

    def nearest_any_rect(self) -> tuple[QRect, str] | None:
        for idx in range(self.current_index - 1, -1, -1):
            row = self.data.rows[idx]
            for slot in range(1, MAX_MOSAICS + 1):
                rect = get_rect(row, slot)
                if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                    return QRect(rect), row.get(f"mosaic{slot}_type", "")
        for idx in range(self.current_index + 1, len(self.data.rows)):
            row = self.data.rows[idx]
            for slot in range(1, MAX_MOSAICS + 1):
                rect = get_rect(row, slot)
                if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                    return QRect(rect), row.get(f"mosaic{slot}_type", "")
        return None

    def prev_frame(self, *args) -> None:
        if self.current_index > 0:
            self.move_to_index(self.current_index - 1)

    def next_frame(self, *args) -> None:
        if self.current_index < len(self.data.rows) - 1:
            self.move_to_index(self.current_index + 1)

    def skip_frame_count(self) -> int:
        try:
            count = int(self.skip_frame_input.text().strip())
        except ValueError:
            count = 10
        count = max(1, count)
        self.skip_frame_input.setText(str(count))
        return count

    def prev_skip_frame(self, *args) -> None:
        self.move_to_index(max(0, self.current_index - self.skip_frame_count()))

    def next_skip_frame(self, *args) -> None:
        self.move_to_index(min(len(self.data.rows) - 1, self.current_index + self.skip_frame_count()))

    def go_to_frame(self, *args) -> None:
        wanted = self.frame_input.text().strip()
        log_user_action("フレーム番号移動", frame=wanted)
        for idx, row in enumerate(self.data.rows):
            if row.get("frame_no") == wanted:
                self.move_to_index(idx)
                return
        APP_LOGGER.info("指定frame_noがCSVにありません: %s", wanted)
        QMessageBox.information(self, "Not found", f"frame_no={wanted} はCSVにありません")

    def move_to_index(self, target_index: int) -> None:
        if target_index == self.current_index:
            return
        if self.playback_active:
            self.stop_preview_playback()
        self.current_index = target_index
        self.load_frame()

    def frame_at_index(self, index: int) -> np.ndarray | None:
        frame_no_text = self.data.rows[index].get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            return None
        return self.read_frame_number(frame_no, prefer_sequential=False)

    def track_slot_to_index(
        self,
        target_index: int,
        slot: int,
        stop_on_low_confidence: bool = False,
        refresh_anchor_from_current: bool = True,
    ) -> tuple[bool, str]:
        row = self.current_row()
        rect = get_rect(row, slot)
        prev_frame = self.source_frame if self.source_frame is not None else self.frame_at_index(self.current_index)
        if refresh_anchor_from_current and rect is not None and is_on(row.get(f"mosaic{slot}_on")) and prev_frame is not None:
            label = row.get(f"mosaic{slot}_type", "") or "manual"
            self.auto_track_anchors[slot] = (self.current_index, QRect(rect), prev_frame.copy(), label)

        anchor = self.auto_track_anchors.get(slot)
        if anchor is None:
            return False, f"mosaic{slot}: 枠なし"
        anchor_index, anchor_rect, anchor_frame, label = anchor
        gap = target_index - anchor_index - 1
        if gap < 0:
            self.auto_track_anchors.pop(slot, None)
            return False, f"mosaic{slot}: リセット"
        if gap > self.max_interpolate_gap():
            return False, f"mosaic{slot}: gap超過"
        next_frame = self.frame_at_index(target_index)
        if next_frame is None:
            return False, f"mosaic{slot}: フレーム読込失敗"
        result = track_rect(
            anchor_frame,
            next_frame,
            anchor_rect,
            allow_scale=self.trace_scale_enabled(slot),
        )
        if result is None:
            return False, f"mosaic{slot}: 失敗"
        tracked_rect, score, method = result
        if stop_on_low_confidence and not self.track_confident_enough(score, method):
            return False, f"mosaic{slot}: 見失い(score={score:.3f})"
        tracked_rect = clamp_rect(tracked_rect, next_frame.shape[1], next_frame.shape[0])
        span = max(1, target_index - anchor_index)
        for idx in range(anchor_index + 1, target_index + 1):
            t = (idx - anchor_index) / span
            fill_rect = tracked_rect if idx == target_index else interpolate_rect(anchor_rect, tracked_rect, t)
            target_row = self.data.rows[idx]
            set_rect(target_row, slot, clamp_rect(fill_rect, next_frame.shape[1], next_frame.shape[0]), on=True)
            target_row[f"mosaic{slot}_type"] = label
            target_row[f"mosaic{slot}_score"] = f"track:{score:.3f}" if idx == target_index else "track:interpolated"
            self.update_frame_table_row(idx, target_row)
        self.auto_track_anchors[slot] = (target_index, QRect(tracked_rect), next_frame.copy(), label)
        self.mark_dirty()
        frame_no = self.data.rows[target_index].get("frame_no", "")
        gap_text = f", gap={gap}" if gap else ""
        return True, f"mosaic{slot}: frame {frame_no} ({method}, score={score:.3f}{gap_text})"

    def trace_slot_range(self, slot: int) -> None:
        start_frame, end_frame = self.trace_range_for_slot(slot)
        log_user_action("範囲追跡開始", slot=slot, start=start_frame, end=end_frame)
        start_index = self.frame_index_for_no(start_frame)
        end_index = self.frame_index_for_no(end_frame)
        if start_index is None or end_index is None:
            APP_LOGGER.warning("範囲追跡失敗: Start/End frame_no がCSVにありません slot=%s start=%s end=%s", slot, start_frame, end_frame)
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: Start/End frame_no がCSVにありません")
            self.refresh_mosaic_table()
            return
        if end_index <= start_index:
            APP_LOGGER.warning("範囲追跡失敗: End が Start 以前 slot=%s start=%s end=%s", slot, start_frame, end_frame)
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: End は Start より後の frame_no を指定してください")
            self.refresh_mosaic_table()
            return
        start_rect = get_rect(self.data.rows[start_index], slot)
        if start_rect is None or not is_on(self.data.rows[start_index].get(f"mosaic{slot}_on")):
            APP_LOGGER.warning("範囲追跡失敗: Start frame に枠がありません slot=%s start=%s", slot, start_frame)
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: Start frame に枠がありません")
            self.refresh_mosaic_table()
            return

        original_index = self.current_index
        self.current_index = start_index
        self.load_frame()
        self.auto_track_status.setText(f"mosaic{slot}: {start_frame}-{end_frame} を追跡中...")
        QApplication.processEvents()
        stop_message = ""
        success_count = 0
        for target_index in range(start_index + 1, end_index + 1):
            success, message = self.track_slot_to_index(
                target_index,
                slot,
                stop_on_low_confidence=True,
                refresh_anchor_from_current=success_count == 0,
            )
            if not success:
                stop_message = message
                APP_LOGGER.warning("範囲追跡停止: slot=%s message=%s", slot, message)
                break
            success_count += 1
            if success_count % 10 == 0:
                frame_no = self.data.rows[target_index].get("frame_no", "")
                self.auto_track_status.setText(f"mosaic{slot}: 範囲追跡中... frame {frame_no}")
                QApplication.processEvents()
        if not stop_message:
            stop_message = f"mosaic{slot}: 範囲追跡完了"
        self.trace_slots.discard(slot)
        self.auto_track_anchors.pop(slot, None)
        self.auto_track_status.setText(f"{stop_message} / 更新 {success_count} frame")
        log_user_action("範囲追跡終了", slot=slot, updated_frames=success_count, message=stop_message)
        self.current_index = original_index
        self.load_frame()

    def track_confident_enough(self, score: float, method: str) -> bool:
        if "template" not in method:
            return False
        return score >= TRACE_TO_END_MIN_SCORE

    def max_interpolate_gap(self) -> int:
        try:
            return max(0, int(self.data.meta_dict.get("interpolate_gap", "0")))
        except ValueError:
            return 0

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

    def save_with_confirm(self, *args) -> None:
        if self.csv_path is None:
            APP_LOGGER.info("保存不可: レシピ未選択")
            QMessageBox.information(self, "未選択", "レシピが開かれていません。")
            return
        if QMessageBox.question(self, "保存確認", f"{self.csv_path} を上書き保存しますか？") != QMessageBox.StandardButton.Yes:
            log_user_action("保存キャンセル", csv_path=self.csv_path)
            return
        log_user_action("保存", csv_path=self.csv_path)
        try:
            write_pre_csv(self.csv_path, self.data)
        except Exception as exc:
            APP_LOGGER.exception("保存失敗: csv=%s", self.csv_path)
            QMessageBox.critical(self, "Error", f"保存に失敗しました: {exc}")
            return
        self.original_rows = [dict(row) for row in self.data.rows]
        self.dirty = False
        self.setWindowTitle(f"pre CSV Editor - {self.csv_path.name}")
        self.populate_frame_table()

    def save_without_confirm(self) -> None:
        if self.csv_path is None:
            return
        APP_LOGGER.info("確認なし保存: csv=%s", self.csv_path)
        try:
            write_pre_csv(self.csv_path, self.data)
        except Exception:
            APP_LOGGER.exception("確認なし保存失敗: csv=%s", self.csv_path)
            raise
        self.original_rows = [dict(row) for row in self.data.rows]
        self.dirty = False
        self.setWindowTitle(f"pre CSV Editor - {self.csv_path.name}")
        self.populate_frame_table()

    def encode_post(self) -> None:
        if self.csv_path is None:
            APP_LOGGER.info("エンコード不可: レシピ未選択")
            QMessageBox.information(self, "未選択", "レシピが開かれていません。")
            return
        log_user_action("エンコード開始要求", csv_path=self.csv_path)
        try:
            self.save_without_confirm()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"保存に失敗したためエンコードできません: {exc}")
            return
        output_path = post_output_path_from_csv(self.csv_path)
        log_path = output_path.with_name(f"{output_path.stem}_log.txt")
        self.encode_button.setEnabled(False)
        dialog = PostProgressDialog(self.csv_path, output_path, log_path, self)
        dialog.finished.connect(lambda _result: self.encode_button.setEnabled(True))
        dialog.start()
        dialog.exec()

    def closeEvent(self, event) -> None:
        if hasattr(self, "playback_active") and self.playback_active:
            self.stop_preview_playback()
        if self.dirty:
            result = QMessageBox.question(self, "未保存", "未保存の変更があります。閉じますか？")
            if result != QMessageBox.StandardButton.Yes:
                log_user_action("ウィンドウを閉じる キャンセル", csv_path=self.csv_path)
                event.ignore()
                return
        if self.cap is not None:
            self.cap.release()
        log_user_action("ウィンドウを閉じる", csv_path=self.csv_path, dirty=self.dirty)
        event.accept()


def main() -> None:
    log_path = setup_app_logging()
    install_exception_logging()
    parser = argparse.ArgumentParser(description="_pre.csv GUI editor")
    parser.add_argument("input", nargs="?", help="編集する _pre.csv、またはAIレシピを作成する動画")
    args = parser.parse_args()
    APP_LOGGER.info("アプリログファイル: %s", log_path)

    app = LoggingApplication(sys.argv)
    input_path = Path(args.input) if args.input else None
    if input_path is None:
        APP_LOGGER.info("起動モード: 空ウィンドウ")
        window = EditorWindow()
        window.show()
        exit_code = app.exec()
        APP_LOGGER.info("アプリ終了: exit_code=%s", exit_code)
        sys.exit(exit_code)
    if not input_path.is_file():
        APP_LOGGER.error("起動入力ファイルが見つかりません: %s", input_path)
        QMessageBox.critical(None, "Error", f"ファイルが見つかりません: {input_path}")
        return

    if input_path.suffix.lower() == ".csv":
        APP_LOGGER.info("起動モード: CSV編集 input=%s", input_path)
        try:
            window = EditorWindow(input_path)
        except Exception as exc:
            APP_LOGGER.exception("起動時CSV読込失敗: %s", input_path)
            QMessageBox.critical(None, "Error", str(exc))
            return
    else:
        APP_LOGGER.info("起動モード: 動画からレシピ生成 input=%s", input_path)
        window = EditorWindow()
        window.show()
        pre_dialog = PreCreateDialog(input_path)
        if pre_dialog.exec() == QDialog.DialogCode.Accepted and pre_dialog.result_csv is not None:
            window.open_editor_window(pre_dialog.result_csv)
        else:
            window.build_empty_ui()
    window.show()
    exit_code = app.exec()
    APP_LOGGER.info("アプリ終了: exit_code=%s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
