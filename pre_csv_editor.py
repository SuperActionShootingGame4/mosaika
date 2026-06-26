#!/usr/bin/env python3
"""_pre.csv editor for mosaic rectangles."""

from __future__ import annotations

import argparse
import bisect
import csv
import faulthandler
import logging
import math
import platform
import shutil
import subprocess
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
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPoint, QPointF, QRect, QRectF, Qt, QThread, QTimer, pyqtSignal, qInstallMessageHandler
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
    QInputDialog,
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
    QAbstractItemView,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_version import APP_VERSION
from mosaic_censor import (
    BASE_CSV_FIELDS,
    CENSOR_EFFECTS,
    CENSOR_SHAPES,
    MAX_CSV_MOSAICS,
    MOSAIC_CSV_SUFFIXES,
    POSE_BACKENDS,
    SKELETON_EDGES,
    build_edge_mask,
    build_grabcut_mask,
    create_blank_pre_csv,
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

MAX_MOSAICS = 64
HANDLE_SIZE = 8
DEFAULT_INTENSITY_SLIDER_MAX = 100
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15
# ＋/− ボタン1回の拡大縮小量（10パーセントポイント）
ZOOM_BUTTON_STEP = 0.10
DEFAULT_TRACE_TO_END_MIN_SCORE = 0.35
POSE_OVERLAY_MIN_SCORE = 0.3
FRAME_SLIDER_LOAD_DELAY_MS = 35
TRACK_PROXY_MAX_DIM = 640
TRACE_ORIGINAL_TEMPLATE_INTERVAL = 10
# B2: フレーム間追跡スコアがこの値以上なら、ローリング参照との類似度が
# 低くても「同じ対象の見た目変化」とみなしドリフト停止しない。
TRACE_LIVE_TRUST_SCORE = 0.5
# スケールドリフト・ガード: 追跡枠が開始枠サイズに対してこの倍率範囲を
# 外れたら、前景の大きな構造物などへの乗り移りとみなして停止する。
# （T scale OFF 時は枠サイズが変わらないため発火しない）
TRACE_MAX_SCALE_RATIO = 2.0
TRACE_MIN_SCALE_RATIO = 0.5

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

    APP_LOGGER.info("アプリ起動: version=%s argv=%s", APP_VERSION, sys.argv)
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
        if "Fatal" in mode_name:
            APP_LOGGER.error("Qt Fatal 発生時のPython全スレッドスタックを出力します")
            for handler in APP_LOGGER.handlers:
                handler.flush()
            if APP_LOG_FILE_HANDLE is not None:
                faulthandler.dump_traceback(file=APP_LOG_FILE_HANDLE, all_threads=True)
                APP_LOG_FILE_HANDLE.flush()

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
EDITOR_CONFIG_SECTION = "editor"
DISPLAY_SEQUENTIAL_MAX_SKIP = 120
FAST_SEEK_PREROLL_SEC = 2.0
FAST_SEEK_TIMEOUT_SEC = 8.0


class RecipeGenerationCancelled(Exception):
    pass


class EncodingCancelled(Exception):
    pass


def progress_text(current: int, total: int, elapsed: float) -> str:
    if current <= 0 or total <= 0:
        return "準備中..."
    remaining = max(0.0, elapsed * (total - current) / current)
    return f"{current}/{total} frame  残り {time.strftime('%H:%M:%S', time.gmtime(remaining))}"


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def load_recipe_config() -> dict:
    return load_config_section(RECIPE_CONFIG_SECTION)


def load_editor_config() -> dict:
    return load_config_section(EDITOR_CONFIG_SECTION)


def load_config_section(section_name: str) -> dict:
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
    section = data.get(section_name, {})
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
    section = {
        "video_path": settings["video_path"],
        "pose_model": settings["pose_model"],
        "all_frames": settings["all_frames"],
        "start_frame": settings["start_frame"],
        "end_frame": settings["end_frame"],
        "confidence": settings["confidence"],
        "intensity": settings["intensity"],
        "effect": settings["effect"],
        "shape": settings["shape"],
        "detect_every": settings["detect_every"],
        "interpolate_gap": settings["interpolate_gap"],
        "yolo_nsfw_model": settings["yolo_nsfw_model"],
        "yolo_confidence": settings["yolo_confidence"],
        "no_crotch": settings["no_crotch"],
        "skip_no_person": settings["skip_no_person"],
        "blank_recipe": settings["blank_recipe"],
    }
    save_config_section(RECIPE_CONFIG_SECTION, section)


def save_editor_config_value(key: str, value) -> None:
    section = load_editor_config()
    section[key] = value
    save_config_section(EDITOR_CONFIG_SECTION, section)


def save_config_section(section_name: str, values: dict) -> None:
    data = load_all_config()
    data[section_name] = values
    lines: list[str] = []
    for section, section_values in data.items():
        if not isinstance(section_values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in section_values.items():
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_all_config() -> dict:
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
    return data if isinstance(data, dict) else {}


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
        self._last_logged_progress = -1
        APP_LOGGER.info(
            "PreCreateWorker作成: worker_id=%s video=%s csv=%s log=%s",
            id(self),
            self.video_path,
            self.csv_path,
            self.log_path,
        )

    def cancel(self) -> None:
        APP_LOGGER.info("PreCreateWorkerキャンセル要求: worker_id=%s", id(self))
        self.cancel_requested = True

    def emit_progress(self, current: int, total: int, elapsed: float) -> None:
        if self.cancel_requested:
            APP_LOGGER.info(
                "PreCreateWorkerキャンセル検知: worker_id=%s current=%s total=%s elapsed=%.3f",
                id(self),
                current,
                total,
                elapsed,
            )
            raise RecipeGenerationCancelled()
        if current != self._last_logged_progress and (
            current == 0 or current == total or current - self._last_logged_progress >= 100
        ):
            APP_LOGGER.info(
                "レシピ生成進捗: worker_id=%s current=%s total=%s elapsed=%.3f",
                id(self),
                current,
                total,
                elapsed,
            )
            self._last_logged_progress = current
        self.progress.emit(current, total, elapsed)

    def run(self) -> None:
        try:
            APP_LOGGER.info(
                "PreCreateWorker.run開始: worker_id=%s video=%s csv=%s log=%s options=%s",
                id(self),
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
                if self.options["blank_recipe"]:
                    lf.write("空レシピを生成します。\n")
                    create_blank_pre_csv(
                        input_path=str(self.video_path),
                        csv_path=str(self.csv_path),
                        intensity=self.options["intensity"],
                        effect=self.options["effect"],
                        shape=self.options["shape"],
                        confidence=self.options["confidence"],
                        detect_every=self.options["detect_every"],
                        yolo_nsfw_model_path=self.options["yolo_nsfw_model"],
                        yolo_confidence=yolo_confidence,
                        max_interpolate_gap=self.options["interpolate_gap"],
                        frame_range=self.options["frame_range"],
                        pose_backend=self.options["pose_model"],
                        no_crotch=self.options["no_crotch"],
                        skip_no_person=self.options["skip_no_person"],
                        progress_callback=self.emit_progress,
                    )
                else:
                    process_video(
                        input_path=str(self.video_path),
                        output_path=None,
                        intensity=self.options["intensity"],
                        effect=self.options["effect"],
                        shape=self.options["shape"],
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
            APP_LOGGER.info("PreCreateWorker.run完了: worker_id=%s csv=%s", id(self), self.csv_path)
            self.finished.emit(str(self.csv_path))
            APP_LOGGER.info("PreCreateWorker.finished emit済み: worker_id=%s", id(self))
        except RecipeGenerationCancelled:
            APP_LOGGER.info("PreCreateWorker.runキャンセル: worker_id=%s csv=%s", id(self), self.csv_path)
            self.csv_path.unlink(missing_ok=True)
            self.failed.emit("レシピ生成をキャンセルしました。")
            APP_LOGGER.info("PreCreateWorker.failed emit済み: worker_id=%s reason=cancel", id(self))
        except SystemExit as exc:
            APP_LOGGER.exception(
                "PreCreateWorker.runでSystemExit: worker_id=%s video=%s csv=%s code=%s",
                id(self),
                self.video_path,
                self.csv_path,
                exc.code,
            )
            self.failed.emit(f"レシピ生成中に処理が終了しました: {exc.code}")
            APP_LOGGER.info("PreCreateWorker.failed emit済み: worker_id=%s reason=system-exit", id(self))
        except Exception as exc:
            APP_LOGGER.exception(
                "PreCreateWorker.run例外: worker_id=%s video=%s csv=%s",
                id(self),
                self.video_path,
                self.csv_path,
            )
            self.failed.emit(str(exc))
            APP_LOGGER.info("PreCreateWorker.failed emit済み: worker_id=%s reason=exception", id(self))


class PostWorker(QObject):
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, csv_path: Path, output_path: Path, log_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.output_path = output_path
        self.log_path = log_path
        self.cancel_requested = False

    def cancel(self) -> None:
        APP_LOGGER.info("PostWorkerキャンセル要求: csv=%s output=%s", self.csv_path, self.output_path)
        self.cancel_requested = True

    def emit_progress(self, current: int, total: int, elapsed: float) -> None:
        if self.cancel_requested:
            raise EncodingCancelled("エンコードをキャンセルしました。")
        self.progress.emit(current, total, elapsed)

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
                    progress_callback=self.emit_progress,
                )
            APP_LOGGER.info("エンコード完了: output=%s", self.output_path)
            self.finished.emit(str(self.output_path))
        except EncodingCancelled as exc:
            APP_LOGGER.info("エンコードキャンセル: csv=%s output=%s", self.csv_path, self.output_path)
            self.failed.emit(str(exc))
        except Exception as exc:
            APP_LOGGER.exception("エンコードエラー: csv=%s output=%s", self.csv_path, self.output_path)
            self.failed.emit(str(exc))


class PreCreateDialog(QDialog):
    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.destroyed.connect(lambda _obj=None, dialog_id=id(self): APP_LOGGER.info(
            "PreCreateDialog破棄: dialog_id=%s",
            dialog_id,
        ))
        self.config = load_recipe_config()
        self.video_path = video_path
        self.result_csv: Path | None = None
        self.thread: threading.Thread | None = None
        self.worker: PreCreateWorker | None = None
        self.worker_failed = False
        self.running = False
        self.setWindowTitle("レシピ生成ウィンドウ")
        self.resize(520, 360)
        self.total_frames = self.detect_total_frames()
        APP_LOGGER.info(
            "PreCreateDialog作成: dialog_id=%s video=%s total_frames=%s",
            id(self),
            self.video_path,
            self.total_frames,
        )
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
        APP_LOGGER.info("PreCreateDialog UI構築開始: dialog_id=%s", id(self))
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
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(CENSOR_SHAPES))
        shape = config_text(self.config, "shape", "square")
        self.shape_combo.setCurrentText(shape if shape in CENSOR_SHAPES else "square")
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
        self.blank_recipe_check = QCheckBox("空レシピを生成")
        self.blank_recipe_check.setChecked(config_bool(self.config, "blank_recipe", False))
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
        form.addRow("shape", self.shape_combo)
        form.addRow("detect_every ※何フレームごとにAI検出するか", self.detect_every_spin)
        form.addRow("interpolate_gap ※検出漏れを補間する最大フレーム数", self.interpolate_gap_spin)
        form.addRow("yolo_nsfw_model ※未選択時はNudeNetを使用", yolo_layout)
        form.addRow("yolo_confidence", self.yolo_confidence_spin)
        form.addRow("", self.no_crotch_check)
        form.addRow("", self.skip_no_person_check)
        form.addRow("", self.blank_recipe_check)
        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("未実行")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        self.buttons = QDialogButtonBox()
        self.start_button = self.buttons.addButton("レシピ生成", QDialogButtonBox.ButtonRole.ActionRole)
        self.cancel_button = self.buttons.addButton("キャンセル", QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(self.buttons)

        self.all_frames_check.stateChanged.connect(self.update_frame_enabled)
        self.video_path_input.editingFinished.connect(self.update_video_path_from_input)
        self.start_button.clicked.connect(self.start_pre_create)
        self.cancel_button.clicked.connect(self.cancel_or_reject)
        APP_LOGGER.info(
            "PreCreateDialog UI構築完了: dialog_id=%s start_button_role=ActionRole cancel_button_role=RejectRole",
            id(self),
        )

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

    def output_paths(self, blank_recipe: bool = False) -> tuple[Path, Path]:
        range_suffix = ""
        frame_range = self.frame_range()
        if frame_range is not None:
            range_suffix = f"_frames{frame_range[0]}-{frame_range[1]}"
        stem = self.video_path.stem
        kind = "blank_rcp" if blank_recipe else "rcp"
        csv_path = self.video_path.with_name(f"{stem}{range_suffix}_{kind}.csv")
        log_path = self.video_path.with_name(f"{stem}{range_suffix}_{kind}_log.txt")
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
            "shape": self.shape_combo.currentText(),
            "detect_every": self.detect_every_spin.value(),
            "interpolate_gap": self.interpolate_gap_spin.value(),
            "yolo_nsfw_model": self.yolo_model_input.text().strip(),
            "yolo_confidence": self.yolo_confidence_spin.value(),
            "no_crotch": self.no_crotch_check.isChecked(),
            "skip_no_person": self.skip_no_person_check.isChecked(),
            "blank_recipe": self.blank_recipe_check.isChecked(),
        }

    def start_pre_create(self) -> None:
        APP_LOGGER.info(
            "PreCreateDialog.start_pre_create呼び出し: dialog_id=%s running=%s thread=%s worker=%s",
            id(self),
            self.running,
            self.thread.name if self.thread is not None else None,
            id(self.worker) if self.worker is not None else None,
        )
        if self.running:
            APP_LOGGER.warning("レシピ生成二重開始を無視: dialog_id=%s", id(self))
            return
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
        blank_recipe = self.blank_recipe_check.isChecked()
        csv_path, log_path = self.output_paths(blank_recipe)
        if csv_path.exists():
            result = QMessageBox.question(
                self,
                "上書き確認",
                f"{csv_path.name} は既に存在します。上書きしますか？",
            )
            if result != QMessageBox.StandardButton.Yes:
                log_user_action("レシピ生成 上書きキャンセル", csv_path=csv_path)
                return
        yolo_model = self.yolo_model_input.text().strip() or None
        yolo_conf = None if self.yolo_confidence_spin.value() <= 0 else self.yolo_confidence_spin.value()
        options = {
            "pose_model": self.pose_combo.currentText(),
            "frame_range": frame_range,
            "confidence": self.confidence_spin.value(),
            "intensity": self.intensity_spin.value(),
            "effect": self.effect_combo.currentText(),
            "shape": self.shape_combo.currentText(),
            "no_crotch": self.no_crotch_check.isChecked(),
            "detect_every": self.detect_every_spin.value(),
            "interpolate_gap": self.interpolate_gap_spin.value(),
            "yolo_nsfw_model": yolo_model,
            "yolo_confidence": yolo_conf,
            "skip_no_person": self.skip_no_person_check.isChecked(),
            "blank_recipe": blank_recipe,
        }
        self.worker_failed = False
        self.result_csv = None
        self.set_running(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("AIレシピを作成中...")
        self.worker = PreCreateWorker(self.video_path, csv_path, log_path, options)
        worker_id = id(self.worker)
        dialog_id = id(self)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.pre_finished)
        self.worker.failed.connect(self.pre_failed)
        self.worker.finished.connect(self.pre_thread_finished)
        self.worker.failed.connect(self.pre_thread_finished)

        def run_worker() -> None:
            APP_LOGGER.info(
                "PreCreateDialog Pythonスレッド開始: dialog_id=%s thread_name=%s worker_id=%s",
                dialog_id,
                threading.current_thread().name,
                worker_id,
            )
            try:
                self.worker.run()
            finally:
                APP_LOGGER.info(
                    "PreCreateDialog Pythonスレッド終了: dialog_id=%s thread_name=%s worker_id=%s",
                    dialog_id,
                    threading.current_thread().name,
                    worker_id,
                )

        self.thread = threading.Thread(
            target=run_worker,
            name=f"PreCreateWorker-{worker_id}",
            daemon=True,
        )
        APP_LOGGER.info(
            "PreCreateDialog Pythonスレッド開始直前: dialog_id=%s thread_name=%s worker_id=%s",
            dialog_id,
            self.thread.name,
            worker_id,
        )
        self.thread.start()
        APP_LOGGER.info(
            "PreCreateDialog Pythonスレッド開始直後: dialog_id=%s thread_name=%s is_alive=%s",
            dialog_id,
            self.thread.name,
            self.thread.is_alive(),
        )

    def set_running(self, running: bool) -> None:
        APP_LOGGER.info(
            "PreCreateDialog.set_running: dialog_id=%s old=%s new=%s",
            id(self),
            self.running,
            running,
        )
        self.running = running
        for widget in (
            self.pose_combo, self.effect_combo, self.shape_combo, self.confidence_spin, self.intensity_spin,
            self.detect_every_spin, self.interpolate_gap_spin, self.no_crotch_check,
            self.skip_no_person_check, self.blank_recipe_check,
            self.all_frames_check, self.start_spin,
            self.end_spin, self.video_path_input, self.browse_video_button,
            self.yolo_model_input, self.yolo_confidence_spin, self.start_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("キャンセル" if running else "キャンセル")
        if not running:
            self.update_frame_enabled()

    def update_progress(self, current: int, total: int, elapsed: float) -> None:
        if current == 0 or current == total or current % 100 == 0:
            APP_LOGGER.info(
                "PreCreateDialog.update_progress: dialog_id=%s current=%s total=%s elapsed=%.3f",
                id(self),
                current,
                total,
                elapsed,
            )
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        self.progress_label.setText(progress_text(current, total, elapsed))

    def pre_finished(self, csv_path: str) -> None:
        APP_LOGGER.info(
            "PreCreateDialog.pre_finished: dialog_id=%s csv=%s thread=%s",
            id(self),
            csv_path,
            self.thread.name if self.thread is not None else None,
        )
        self.result_csv = Path(csv_path)
        self.progress_label.setText(f"完了: {csv_path}")
        log_user_action("レシピ生成完了", csv_path=csv_path)

    def pre_failed(self, message: str) -> None:
        APP_LOGGER.info(
            "PreCreateDialog.pre_failed: dialog_id=%s message=%s thread=%s",
            id(self),
            message,
            self.thread.name if self.thread is not None else None,
        )
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
        APP_LOGGER.info(
            "PreCreateDialog.pre_thread_finished開始: dialog_id=%s running=%s worker_failed=%s result_csv=%s thread=%s",
            id(self),
            self.running,
            self.worker_failed,
            self.result_csv,
            self.thread.name if self.thread is not None else None,
        )
        self.running = False
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.thread = None
        if self.result_csv is not None and not self.worker_failed:
            APP_LOGGER.info("PreCreateDialog.accept呼び出し: dialog_id=%s", id(self))
            self.accept()
        APP_LOGGER.info("PreCreateDialog.pre_thread_finished終了: dialog_id=%s", id(self))

    def cancel_or_reject(self) -> None:
        APP_LOGGER.info(
            "PreCreateDialog.cancel_or_reject: dialog_id=%s running=%s worker=%s thread=%s",
            id(self),
            self.running,
            id(self.worker) if self.worker is not None else None,
            self.thread.name if self.thread is not None else None,
        )
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
        APP_LOGGER.info("PreCreateDialog.reject: dialog_id=%s running=%s", id(self), self.running)
        if self.running:
            self.cancel_or_reject()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        APP_LOGGER.info("PreCreateDialog.closeEvent: dialog_id=%s running=%s", id(self), self.running)
        if self.running:
            self.cancel_or_reject()
            event.ignore()
            return
        super().closeEvent(event)

    def accept(self) -> None:
        APP_LOGGER.info("PreCreateDialog.accept: dialog_id=%s running=%s result_csv=%s", id(self), self.running, self.result_csv)
        super().accept()

    def done(self, result: int) -> None:
        APP_LOGGER.info("PreCreateDialog.done: dialog_id=%s result=%s running=%s", id(self), result, self.running)
        super().done(result)


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
        self.cancel_requested = False
        self.running = False
        self.setWindowTitle("エンコード")
        self.resize(460, 140)
        layout = QVBoxLayout(self)
        self.label = QLabel("エンコード準備中...")
        self.progress_bar = QProgressBar()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.rejected.connect(self.request_cancel)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.buttons)

    def set_stage_progress(self, message: str, current: int, total: int, started_at: float | None = None) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        percent = int(current * 100 / max(1, total))
        if started_at is None or current <= 0:
            self.label.setText(f"{message} {percent}%")
        else:
            elapsed = time.monotonic() - started_at
            remaining = elapsed * (total - current) / current
            self.label.setText(
                f"{message} {percent}%  残り "
                f"{time.strftime('%H:%M:%S', time.gmtime(max(0, remaining)))}"
            )
        QApplication.processEvents()

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.label.setText("キャンセル中...")
        self.buttons.setEnabled(False)
        if self.worker is not None:
            self.worker.cancel()
        QApplication.processEvents()

    def start(self) -> None:
        self.running = True
        log_user_action("エンコードダイアログ開始", csv_path=self.csv_path, output_path=self.output_path)
        self.set_stage_progress("エンコード準備中...", 0, 1)
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
        if self.cancel_requested and self.worker is not None:
            self.worker.cancel()
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        self.label.setText(f"エンコード中... {progress_text(current, total, elapsed)}")

    def post_finished(self, output_path: str) -> None:
        self.finished_output = output_path

    def post_failed(self, message: str) -> None:
        self.failed_message = message

    def post_thread_finished(self) -> None:
        self.running = False
        if self.failed_message:
            if self.cancel_requested or "キャンセル" in self.failed_message:
                APP_LOGGER.info("エンコードキャンセル表示: %s", self.failed_message)
                QMessageBox.information(self, "キャンセル", self.failed_message)
            else:
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
            self.request_cancel()
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


def read_pre_csv(path: Path, progress_callback=None) -> CsvData:
    meta: list[list[str]] = []
    total_size = max(1, path.stat().st_size)
    last_progress = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "frame_no":
                fieldnames = row
                rows: list[dict[str, str]] = []
                dict_reader = csv.DictReader(f, fieldnames=fieldnames)
                for data_row in dict_reader:
                    rows.append(data_row)
                    if progress_callback is not None:
                        current = min(total_size, f.buffer.tell())
                        if current - last_progress >= total_size // 200 or current == total_size:
                            progress_callback(current, total_size)
                            last_progress = current
                if progress_callback is not None:
                    progress_callback(total_size, total_size)
                return CsvData(meta=meta, fieldnames=fieldnames, rows=rows)
            meta.append(row)
    raise RuntimeError("frame_no ヘッダ行が見つかりません")


def compact_pre_csv_fieldnames(data: CsvData, progress_callback=None) -> list[str]:
    max_slot = 0
    total = max(1, len(data.rows) * 2)
    for row_index, row in enumerate(data.rows):
        for slot in range(1, MAX_CSV_MOSAICS + 1):
            if is_on(row.get(f"mosaic{slot}_on")) or get_rect(row, slot) is not None:
                max_slot = max(max_slot, slot)
                continue
            if any((row.get(f"mosaic{slot}_{suffix}") or "").strip() for suffix in MOSAIC_CSV_SUFFIXES if suffix != "on"):
                max_slot = max(max_slot, slot)
        if progress_callback is not None and (row_index % 20 == 0 or row_index + 1 == len(data.rows)):
            progress_callback(row_index + 1, total)

    fieldnames = list(BASE_CSV_FIELDS)
    for slot in range(1, max_slot + 1):
        fieldnames.extend(f"mosaic{slot}_{suffix}" for suffix in MOSAIC_CSV_SUFFIXES)

    known = set(fieldnames)
    for fieldname in data.fieldnames:
        if fieldname not in known and not fieldname.startswith("mosaic"):
            fieldnames.append(fieldname)
            known.add(fieldname)
    return fieldnames


def write_pre_csv(path: Path, data: CsvData, progress_callback=None) -> None:
    row_count = len(data.rows)
    total = max(1, row_count * 2)
    fieldnames = compact_pre_csv_fieldnames(data, progress_callback=progress_callback)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data.meta)
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        dict_writer.writeheader()
        for idx, row in enumerate(data.rows):
            dict_writer.writerow(row)
            if progress_callback is not None and (idx % 20 == 0 or idx + 1 == row_count):
                progress_callback(row_count + idx + 1, total)
    data.fieldnames = fieldnames


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


def csrt_available() -> bool:
    """opencv-contrib の CSRT トラッカーが使えるか。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return True
    legacy = getattr(cv2, "legacy", None)
    return legacy is not None and hasattr(legacy, "TrackerCSRT_create")


def create_csrt_tracker():
    """CSRT トラッカーを生成する（OpenCV のバージョン差を吸収）。無ければ None。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    return None


class FrameTableModel(QAbstractTableModel):
    HEADERS = ["frame_no", "modify", "Keep", "Persons", "Mosaic", "Crotch", "comment"]

    def __init__(self, editor: "EditorWindow") -> None:
        super().__init__(editor)
        self.editor = editor
        self._keep_ranges: list[tuple[int, int]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.editor.data.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return str(section + 1)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if index.column() == 6:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row_index = index.row()
        if row_index < 0 or row_index >= len(self.editor.data.rows):
            return None
        row = self.editor.data.rows[row_index]
        mosaic_count = enabled_mosaic_count(row)

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            nsfw_detection_count = row.get("nsfw_detection_count") or "?"
            values = [
                row.get("frame_no", ""),
                "T" if self.editor.row_modified(row_index) else "F",
                "keep" if self.editor.frame_is_kept(row, self._keep_ranges) else "cut",
                self.editor.person_count_for_row(row),
                f"{mosaic_count}/{nsfw_detection_count}",
                "yes" if is_on(row.get("crotch_detected")) else "none",
                row.get("comment", ""),
            ]
            return values[index.column()]

        if role == Qt.ItemDataRole.BackgroundRole:
            if not self.editor.frame_is_kept(row, self._keep_ranges):
                return QColor(205, 205, 205)
            return QColor(255, 220, 230) if mosaic_count else QColor(232, 232, 232)

        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or not index.isValid() or index.column() != 6:
            return False
        row_index = index.row()
        if row_index < 0 or row_index >= len(self.editor.data.rows):
            return False
        self.editor.data.rows[row_index]["comment"] = str(value).strip()
        log_user_action(
            "フレームコメント変更",
            row=row_index,
            frame=self.editor.data.rows[row_index].get("frame_no", ""),
        )
        self.editor.mark_dirty()
        self.refresh_row(row_index)
        return True

    def refresh_all(self) -> None:
        self._keep_ranges = self.editor.keep_ranges()
        if self.rowCount() == 0:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, [
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
            Qt.ItemDataRole.BackgroundRole,
        ])

    def refresh_row(self, row_index: int) -> None:
        if row_index < 0 or row_index >= self.rowCount():
            return
        self.refresh_rows(row_index, row_index)

    def refresh_rows(self, start_row: int, end_row: int) -> None:
        if self.rowCount() == 0:
            return
        start_row = max(0, min(self.rowCount() - 1, start_row))
        end_row = max(0, min(self.rowCount() - 1, end_row))
        if end_row < start_row:
            start_row, end_row = end_row, start_row
        top_left = self.index(start_row, 0)
        bottom_right = self.index(end_row, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, [
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
            Qt.ItemDataRole.BackgroundRole,
        ])


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


def scale_rect(rect: QRect, scale: float) -> QRect:
    return QRect(
        round(rect.left() * scale),
        round(rect.top() * scale),
        max(1, round(rect.width() * scale)),
        max(1, round(rect.height() * scale)),
    )


def track_rect_proxy(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    rect: QRect,
    allow_scale: bool = True,
    max_dim: int = TRACK_PROXY_MAX_DIM,
) -> tuple[QRect, float, str] | None:
    height, width = prev_frame.shape[:2]
    scale = min(1.0, max_dim / max(1, max(width, height)))
    if scale >= 0.999:
        return track_rect(prev_frame, next_frame, rect, allow_scale=allow_scale)

    proxy_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    prev_proxy = cv2.resize(prev_frame, proxy_size, interpolation=cv2.INTER_AREA)
    next_proxy = cv2.resize(next_frame, proxy_size, interpolation=cv2.INTER_AREA)
    proxy_rect = scale_rect(rect, scale)
    result = track_rect(prev_proxy, next_proxy, proxy_rect, allow_scale=allow_scale)
    if result is None:
        return None
    tracked_proxy_rect, score, method = result
    inv_scale = 1.0 / scale
    tracked_rect = scale_rect(tracked_proxy_rect, inv_scale)
    return tracked_rect, score, f"{method}:proxy"


def template_similarity_score(
    template_frame: np.ndarray,
    template_rect: QRect,
    target_frame: np.ndarray,
    target_rect: QRect,
) -> float | None:
    template_rect = template_rect.normalized()
    target_rect = target_rect.normalized()
    template_x, template_y, template_w, template_h = rect_to_xywh(template_rect)
    target_x, target_y, target_w, target_h = rect_to_xywh(target_rect)
    if min(template_w, template_h, target_w, target_h) < 4:
        return None
    template_frame_h, template_frame_w = template_frame.shape[:2]
    target_frame_h, target_frame_w = target_frame.shape[:2]
    if (
        template_x < 0
        or template_y < 0
        or template_x + template_w > template_frame_w
        or template_y + template_h > template_frame_h
        or target_x < 0
        or target_y < 0
        or target_x + target_w > target_frame_w
        or target_y + target_h > target_frame_h
    ):
        return None

    template = cv2.cvtColor(
        template_frame[template_y:template_y + template_h, template_x:template_x + template_w],
        cv2.COLOR_BGR2GRAY,
    )
    target = cv2.cvtColor(
        target_frame[target_y:target_y + target_h, target_x:target_x + target_w],
        cv2.COLOR_BGR2GRAY,
    )
    if template.size == 0 or target.size == 0:
        return None
    resized_template = cv2.resize(template, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if float(resized_template.std()) < 1.0:
        return None
    if float(target.std()) < 1.0:
        return 0.0
    result = cv2.matchTemplate(target, resized_template, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(result)
    return float(score) if np.isfinite(score) else None


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
    shape: str,
) -> None:
    x1, y1 = rect.left(), rect.top()
    x2, y2 = rect.right() + 1, rect.bottom() + 1
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    if effect == "blur":
        kernel_size = intensity if intensity % 2 == 1 else intensity + 1
        processed = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    else:
        small = cv2.resize(
            roi,
            (max(1, w // intensity), max(1, h // intensity)),
            interpolation=cv2.INTER_LINEAR,
        )
        processed = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )
    if shape == "circle":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (max(0, w // 2), max(0, h // 2)),
                    (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
        roi[mask > 0] = processed[mask > 0]
    elif shape == "edge":
        mask = build_edge_mask(roi)
        if mask is None:
            # エッジが取れないフレームは矩形全体にかけて検閲漏れを防ぐ。
            frame[y1:y2, x1:x2] = processed
        else:
            roi[mask > 0] = processed[mask > 0]
    elif shape == "grabcut":
        mask = build_grabcut_mask(roi)
        if mask is None:
            # 前景が取れないフレームは矩形全体にかけて検閲漏れを防ぐ。
            frame[y1:y2, x1:x2] = processed
        else:
            roi[mask > 0] = processed[mask > 0]
    else:
        frame[y1:y2, x1:x2] = processed


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
        self.person_ranges: list[tuple[int, int, int]] = []
        self.modified_ranges: list[tuple[int, int]] = []
        self.mosaic_ranges: list[tuple[int, int]] = []
        self.keep_start_marker: int | None = None
        self.keep_end_marker: int | None = None
        self.jump_dragging = False
        self.setTracking(True)
        self.setMinimumHeight(56)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: transparent;
            }
            QSlider::handle:horizontal {
                width: 1px;
                margin: -9px 0;
                background: #0068c9;
                border: 1px solid #004b91;
            }
            """
        )

    def set_keep_ranges(self, keep_ranges: list[tuple[int, int]]) -> None:
        self.keep_ranges = keep_ranges
        self.update()

    def set_person_ranges(self, person_ranges: list[tuple[int, int, int]]) -> None:
        self.person_ranges = person_ranges
        self.update()

    def set_modified_ranges(self, modified_ranges: list[tuple[int, int]]) -> None:
        self.modified_ranges = modified_ranges
        self.update()

    def set_mosaic_ranges(self, mosaic_ranges: list[tuple[int, int]]) -> None:
        self.mosaic_ranges = mosaic_ranges
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

    def frame_tick_step(self) -> int:
        if not self.frame_numbers:
            return 100
        span = max(1, self.frame_numbers[-1] - self.frame_numbers[0])
        rough_step = max(1, span / 10)
        magnitude = 10 ** math.floor(math.log10(rough_step))
        for multiplier in (1, 2, 5, 10):
            step = int(multiplier * magnitude)
            if step >= rough_step:
                return max(1, step)
        return max(1, int(10 * magnitude))

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
        tick_step = self.frame_tick_step()
        first_tick = ((first_frame + tick_step - 1) // tick_step) * tick_step
        tick_frames = [first_frame]
        tick_frames.extend(range(first_tick, last_frame + 1, tick_step))
        if tick_frames[-1] != last_frame:
            tick_frames.append(last_frame)
        for frame_no in dict.fromkeys(tick_frames):
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
        self.paint_person_ranges(painter, option, handle)
        self.paint_mosaic_ranges(painter, option, handle)
        self.paint_modified_ranges(painter, option, handle)
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

    def paint_person_ranges(self, painter: QPainter, option: QStyleOptionSlider, handle: QRect) -> None:
        if not self.frame_numbers or not self.person_ranges:
            return
        slider_max = max(1, self.width() - handle.width())
        left_offset = handle.width() // 2

        def frame_to_x(frame_no: int) -> int:
            index = self.tick_index(frame_no)
            position = QStyle.sliderPositionFromValue(
                self.minimum(),
                self.maximum(),
                max(self.minimum(), min(self.maximum(), index)),
                slider_max,
                upsideDown=option.upsideDown,
            )
            return position + left_offset

        color = QColor(20, 120, 70)
        painter.setPen(QPen(color, 2))
        y = 10
        for start_frame, end_frame, person_count in self.person_ranges:
            left = frame_to_x(start_frame)
            right = frame_to_x(end_frame)
            if right < left:
                left, right = right, left
            label = "P" if person_count == 1 else f"{person_count}P"
            painter.drawText(max(0, left - 10), 0, 34, 10, Qt.AlignmentFlag.AlignLeft, label)
            line_left = min(self.width() - 1, left + 14)
            line_right = max(line_left + 8, right)
            painter.drawLine(line_left, y, min(self.width() - 1, line_right), y)

    def paint_mosaic_ranges(self, painter: QPainter, option: QStyleOptionSlider, handle: QRect) -> None:
        if not self.frame_numbers or not self.mosaic_ranges:
            return
        slider_max = max(1, self.width() - handle.width())
        left_offset = handle.width() // 2
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

        def frame_to_x(frame_no: int) -> int:
            index = self.tick_index(frame_no)
            position = QStyle.sliderPositionFromValue(
                self.minimum(),
                self.maximum(),
                max(self.minimum(), min(self.maximum(), index)),
                slider_max,
                upsideDown=option.upsideDown,
            )
            return position + left_offset

        painter.setPen(QPen(QColor(30, 180, 60), 3))
        y = min(self.height() - 1, groove.bottom() + 7)
        for start_frame, end_frame in self.mosaic_ranges:
            left = frame_to_x(start_frame)
            right = frame_to_x(end_frame)
            if right < left:
                left, right = right, left
            painter.drawLine(left, y, max(left + 2, right), y)

    def paint_modified_ranges(self, painter: QPainter, option: QStyleOptionSlider, handle: QRect) -> None:
        if not self.frame_numbers or not self.modified_ranges:
            return
        slider_max = max(1, self.width() - handle.width())
        left_offset = handle.width() // 2
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

        def frame_to_x(frame_no: int) -> int:
            index = self.tick_index(frame_no)
            position = QStyle.sliderPositionFromValue(
                self.minimum(),
                self.maximum(),
                max(self.minimum(), min(self.maximum(), index)),
                slider_max,
                upsideDown=option.upsideDown,
            )
            return position + left_offset

        painter.setPen(QPen(QColor(220, 40, 40), 3))
        y = min(self.height() - 1, groove.bottom() + 3)
        for start_frame, end_frame in self.modified_ranges:
            left = frame_to_x(start_frame)
            right = frame_to_x(end_frame)
            if right < left:
                left, right = right, left
            painter.drawLine(left, y, max(left + 2, right), y)

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
            self.jump_dragging = True
            self.set_value_from_mouse_x(event.position().x(), option, handle)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.jump_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                option,
                QStyle.SubControl.SC_SliderHandle,
                self,
            )
            self.set_value_from_mouse_x(event.position().x(), option, handle)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self.jump_dragging = False
        super().mouseReleaseEvent(event)

    def set_value_from_mouse_x(self, x: float, option: QStyleOptionSlider, handle: QRect) -> None:
        slider_max = max(0, self.width() - handle.width())
        position = round(x - handle.width() / 2)
        self.setValue(
            QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                position,
                slider_max,
                upsideDown=option.upsideDown,
            )
        )


class CanvasScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.zoom_overlay: QWidget | None = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        canvas = self.widget()
        if canvas is not None and hasattr(canvas, "update_canvas_size"):
            canvas.update_canvas_size()
        self.position_zoom_overlay()

    def position_zoom_overlay(self) -> None:
        if self.zoom_overlay is None:
            return
        self.zoom_overlay.adjustSize()
        margin = 8
        viewport = self.viewport()
        x = viewport.width() - self.zoom_overlay.width() - margin
        y = viewport.height() - self.zoom_overlay.height() - margin
        self.zoom_overlay.move(max(0, x), max(0, y))
        self.zoom_overlay.raise_()


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
        shape = self.parent_window.data.meta_dict.get("shape", "square")
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
            if shape == "circle":
                painter.drawEllipse(wrect)
            else:
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
            # ドラッグ中は全フレーム走査のスライダーマーカー再計算をスキップし、
            # 確定時（mouseReleaseEvent）に一度だけ更新する。
            self.parent_window.mark_dirty(refresh_markers=False)
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
            # ドラッグ中はスキップしていたスライダーマーカーをここで一度だけ更新。
            self.parent_window.refresh_modified_markers()
        self.drag_mode = None

    def wheelEvent(self, event) -> None:
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.apply_zoom(self.zoom_factor * ZOOM_STEP)
        elif event.angleDelta().y() < 0:
            self.apply_zoom(self.zoom_factor / ZOOM_STEP)
        event.accept()

    def apply_zoom(self, new_zoom: float) -> None:
        self.zoom_factor = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
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
        if isinstance(self.scroll_area, CanvasScrollArea):
            self.scroll_area.position_zoom_overlay()
        log_user_action("表示ズーム変更", zoom=round(self.zoom_factor, 3))


class EditorWindow(QMainWindow):
    def __init__(self, csv_path: Path | None = None) -> None:
        super().__init__()
        APP_LOGGER.info("編集ウィンドウ初期化: csv_path=%s", csv_path)
        self.csv_path: Path | None = None
        self.progress_dialog: QProgressDialog | None = None
        self.dirty = False
        self.editor_windows: list[EditorWindow] = []
        self.cap = None
        self.setWindowTitle("pre CSV Editor")
        self.resize(1680, 980)
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
        progress = self.reusable_progress_dialog("読み込み中", "レシピファイルを読み込み中...", 100)
        progress.setRange(0, 100)
        progress.setValue(0)
        QApplication.processEvents()

        def set_load_progress(message: str, value: int) -> None:
            progress.setLabelText(message)
            progress.setValue(max(0, min(100, value)))
            progress.show()
            QApplication.processEvents()

        if hasattr(self, "playback_active") and self.playback_active:
            self.stop_preview_playback()
        if hasattr(self, "slider_load_timer"):
            self.slider_load_timer.stop()
            self.pending_slider_index = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        old_central = self.centralWidget()
        if old_central is not None:
            old_central.deleteLater()
        self.menuBar().clear()
        self.csv_path = csv_path
        set_load_progress("レシピファイルを読み込み中...", 1)

        def on_csv_progress(current: int, total: int) -> None:
            ratio = current / max(1, total)
            set_load_progress(f"レシピファイルを読み込み中... {int(ratio * 100)}%", 1 + round(ratio * 69))

        self.data = read_pre_csv(csv_path, progress_callback=on_csv_progress)
        set_load_progress("読み込みデータを準備中...", 72)
        self.original_rows = [dict(row) for row in self.data.rows]
        set_load_progress("元動画を確認中...", 76)
        self.video_path = source_video_path(self.data, self.csv_path)
        if not self.video_path.is_file():
            APP_LOGGER.error("元動画が見つかりません: csv=%s video=%s", self.csv_path, self.video_path)
            raise RuntimeError(f"元動画が見つかりません: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            APP_LOGGER.error("動画を開けません: csv=%s video=%s", self.csv_path, self.video_path)
            raise RuntimeError(f"動画を開けません: {self.video_path}")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        self.trace_min_score = config_float(
            load_editor_config(),
            "trace_min_score",
            DEFAULT_TRACE_TO_END_MIN_SCORE,
            0.0,
            1.0,
        )
        self.csrt_trace = config_bool(load_editor_config(), "csrt_trace", True)
        self.auto_track_anchors: dict[int, tuple[int, QRect, np.ndarray, str]] = {}
        self.pose_model_bundle = None
        self.pose_model_error: str | None = None
        self.pose_overlay_cache: dict[int, tuple[list[tuple[int, int, int, int]], list[np.ndarray]]] = {}
        self.keep_range_start: int | None = None
        self.keep_range_end: int | None = None
        self.display_cap_pos: int | None = None
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_next_frame)
        self.slider_load_timer = QTimer(self)
        self.slider_load_timer.setSingleShot(True)
        self.slider_load_timer.timeout.connect(self.load_pending_slider_frame)
        self.pending_slider_index: int | None = None
        self.playback_index: int | None = None
        self.playback_cap_pos: int | None = None
        self.playback_active = False

        self.setWindowTitle(f"pre CSV Editor - {csv_path.name}")
        set_load_progress("画面を構築中...", 82)
        self.build_ui(show_load_progress=True)
        set_load_progress("先頭フレームを読み込み中...", 96)
        self.load_frame()
        set_load_progress("読み込み完了", 100)
        progress.hide()
        self.remember_recipe_path(csv_path)
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
        delete_mosaic_action.setShortcut(QKeySequence(Qt.Key.Key_Backspace))
        delete_mosaic_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        delete_mosaic_action.triggered.connect(self.disable_selected)
        self.addAction(delete_mosaic_action)

        clear_frame_action = QAction("選択フレームをクリア", self)
        clear_frame_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        clear_frame_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        clear_frame_action.triggered.connect(self.clear_selected_frames)
        self.addAction(clear_frame_action)

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
        self.build_zoom_overlay()
        self.frame_label = QLabel()
        self.preview_checkbox = QCheckBox("mosaicプレビュー")
        self.preview_checkbox.setChecked(True)
        self.crotch_overlay_checkbox = QCheckBox("Crotch")
        self.crotch_overlay_checkbox.setChecked(True)
        self.skeleton_overlay_checkbox = QCheckBox("Skeleton")
        self.skeleton_overlay_checkbox.setChecked(True)
        frame_status_layout = QHBoxLayout()
        frame_status_layout.addWidget(self.frame_label)
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
        self.sequence_slider.set_person_ranges(self.person_ranges())
        self.sequence_slider.set_modified_ranges(self.modified_ranges())
        self.sequence_slider.set_mosaic_ranges(self.mosaic_active_ranges())
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
        self.effect_combo.addItems(list(CENSOR_EFFECTS))
        effect = meta.get("effect", "mosaic")
        self.effect_combo.setCurrentText(effect if effect in CENSOR_EFFECTS else "mosaic")
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(CENSOR_SHAPES))
        shape = meta.get("shape", "square")
        self.shape_combo.setCurrentText(shape if shape in CENSOR_SHAPES else "square")
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
        self.trace_min_score_spin = QDoubleSpinBox()
        self.trace_min_score_spin.setRange(0.0, 1.0)
        self.trace_min_score_spin.setDecimals(2)
        self.trace_min_score_spin.setSingleStep(0.05)
        self.trace_min_score_spin.setFixedWidth(70)
        self.trace_min_score_spin.setValue(self.trace_min_score)
        self.csrt_trace_check = QCheckBox("CSRT追跡（白い車など低コントラスト対象に強い）")
        self.csrt_trace_check.setChecked(self.csrt_trace and csrt_available())
        self.csrt_trace_check.setEnabled(csrt_available())
        if not csrt_available():
            self.csrt_trace_check.setToolTip("opencv-contrib-python が必要です")
        meta_layout.addRow("intensity", intensity_layout)
        meta_layout.addRow("effect", self.effect_combo)
        meta_layout.addRow("shape", self.shape_combo)
        meta_layout.addRow("Trace min score", self.trace_min_score_spin)
        meta_layout.addRow("", self.csrt_trace_check)
        for key in ("confidence", "pose_model", "yolo_nsfw_model", "interpolate_gap", "no_crotch", "skip_no_person"):
            meta_layout.addRow(key, QLabel(meta.get(key, "")))
        right_layout.addLayout(meta_layout)

        self.frame_table_model = FrameTableModel(self)
        self.frame_table = QTableView()
        self.frame_table.setModel(self.frame_table_model)
        self.frame_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        self.frame_table.verticalHeader().setVisible(False)
        for col, width in enumerate((86, 56, 56, 64, 76, 64, 180)):
            self.frame_table.setColumnWidth(col, width)
        # Keep 列（col 2）はユーザーが普段見ないため非表示にする。
        self.frame_table.setColumnHidden(FrameTableModel.HEADERS.index("Keep"), True)
        right_layout.addWidget(QLabel("CSV行"))
        right_layout.addWidget(self.frame_table, stretch=1)

        # モザイクマトリクスはフレーム行選択マトリクスのすぐ下に配置する。
        self.mosaic_table = QTableWidget(1, 14)
        self.mosaic_table.setHorizontalHeaderLabels(
            [
                "mosaic", "Trace", "T scale", "Start", "End", "type", "score",
                "w", "h", "x1", "y1", "x2", "y2", "comment",
            ]
        )
        self.mosaic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mosaic_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.mosaic_table, stretch=1)

        self.auto_track_status = QLabel("")
        self.auto_track_status.setWordWrap(True)
        self.create_from_nearest_button = QPushButton("直近枠から作成")
        self.restore_frame_button = QPushButton("選択したフレームを元に戻す(複数可能)")
        right_layout.addWidget(self.auto_track_status)
        right_layout.addWidget(self.create_from_nearest_button)
        right_layout.addWidget(self.restore_frame_button)

        splitter.addWidget(right)
        splitter.setSizes([1100, 580])

        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.prev_skip_button.clicked.connect(self.prev_skip_frame)
        self.next_skip_button.clicked.connect(self.next_skip_frame)
        self.go_button.clicked.connect(self.go_to_frame)
        self.sequence_slider.valueChanged.connect(self.select_sequence_frame)
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
        self.shape_combo.currentTextChanged.connect(self.update_preview_meta)
        self.intensity_slider.valueChanged.connect(self.intensity_spin.setValue)
        self.intensity_spin.valueChanged.connect(self.sync_intensity_slider)
        self.intensity_spin.valueChanged.connect(self.update_preview_meta)
        self.trace_min_score_spin.valueChanged.connect(self.update_trace_min_score)
        self.csrt_trace_check.toggled.connect(self.update_csrt_trace)
        self.mosaic_table.cellClicked.connect(self.select_mosaic)
        self.mosaic_table.itemChanged.connect(self.update_mosaic_from_table)
        self.frame_table.clicked.connect(self.select_frame_row)
        self.frame_table.selectionModel().selectionChanged.connect(self.select_selected_frame_row)
        self.populate_frame_table(
            show_progress=show_load_progress,
            progress_message="レシピファイルを読み込み中...",
        )

    def build_menu(self) -> None:
        menu = self.menuBar().addMenu("ファイル")
        operation_menu = self.menuBar().addMenu("操作")

        version_action = QAction(f"version {APP_VERSION}", self)
        version_action.setEnabled(False)
        self.menuBar().addAction(version_action)

        create_action = QAction("レシピ生成", self)
        create_action.triggered.connect(self.create_recipe_from_menu)
        operation_menu.addAction(create_action)

        encode_action = QAction("エンコード", self)
        encode_action.triggered.connect(self.encode_post)
        encode_action.setEnabled(self.csv_path is not None)
        operation_menu.addAction(encode_action)

        open_action = QAction("レシピを開く", self)
        open_action.triggered.connect(self.open_recipe_from_menu)
        menu.addAction(open_action)

        last_csv = self.last_recipe_path()
        reopen_action = QAction("前回のレシピを開く", self)
        reopen_action.triggered.connect(self.open_last_recipe)
        reopen_action.setEnabled(last_csv is not None)
        if last_csv is not None:
            reopen_action.setToolTip(str(last_csv))
        menu.addAction(reopen_action)

        save_action = QAction("レシピを保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_with_confirm)
        save_action.setEnabled(self.csv_path is not None)
        menu.addAction(save_action)

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
        last_csv = self.last_recipe_path()
        start_dir = str(last_csv.parent) if last_csv is not None else ""
        selected, _ = QFileDialog.getOpenFileName(self, "_pre.csv を選択", start_dir, "CSV (*.csv)")
        if selected:
            log_user_action("レシピ選択", path=selected)
            self.open_editor_window(Path(selected))

    def last_recipe_path(self) -> Path | None:
        path_text = config_text(load_editor_config(), "last_recipe_path", "").strip()
        if not path_text:
            return None
        path = Path(path_text).expanduser()
        return path if path.is_file() else None

    def remember_recipe_path(self, csv_path: Path) -> None:
        try:
            save_editor_config_value("last_recipe_path", str(csv_path.resolve()))
        except Exception:
            APP_LOGGER.exception("前回レシピパスを保存できません: %s", csv_path)

    def open_last_recipe(self) -> None:
        last_csv = self.last_recipe_path()
        if last_csv is None:
            QMessageBox.information(self, "未設定", "前回開いたレシピが見つかりません。")
            return
        log_user_action("前回レシピを開く", path=last_csv)
        self.open_editor_window(last_csv)

    def open_editor_window(self, csv_path: Path) -> None:
        log_user_action("レシピを開く", csv_path=csv_path)
        if self.csv_path is None:
            try:
                self.load_csv(csv_path)
                self.remember_recipe_path(csv_path)
            except Exception as exc:
                APP_LOGGER.exception("レシピを開けません: %s", csv_path)
                QMessageBox.critical(self, "Error", str(exc))
            return
        try:
            window = EditorWindow(csv_path)
            self.remember_recipe_path(csv_path)
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

    def build_zoom_overlay(self) -> None:
        """動画プレビュー右下に浮かぶズーム操作（＋ − リセット ％）を作る。"""
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(46)
        zoom_in_button = QPushButton("＋")
        zoom_out_button = QPushButton("－")
        zoom_reset_button = QPushButton("リセット")
        zoom_in_button.setFixedWidth(30)
        zoom_out_button.setFixedWidth(30)
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_reset_button.clicked.connect(self.zoom_reset)
        overlay = QWidget(self.canvas_scroll_area.viewport())
        layout = QHBoxLayout(overlay)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        layout.addWidget(zoom_in_button)
        layout.addWidget(zoom_out_button)
        layout.addWidget(zoom_reset_button)
        layout.addWidget(self.zoom_label)
        overlay.setStyleSheet(
            "QWidget { background-color: rgba(30, 30, 30, 190); border-radius: 6px; }"
            "QLabel { color: white; background: transparent; }"
            "QPushButton { color: white; }"
        )
        self.zoom_overlay = overlay
        self.canvas_scroll_area.zoom_overlay = overlay
        self.canvas_scroll_area.position_zoom_overlay()
        overlay.show()

    def zoom_in(self) -> None:
        self.canvas.apply_zoom(round(self.canvas.zoom_factor + ZOOM_BUTTON_STEP, 2))

    def zoom_out(self) -> None:
        self.canvas.apply_zoom(round(self.canvas.zoom_factor - ZOOM_BUTTON_STEP, 2))

    def zoom_reset(self) -> None:
        self.canvas.apply_zoom(1.0)

    def update_zoom_label(self, zoom_factor: float) -> None:
        self.zoom_label.setText(f"{round(zoom_factor * 100)}%")

    def mark_dirty(self, refresh_markers: bool = True) -> None:
        self.dirty = True
        if not self.windowTitle().endswith("*"):
            self.setWindowTitle(self.windowTitle() + " *")
        if refresh_markers:
            self.refresh_modified_markers()

    def reusable_progress_dialog(self, title: str, label: str, maximum: int, cancelable: bool = False) -> QProgressDialog:
        if self.progress_dialog is None:
            self.progress_dialog = QProgressDialog(self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
        progress = self.progress_dialog
        progress.reset()
        if cancelable:
            progress.setCancelButtonText("Cancel")
        else:
            progress.setCancelButton(None)
        progress.setWindowTitle(title)
        progress.setLabelText(label)
        progress.setRange(0, max(1, maximum))
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        return progress

    def update_progress_dialog(
        self,
        progress: QProgressDialog,
        message: str,
        done: int,
        total: int,
        started_at: float,
    ) -> None:
        percent = int(done * 100 / max(1, total))
        elapsed = time.monotonic() - started_at
        remaining = elapsed * (total - done) / done if done else 0
        progress.setLabelText(
            f"{message} {percent}%  残り "
            f"{time.strftime('%H:%M:%S', time.gmtime(max(0, remaining)))}"
        )
        progress.setValue(done)
        QApplication.processEvents()

    def load_frame(self) -> None:
        self.load_frame_with_options(skip_pose_overlay=False, fast_seek=False)

    def load_frame_with_options(self, skip_pose_overlay: bool = False, fast_seek: bool = False) -> None:
        self.pending_slider_index = None
        frame_no_text = self.current_row().get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            APP_LOGGER.warning("frame_no が不正です: index=%s value=%s", self.current_index, frame_no_text)
            QMessageBox.warning(self, "Error", f"frame_no が不正です: {frame_no_text}")
            return
        frame = self.read_frame_number(frame_no, prefer_sequential=True, fast_seek=fast_seek)
        if frame is None:
            APP_LOGGER.warning("フレームを読めません: frame_no=%s video=%s", frame_no, self.video_path)
            QMessageBox.warning(self, "Error", f"フレームを読めません: frame_no={frame_no}")
            return
        self.source_frame = frame
        h, w = frame.shape[:2]
        self.image_width = w
        self.image_height = h
        self.refresh_canvas_frame(skip_pose_overlay=skip_pose_overlay)
        self.frame_label.setText(f"{self.current_index + 1}/{len(self.data.rows)}")
        self.frame_input.setText(str(frame_no))
        self.sequence_slider.blockSignals(True)
        self.sequence_slider.setValue(self.current_index)
        self.sequence_slider.blockSignals(False)
        self.select_current_frame_row()
        self.refresh_mosaic_table(skip_pose_overlay=skip_pose_overlay)

    def update_frame_position_labels(self) -> None:
        row = self.current_row()
        self.frame_label.setText(f"{self.current_index + 1}/{len(self.data.rows)}")
        self.frame_input.setText(row.get("frame_no", ""))

    def schedule_slider_frame_load(self, index: int) -> None:
        if self.playback_active:
            self.stop_preview_playback()
        self.current_index = index
        self.pending_slider_index = index
        self.update_frame_position_labels()
        self.slider_load_timer.start(FRAME_SLIDER_LOAD_DELAY_MS)

    def load_pending_slider_frame(self) -> None:
        if self.pending_slider_index is None:
            return
        target_index = self.pending_slider_index
        self.pending_slider_index = None
        if not (0 <= target_index < len(self.data.rows)):
            return
        if self.current_index != target_index:
            self.current_index = target_index
        self.load_frame_with_options(skip_pose_overlay=True, fast_seek=False)

    def read_frame_number(self, frame_no: int, prefer_sequential: bool, fast_seek: bool = False) -> np.ndarray | None:
        if self.cap is None:
            return None
        can_read_forward = (
            prefer_sequential
            and self.display_cap_pos is not None
            and frame_no >= self.display_cap_pos
            and frame_no - self.display_cap_pos <= DISPLAY_SEQUENTIAL_MAX_SKIP
        )
        if not can_read_forward:
            if fast_seek:
                frame = self.read_frame_number_fast_seek(frame_no)
                if frame is not None:
                    self.display_cap_pos = None
                    return frame
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

    def read_frame_number_fast_seek(self, frame_no: int) -> np.ndarray | None:
        if shutil.which("ffmpeg") is None:
            return None
        fps = self.video_fps if getattr(self, "video_fps", 0) and self.video_fps > 0 else 30.0
        target_sec = max(0.0, frame_no / fps)
        pre_seek_sec = max(0.0, target_sec - FAST_SEEK_PREROLL_SEC)
        inner_seek_sec = target_sec - pre_seek_sec
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{pre_seek_sec:.6f}",
            "-i",
            str(self.video_path),
            "-ss",
            f"{inner_seek_sec:.6f}",
            "-frames:v",
            "1",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        started_at = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=FAST_SEEK_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            APP_LOGGER.warning("ffmpeg高速シークがタイムアウト: frame=%s", frame_no)
            return None
        elapsed = time.monotonic() - started_at
        if proc.returncode != 0 or not proc.stdout:
            APP_LOGGER.warning(
                "ffmpeg高速シーク失敗: frame=%s returncode=%s stderr=%s",
                frame_no,
                proc.returncode,
                proc.stderr.decode("utf-8", errors="replace")[:500],
            )
            return None
        image = cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            APP_LOGGER.warning("ffmpeg高速シーク画像デコード失敗: frame=%s", frame_no)
            return None
        APP_LOGGER.info("ffmpeg高速シーク完了: frame=%s elapsed=%.3f", frame_no, elapsed)
        return image

    def refresh_canvas_frame(self, *args, skip_pose_overlay: bool = False) -> None:
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
            shape = meta.get("shape", "square")
            for slot in range(1, MAX_MOSAICS + 1):
                if not is_on(self.current_row().get(f"mosaic{slot}_on")):
                    continue
                rect = get_rect(self.current_row(), slot)
                if rect is not None:
                    apply_preview_effect(frame, rect, intensity, effect, shape)
        draw_crotch = self.crotch_overlay_checkbox.isChecked()
        draw_skeleton = self.skeleton_overlay_checkbox.isChecked()
        if (draw_crotch or draw_skeleton) and not skip_pose_overlay:
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
        shape = meta.get("shape", "square")
        for slot in range(1, MAX_MOSAICS + 1):
            if not is_on(row.get(f"mosaic{slot}_on")):
                continue
            rect = get_rect(row, slot)
            if rect is not None:
                apply_preview_effect(result, rect, intensity, effect, shape)
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
        set_meta_value(self.data, "shape", self.shape_combo.currentText())
        set_meta_value(self.data, "intensity", str(self.intensity_spin.value()))
        log_user_action(
            "プレビュー設定変更",
            effect=self.effect_combo.currentText(),
            shape=self.shape_combo.currentText(),
            intensity=self.intensity_spin.value(),
        )
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
        progress: QProgressDialog | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> None:
        total = len(self.data.rows)
        progress_total = progress_total or total
        owns_progress = False
        if show_progress:
            if progress is None:
                progress = self.reusable_progress_dialog("処理中", progress_message, max(1, progress_total))
                owns_progress = True
            else:
                progress.setRange(0, max(1, progress_total))
                progress.setLabelText(progress_message)
                progress.show()
            progress.setValue(min(progress_total, progress_offset + total))
            QApplication.processEvents()
        self.frame_table_model.refresh_all()
        if progress is not None and owns_progress:
            progress.setValue(max(1, progress_total))
            progress.hide()
        self.sequence_slider.set_person_ranges(self.person_ranges())
        self.sequence_slider.set_modified_ranges(self.modified_ranges())
        self.sequence_slider.set_mosaic_ranges(self.mosaic_active_ranges())

    def row_modified(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.original_rows):
            return False
        return self.data.rows[idx] != self.original_rows[idx]

    def modified_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        for idx, row in enumerate(self.data.rows):
            if not self.row_modified(idx):
                if current_start is not None and current_end is not None:
                    ranges.append((current_start, current_end))
                current_start = None
                current_end = None
                continue
            try:
                frame_no = int(row.get("frame_no", ""))
            except ValueError:
                continue
            if current_start is not None and current_end is not None and frame_no == current_end + 1:
                current_end = frame_no
                continue
            if current_start is not None and current_end is not None:
                ranges.append((current_start, current_end))
            current_start = frame_no
            current_end = frame_no
        if current_start is not None and current_end is not None:
            ranges.append((current_start, current_end))
        return ranges

    def refresh_modified_markers(self) -> None:
        if hasattr(self, "sequence_slider"):
            self.sequence_slider.set_modified_ranges(self.modified_ranges())
            self.sequence_slider.set_mosaic_ranges(self.mosaic_active_ranges())

    def mosaic_slot_count(self) -> int:
        """CSV のヘッダに実在する mosaic スロットの最大番号を返す（上限固定の全走査を避ける）。"""
        max_slot = 0
        for field in self.data.fieldnames:
            if field.startswith("mosaic") and field.endswith("_x1"):
                try:
                    max_slot = max(max_slot, int(field[len("mosaic"):-len("_x1")]))
                except ValueError:
                    continue
        return min(MAX_MOSAICS, max_slot)

    def mosaic_active_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        slot_count = self.mosaic_slot_count()
        for row in self.data.rows:
            try:
                frame_no = int(row.get("frame_no", ""))
            except ValueError:
                continue
            # モザイクマトリクスに行が出る条件（rect を持つスロット）と一致させる。
            # ON/OFF は問わない。最初の枠が見つかった時点で打ち切る。
            has_active = False
            for slot in range(1, slot_count + 1):
                if row.get(f"mosaic{slot}_x1") and get_rect(row, slot) is not None:
                    has_active = True
                    break
            if not has_active:
                if current_start is not None and current_end is not None:
                    ranges.append((current_start, current_end))
                current_start = None
                current_end = None
                continue
            if current_start is not None and current_end is not None and frame_no == current_end + 1:
                current_end = frame_no
                continue
            if current_start is not None and current_end is not None:
                ranges.append((current_start, current_end))
            current_start = frame_no
            current_end = frame_no
        if current_start is not None and current_end is not None:
            ranges.append((current_start, current_end))
        return ranges

    def keep_ranges(self) -> list[tuple[int, int]]:
        try:
            return parse_frame_ranges(self.data.meta_dict.get("keep_ranges", ""))
        except ValueError:
            return []

    def person_ranges(self) -> list[tuple[int, int, int]]:
        ranges: list[tuple[int, int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_count = 0
        for row in self.data.rows:
            try:
                frame_no = int(row.get("frame_no", ""))
                person_count = int(row.get("person_count", "0") or "0")
            except ValueError:
                continue
            if person_count <= 0:
                if current_start is not None and current_end is not None:
                    ranges.append((current_start, current_end, current_count))
                current_start = None
                current_end = None
                current_count = 0
                continue
            if (
                current_start is not None
                and current_end is not None
                and current_count == person_count
                and frame_no == current_end + 1
            ):
                current_end = frame_no
                continue
            if current_start is not None and current_end is not None:
                ranges.append((current_start, current_end, current_count))
            current_start = frame_no
            current_end = frame_no
            current_count = person_count
        if current_start is not None and current_end is not None:
            ranges.append((current_start, current_end, current_count))
        return ranges

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
        self.frame_table_model.refresh_row(idx)

    def person_count_for_row(self, row: dict[str, str]) -> str:
        person_count = row.get("person_count")
        if person_count not in (None, ""):
            return person_count
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
        selection_model = self.frame_table.selectionModel()
        selection_model.blockSignals(True)
        self.frame_table.selectRow(self.current_index)
        self.frame_table.scrollTo(self.frame_table_model.index(self.current_index, 0))
        selection_model.blockSignals(False)

    def refresh_mosaic_table(self, skip_pose_overlay: bool = False) -> None:
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
        self.update_frame_table_row(self.current_index, row)
        self.refresh_canvas_frame(skip_pose_overlay=skip_pose_overlay)
        self.canvas.update()

    def visible_mosaic_slots(self, row: dict[str, str]) -> list[int]:
        visible = [
            slot for slot in range(1, MAX_MOSAICS + 1)
            if get_rect(row, slot) is not None or slot in self.trace_slots
        ]
        if not visible:
            visible.append(1)
        if self.selected_slot not in visible:
            visible.append(self.selected_slot)
            visible.sort()
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

    def update_trace_min_score(self, value: float) -> None:
        self.trace_min_score = float(value)
        try:
            save_editor_config_value("trace_min_score", self.trace_min_score)
        except Exception:
            APP_LOGGER.exception("追跡スコア閾値を保存できません: value=%s", value)
        log_user_action("追跡スコア閾値変更", value=self.trace_min_score)

    def update_csrt_trace(self, enabled: bool) -> None:
        self.csrt_trace = bool(enabled)
        try:
            save_editor_config_value("csrt_trace", self.csrt_trace)
        except Exception:
            APP_LOGGER.exception("CSRT追跡設定を保存できません: value=%s", enabled)
        log_user_action("CSRT追跡設定変更", enabled=self.csrt_trace)

    def csrt_trace_enabled(self) -> bool:
        return bool(getattr(self, "csrt_trace", False)) and csrt_available()

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
                self.start_trace_with_length(slot, source="table")
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

    def select_frame_row(self, index: QModelIndex) -> None:
        row = index.row()
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
            self.schedule_slider_frame_load(index)

    def select_mosaic(self, row: int, col: int) -> None:
        header_item = self.mosaic_table.verticalHeaderItem(row)
        if header_item is None:
            return
        try:
            self.selected_slot = int(header_item.text())
        except ValueError:
            return
        if col == 0:
            slot = self.selected_slot
            key = f"mosaic{slot}_on"
            # クリックした現在行のトグル結果を、選択中の全フレーム行に同じ値で適用する。
            on = not is_on(self.current_row().get(key))
            rows = sorted({index.row() for index in self.frame_table.selectedIndexes()})
            if not rows:
                rows = [self.current_index]
            for row_index in rows:
                if not (0 <= row_index < len(self.data.rows)):
                    continue
                row = self.data.rows[row_index]
                row[key] = "1" if on else "0"
                # ON にする行にモザイク枠が無ければ、その行を基準に直近座標で埋める。
                if on and get_rect(row, slot) is None:
                    self.populate_row_from_nearest(row_index, slot, on=True)
                self.update_frame_table_row(row_index, row)
            self.mark_dirty()
            log_user_action("モザイク有効切替", frames=rows, slot=slot, enabled=on)
            self.refresh_mosaic_table()
            return
        if col == 1:
            if self.selected_slot in self.trace_slots:
                self.trace_slots.discard(self.selected_slot)
                self.auto_track_anchors.pop(self.selected_slot, None)
                log_user_action("追跡解除", frame=self.current_frame_no(), slot=self.selected_slot, source="click")
                self.refresh_mosaic_table()
            else:
                self.start_trace_with_length(self.selected_slot, source="click")
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

    def start_trace_with_length(self, slot: int, source: str) -> None:
        start_frame = self.current_frame_no()
        if start_frame is None:
            APP_LOGGER.warning("追跡開始失敗: 現在フレーム番号が不正 slot=%s", slot)
            QMessageBox.warning(self, "Error", "現在フレーム番号が不正です")
            self.refresh_mosaic_table()
            return
        if self.current_index >= len(self.data.rows) - 1:
            APP_LOGGER.info("追跡開始不可: 最終フレーム slot=%s frame=%s", slot, start_frame)
            QMessageBox.information(self, "追跡不可", "最終フレームのため、これ以上トレースできません。")
            self.refresh_mosaic_table()
            return

        current_start, current_end = self.trace_range_for_slot(slot)
        default_length = 1
        if current_start == start_frame and current_end is not None:
            default_length = max(1, min(999999, int(current_end) - start_frame))
        length, ok = QInputDialog.getInt(
            self,
            "トレース長さ",
            (
                f"mosaic{slot} を現在フレーム {start_frame} から何フレーム先までトレースしますか？\n"
                "途中で動体検出を見失った場合は、そのフレームをTrace Endにして停止します。"
            ),
            default_length,
            1,
            999999,
            1,
        )
        if not ok:
            log_user_action("追跡開始キャンセル", frame=start_frame, slot=slot, source=source)
            self.refresh_mosaic_table()
            return

        end_index = min(len(self.data.rows) - 1, self.current_index + length)
        try:
            end_frame = int(self.data.rows[end_index].get("frame_no", ""))
        except ValueError:
            APP_LOGGER.warning("追跡開始失敗: End frame_no が不正 slot=%s index=%s", slot, end_index)
            QMessageBox.warning(self, "Error", "End frame_no が不正です")
            self.refresh_mosaic_table()
            return

        self.trace_ranges[slot] = (start_frame, end_frame)
        self.trace_slots.add(slot)
        self.selected_slot = slot
        log_user_action(
            "追跡開始",
            frame=start_frame,
            slot=slot,
            source=source,
            length=length,
            requested_end=end_frame,
        )
        self.refresh_mosaic_table()
        self.trace_slot_range(slot)

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
        self.refresh_modified_markers()
        self.load_frame()
        self.refresh_mosaic_table()

    def clear_selected_frames(self, *args) -> None:
        rows = sorted({index.row() for index in self.frame_table.selectedIndexes()})
        if not rows:
            rows = [self.current_index]
        log_user_action("フレームクリア", rows=rows)
        for row_index in rows:
            if 0 <= row_index < len(self.data.rows):
                row = self.data.rows[row_index]
                # frame_no 以外（検出メタ・comment・全モザイク枠）を空に初期化する。
                cleared = {key: "" for key in row}
                cleared["frame_no"] = row.get("frame_no", "")
                self.data.rows[row_index] = cleared
                self.update_frame_table_row(row_index, cleared)
        self.selected_slot = 1
        self.mark_dirty()
        self.refresh_modified_markers()
        self.load_frame()
        self.refresh_mosaic_table()

    def populate_selected_from_nearest(self, on: bool) -> bool:
        return self.populate_row_from_nearest(self.current_index, self.selected_slot, on)

    def populate_row_from_nearest(self, row_index: int, slot: int, on: bool) -> bool:
        row = self.data.rows[row_index]
        if get_rect(row, slot) is not None:
            return True
        candidate = (
            self.nearest_rect(slot, anchor_index=row_index)
            or self.nearest_any_rect(anchor_index=row_index)
        )
        if candidate is None:
            return False
        rect, label = candidate
        rect = clamp_rect(rect, self.image_width, self.image_height)
        set_rect(row, slot, rect, on=on)
        if label:
            row[f"mosaic{slot}_type"] = label
        return True

    def create_from_nearest(self, *args) -> None:
        if self.populate_selected_from_nearest(on=True):
            log_user_action("直近枠から作成", frame=self.current_frame_no(), slot=self.selected_slot)
            self.mark_dirty()
            self.refresh_mosaic_table()

    def nearest_rect(self, slot: int, anchor_index: int | None = None) -> tuple[QRect, str] | None:
        if anchor_index is None:
            anchor_index = self.current_index
        for idx in range(anchor_index - 1, -1, -1):
            row = self.data.rows[idx]
            rect = get_rect(row, slot)
            if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                return QRect(rect), row.get(f"mosaic{slot}_type", "")
        for idx in range(anchor_index + 1, len(self.data.rows)):
            row = self.data.rows[idx]
            rect = get_rect(row, slot)
            if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                return QRect(rect), row.get(f"mosaic{slot}_type", "")
        return None

    def nearest_any_rect(self, anchor_index: int | None = None) -> tuple[QRect, str] | None:
        if anchor_index is None:
            anchor_index = self.current_index
        for idx in range(anchor_index - 1, -1, -1):
            row = self.data.rows[idx]
            for slot in range(1, MAX_MOSAICS + 1):
                rect = get_rect(row, slot)
                if rect is not None and is_on(row.get(f"mosaic{slot}_on")):
                    return QRect(rect), row.get(f"mosaic{slot}_type", "")
        for idx in range(anchor_index + 1, len(self.data.rows)):
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
        if hasattr(self, "slider_load_timer"):
            self.slider_load_timer.stop()
            self.pending_slider_index = None
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

        anchor_frame = self.frame_at_index(start_index)
        if anchor_frame is None:
            APP_LOGGER.warning("範囲追跡失敗: Start frame を読めません slot=%s start=%s", slot, start_frame)
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: Start frame を読めません")
            self.refresh_mosaic_table()
            return
        label = self.data.rows[start_index].get(f"mosaic{slot}_type", "") or "manual"
        # CSRT 追跡が有効なら専用ループに委譲する（白い車など低コントラスト対象に強い）。
        if self.csrt_trace_enabled():
            self._trace_slot_range_csrt(
                slot, start_index, end_index, start_frame, end_frame, start_rect, anchor_frame, label
            )
            return
        # B1: ドリフト照合の基準を開始フレーム固定にせず、直近の信頼できた
        # フレームへ定期更新するローリング参照にする（緩やかな見た目変化を許容）。
        reference_frame = anchor_frame.copy()
        reference_rect = QRect(start_rect)
        anchor_index = start_index
        anchor_rect = QRect(start_rect)
        total_trace_frames = max(1, end_index - start_index)
        trace_started_at = time.monotonic()
        progress = self.reusable_progress_dialog(
            "フレーム追跡",
            f"mosaic{slot}: {start_frame}-{end_frame} を追跡中...",
            total_trace_frames,
            cancelable=True,
        )
        self.auto_track_status.setText(f"mosaic{slot}: {start_frame}-{end_frame} を追跡中...")
        QApplication.processEvents()
        stop_message = ""
        success_count = 0
        final_end_frame = start_frame
        updated_start: int | None = None
        updated_end: int | None = None
        try:
            for target_index in range(start_index + 1, end_index + 1):
                processed_count = target_index - start_index
                frame_no_text = self.data.rows[target_index].get("frame_no", "")
                try:
                    target_frame_no = int(frame_no_text)
                except ValueError:
                    stop_message = f"mosaic{slot}: frame_no不正"
                    APP_LOGGER.warning("範囲追跡停止: slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    break
                if processed_count == 1 or processed_count % 5 == 0 or processed_count == total_trace_frames:
                    self.update_progress_dialog(
                        progress,
                        f"mosaic{slot}: frame {target_frame_no} を追跡中... {processed_count}/{total_trace_frames}",
                        processed_count,
                        total_trace_frames,
                        trace_started_at,
                    )
                    if progress.wasCanceled():
                        stop_message = f"mosaic{slot}: キャンセル"
                        APP_LOGGER.info("範囲追跡キャンセル: slot=%s frame=%s", slot, target_frame_no)
                        break
                gap = target_index - anchor_index - 1
                if gap > self.max_interpolate_gap():
                    stop_message = f"mosaic{slot}: gap超過"
                    APP_LOGGER.warning("範囲追跡停止: slot=%s message=%s", slot, stop_message)
                    final_end_frame = target_frame_no
                    break
                next_frame = self.read_frame_number(target_frame_no, prefer_sequential=True)
                if next_frame is None:
                    stop_message = f"mosaic{slot}: フレーム読込失敗"
                    APP_LOGGER.warning("範囲追跡停止: slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    final_end_frame = target_frame_no
                    break
                result = track_rect_proxy(
                    anchor_frame,
                    next_frame,
                    anchor_rect,
                    allow_scale=self.trace_scale_enabled(slot),
                )
                if result is None:
                    stop_message = f"mosaic{slot}: 失敗"
                    APP_LOGGER.warning("範囲追跡停止: slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    final_end_frame = target_frame_no
                    break
                tracked_rect, score, method = result
                if not self.track_confident_enough(score, method):
                    stop_message = f"mosaic{slot}: 見失い(score={score:.3f})"
                    APP_LOGGER.warning("範囲追跡停止: slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    final_end_frame = target_frame_no
                    break
                tracked_rect = clamp_rect(tracked_rect, next_frame.shape[1], next_frame.shape[0])
                # スケールドリフト・ガード: 開始枠サイズに対する累積倍率が範囲外なら、
                # 前景の大きな構造物などへの乗り移りとみなして停止する。
                scale_w = tracked_rect.width() / max(1, start_rect.width())
                scale_h = tracked_rect.height() / max(1, start_rect.height())
                if (
                    max(scale_w, scale_h) > TRACE_MAX_SCALE_RATIO
                    or min(scale_w, scale_h) < TRACE_MIN_SCALE_RATIO
                ):
                    stop_message = f"mosaic{slot}: スケールドリフト(倍率 w={scale_w:.2f}, h={scale_h:.2f})"
                    APP_LOGGER.warning(
                        "範囲追跡停止: slot=%s target_index=%s message=%s scale_w=%.3f scale_h=%.3f max=%.2f min=%.2f",
                        slot,
                        target_index,
                        stop_message,
                        scale_w,
                        scale_h,
                        TRACE_MAX_SCALE_RATIO,
                        TRACE_MIN_SCALE_RATIO,
                    )
                    final_end_frame = target_frame_no
                    break
                should_check_original = (
                    processed_count == 1
                    or processed_count % TRACE_ORIGINAL_TEMPLATE_INTERVAL == 0
                    or processed_count == total_trace_frames
                )
                if should_check_original:
                    drift_score = template_similarity_score(
                        reference_frame,
                        reference_rect,
                        next_frame,
                        tracked_rect,
                    )
                    min_score = getattr(self, "trace_min_score", DEFAULT_TRACE_TO_END_MIN_SCORE)
                    # B2: 参照との類似度が低くても、フレーム間追跡スコアが十分高ければ
                    # 「同じ対象の見た目変化」とみなして停止しない。両方低いときだけ停止。
                    if (
                        drift_score is not None
                        and drift_score < min_score
                        and score < TRACE_LIVE_TRUST_SCORE
                    ):
                        stop_message = f"mosaic{slot}: ドリフト検出(ref={drift_score:.3f}, live={score:.3f})"
                        APP_LOGGER.warning(
                            "範囲追跡停止: slot=%s target_index=%s message=%s ref=%.3f live=%.3f min=%.3f",
                            slot,
                            target_index,
                            stop_message,
                            drift_score,
                            score,
                            min_score,
                        )
                        final_end_frame = target_frame_no
                        break
                    # B1: 停止しなかった信頼できるフレームを次回比較の基準に更新する。
                    reference_frame = next_frame.copy()
                    reference_rect = QRect(tracked_rect)
                span = max(1, target_index - anchor_index)
                for idx in range(anchor_index + 1, target_index + 1):
                    t = (idx - anchor_index) / span
                    fill_rect = tracked_rect if idx == target_index else interpolate_rect(anchor_rect, tracked_rect, t)
                    target_row = self.data.rows[idx]
                    set_rect(target_row, slot, clamp_rect(fill_rect, next_frame.shape[1], next_frame.shape[0]), on=True)
                    target_row[f"mosaic{slot}_type"] = label
                    target_row[f"mosaic{slot}_score"] = f"track:{score:.3f}" if idx == target_index else "track:interpolated"
                    updated_start = idx if updated_start is None else min(updated_start, idx)
                    updated_end = idx if updated_end is None else max(updated_end, idx)
                anchor_index = target_index
                anchor_rect = QRect(tracked_rect)
                anchor_frame = next_frame
                final_end_frame = target_frame_no
                success_count += 1
                if success_count % 50 == 0:
                    frame_no = self.data.rows[target_index].get("frame_no", "")
                    self.auto_track_status.setText(f"mosaic{slot}: 範囲追跡中... frame {frame_no}")
                    QApplication.processEvents()
        finally:
            progress.setValue(total_trace_frames if not stop_message else min(total_trace_frames, success_count + 1))
            progress.hide()
        if not stop_message:
            stop_message = f"mosaic{slot}: 範囲追跡完了"
        self.trace_slots.discard(slot)
        self.auto_track_anchors.pop(slot, None)
        self.trace_ranges[slot] = (start_frame, final_end_frame)
        self.auto_track_status.setText(f"{stop_message} / End {final_end_frame} / 更新 {success_count} frame")
        log_user_action("範囲追跡終了", slot=slot, updated_frames=success_count, end=final_end_frame, message=stop_message)
        self.mark_dirty()
        if updated_start is not None and updated_end is not None:
            self.frame_table_model.refresh_rows(updated_start, updated_end)
        self.refresh_modified_markers()
        self.load_frame()

    def _trace_slot_range_csrt(
        self,
        slot: int,
        start_index: int,
        end_index: int,
        start_frame: int,
        end_frame: int,
        start_rect: QRect,
        anchor_frame: np.ndarray,
        label: str,
    ) -> None:
        """CSRT トラッカーによる範囲追跡。色＋HOG＋空間信頼性を使い、白い車など
        低コントラスト対象や背景への乗り移りに強い。処理は縮小画像で行う。"""
        tracker = create_csrt_tracker()
        if tracker is None:
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: CSRT が利用できません（opencv-contrib-python が必要）")
            self.refresh_mosaic_table()
            return
        # T scale フラグ: F のときは枠サイズを開始時のまま固定し、中心だけ追従する。
        allow_scale = self.trace_scale_enabled(slot)
        h0, w0 = anchor_frame.shape[:2]
        scale = min(1.0, TRACK_PROXY_MAX_DIM / max(1, max(h0, w0)))

        def to_proxy(frame: np.ndarray) -> np.ndarray:
            if scale >= 0.999:
                return frame
            return cv2.resize(
                frame,
                (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )

        proxy_start = to_proxy(anchor_frame)
        ph_h, ph_w = proxy_start.shape[:2]
        init_rect = scale_rect(start_rect, scale) if scale < 0.999 else QRect(start_rect)
        px, py, pw, ph = rect_to_xywh(init_rect)
        px = max(0, min(px, ph_w - 2))
        py = max(0, min(py, ph_h - 2))
        pw = max(2, min(pw, ph_w - px))
        ph = max(2, min(ph, ph_h - py))
        try:
            tracker.init(proxy_start, (int(px), int(py), int(pw), int(ph)))
        except cv2.error as exc:
            APP_LOGGER.warning("CSRT init 失敗: slot=%s err=%s", slot, exc)
            self.trace_slots.discard(slot)
            self.auto_track_status.setText(f"mosaic{slot}: CSRT 初期化に失敗しました")
            self.refresh_mosaic_table()
            return
        inv = 1.0 / scale if scale > 0 else 1.0
        total_trace_frames = max(1, end_index - start_index)
        trace_started_at = time.monotonic()
        progress = self.reusable_progress_dialog(
            "フレーム追跡",
            f"mosaic{slot}: {start_frame}-{end_frame} を追跡中...(CSRT)",
            total_trace_frames,
            cancelable=True,
        )
        self.auto_track_status.setText(f"mosaic{slot}: {start_frame}-{end_frame} を追跡中...(CSRT)")
        QApplication.processEvents()
        stop_message = ""
        success_count = 0
        final_end_frame = start_frame
        updated_start: int | None = None
        updated_end: int | None = None
        try:
            for target_index in range(start_index + 1, end_index + 1):
                processed_count = target_index - start_index
                try:
                    target_frame_no = int(self.data.rows[target_index].get("frame_no", ""))
                except ValueError:
                    stop_message = f"mosaic{slot}: frame_no不正"
                    APP_LOGGER.warning("範囲追跡停止(CSRT): slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    break
                if processed_count == 1 or processed_count % 5 == 0 or processed_count == total_trace_frames:
                    self.update_progress_dialog(
                        progress,
                        f"mosaic{slot}: frame {target_frame_no} を追跡中...(CSRT) {processed_count}/{total_trace_frames}",
                        processed_count,
                        total_trace_frames,
                        trace_started_at,
                    )
                    if progress.wasCanceled():
                        stop_message = f"mosaic{slot}: キャンセル"
                        APP_LOGGER.info("範囲追跡キャンセル(CSRT): slot=%s frame=%s", slot, target_frame_no)
                        break
                next_frame = self.read_frame_number(target_frame_no, prefer_sequential=True)
                if next_frame is None:
                    stop_message = f"mosaic{slot}: フレーム読込失敗"
                    final_end_frame = target_frame_no
                    break
                ok, box = tracker.update(to_proxy(next_frame))
                if not ok:
                    stop_message = f"mosaic{slot}: 見失い(CSRT)"
                    APP_LOGGER.warning("範囲追跡停止(CSRT): slot=%s target_index=%s message=%s", slot, target_index, stop_message)
                    final_end_frame = target_frame_no
                    break
                bx, by, bw, bh = box
                tracked_rect = QRect(
                    round(bx * inv), round(by * inv), max(1, round(bw * inv)), max(1, round(bh * inv))
                )
                if not allow_scale:
                    # T scale = F: CSRT が推定した中心だけ使い、サイズは開始枠で固定する。
                    center = tracked_rect.center()
                    fixed_w, fixed_h = start_rect.width(), start_rect.height()
                    tracked_rect = QRect(
                        round(center.x() - fixed_w / 2),
                        round(center.y() - fixed_h / 2),
                        fixed_w,
                        fixed_h,
                    )
                tracked_rect = clamp_rect(tracked_rect, next_frame.shape[1], next_frame.shape[0])
                scale_w = tracked_rect.width() / max(1, start_rect.width())
                scale_h = tracked_rect.height() / max(1, start_rect.height())
                if (
                    max(scale_w, scale_h) > TRACE_MAX_SCALE_RATIO
                    or min(scale_w, scale_h) < TRACE_MIN_SCALE_RATIO
                ):
                    stop_message = f"mosaic{slot}: スケールドリフト(倍率 w={scale_w:.2f}, h={scale_h:.2f})"
                    APP_LOGGER.warning(
                        "範囲追跡停止(CSRT): slot=%s target_index=%s message=%s scale_w=%.3f scale_h=%.3f",
                        slot, target_index, stop_message, scale_w, scale_h,
                    )
                    final_end_frame = target_frame_no
                    break
                target_row = self.data.rows[target_index]
                set_rect(target_row, slot, tracked_rect, on=True)
                target_row[f"mosaic{slot}_type"] = label
                target_row[f"mosaic{slot}_score"] = "track:csrt"
                updated_start = target_index if updated_start is None else min(updated_start, target_index)
                updated_end = target_index if updated_end is None else max(updated_end, target_index)
                final_end_frame = target_frame_no
                success_count += 1
                if success_count % 50 == 0:
                    self.auto_track_status.setText(f"mosaic{slot}: 範囲追跡中...(CSRT) frame {target_frame_no}")
                    QApplication.processEvents()
        finally:
            progress.setValue(total_trace_frames if not stop_message else min(total_trace_frames, success_count + 1))
            progress.hide()
        if not stop_message:
            stop_message = f"mosaic{slot}: 範囲追跡完了(CSRT)"
        self.trace_slots.discard(slot)
        self.auto_track_anchors.pop(slot, None)
        self.trace_ranges[slot] = (start_frame, final_end_frame)
        self.auto_track_status.setText(f"{stop_message} / End {final_end_frame} / 更新 {success_count} frame")
        log_user_action("範囲追跡終了", slot=slot, updated_frames=success_count, end=final_end_frame, message=stop_message, tracker="csrt")
        self.mark_dirty()
        if updated_start is not None and updated_end is not None:
            self.frame_table_model.refresh_rows(updated_start, updated_end)
        self.refresh_modified_markers()
        self.load_frame()

    def track_confident_enough(self, score: float, method: str) -> bool:
        if "template" not in method:
            return False
        return score >= getattr(self, "trace_min_score", DEFAULT_TRACE_TO_END_MIN_SCORE)

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
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "レシピを保存",
            str(self.csv_path),
            "CSV (*.csv);;All files (*)",
        )
        if not selected:
            log_user_action("保存キャンセル", csv_path=self.csv_path)
            return
        save_path = Path(selected)
        if save_path.suffix.lower() != ".csv":
            save_path = save_path.with_suffix(".csv")
        if save_path.exists() and save_path != self.csv_path:
            result = QMessageBox.question(self, "上書き確認", f"{save_path} は既に存在します。上書きしますか？")
            if result != QMessageBox.StandardButton.Yes:
                log_user_action("保存キャンセル", csv_path=self.csv_path, save_path=save_path)
                return
        log_user_action("保存", csv_path=self.csv_path, save_path=save_path)
        try:
            self.save_current_recipe_with_progress("レシピを保存中...", save_path=save_path)
        except Exception as exc:
            APP_LOGGER.exception("保存失敗: csv=%s save_path=%s", self.csv_path, save_path)
            QMessageBox.critical(self, "Error", f"保存に失敗しました: {exc}")
            return

    def save_without_confirm(self) -> None:
        if self.csv_path is None:
            return
        APP_LOGGER.info("確認なし保存: csv=%s", self.csv_path)
        try:
            self.save_current_recipe_with_progress("レシピを保存中...", refresh_table=False)
        except Exception:
            APP_LOGGER.exception("確認なし保存失敗: csv=%s", self.csv_path)
            raise

    def save_current_recipe_with_progress(
        self,
        message: str,
        progress_dialog: PostProgressDialog | None = None,
        refresh_table: bool = True,
        save_path: Path | None = None,
    ) -> None:
        if self.csv_path is None:
            return
        target_path = save_path or self.csv_path
        total_rows = len(self.data.rows)
        total_steps = max(1, total_rows * (3 if refresh_table else 2))
        started_at = time.monotonic()
        progress = None
        if progress_dialog is None:
            progress = self.reusable_progress_dialog("保存中", message, total_steps)
        else:
            progress_dialog.set_stage_progress(message, 0, total_steps)

        def on_write_progress(current: int, total: int) -> None:
            done = min(total_steps, current)
            stage_message = "CSV列を確認中..." if current <= total // 2 else "CSVを書き込み中..."
            if progress_dialog is not None:
                progress_dialog.set_stage_progress(stage_message, done, total_steps, started_at)
                if progress_dialog.cancel_requested:
                    raise EncodingCancelled("エンコードをキャンセルしました。")
            else:
                self.update_progress_dialog(
                    progress,
                    stage_message,
                    done,
                    total_steps,
                    started_at,
                )

        try:
            write_pre_csv(target_path, self.data, progress_callback=on_write_progress)
            self.csv_path = target_path
            self.remember_recipe_path(target_path)
            self.original_rows = [dict(row) for row in self.data.rows]
            self.dirty = False
            self.setWindowTitle(f"pre CSV Editor - {self.csv_path.name}")
            self.refresh_modified_markers()
            if hasattr(self, "frame_table_model"):
                self.frame_table_model.refresh_all()
            if refresh_table:
                self.populate_frame_table(
                    show_progress=True,
                    progress_message="画面を更新中...",
                    progress=progress,
                    progress_offset=total_rows * 2,
                    progress_total=total_steps,
                )
            if progress_dialog is not None:
                progress_dialog.set_stage_progress("保存完了", total_steps, total_steps)
            else:
                progress.setValue(total_steps)
                progress.setLabelText("保存完了")
            QApplication.processEvents()
        finally:
            if progress is not None:
                progress.hide()

    def encode_estimate_text(self) -> str:
        frame_total = len(self.data.rows)
        try:
            keep_ranges = self.keep_ranges()
        except Exception:
            keep_ranges = []
        if keep_ranges:
            frame_total = sum(end - start + 1 for start, end in keep_ranges)
            frame_total = min(frame_total, len(self.data.rows))
        fps = self.video_fps if getattr(self, "video_fps", 0) and self.video_fps > 0 else 30.0
        video_seconds = frame_total / fps if fps > 0 else 0.0
        estimate_min = video_seconds * 0.5
        estimate_max = video_seconds * 1.5
        return (
            f"対象フレーム数: {frame_total}\n"
            f"対象動画時間: 約 {format_duration(video_seconds)}\n"
            f"予測時間: 約 {format_duration(estimate_min)} - {format_duration(estimate_max)}\n\n"
            "エンコードしますか？"
        )

    def encode_post(self) -> None:
        if self.csv_path is None:
            APP_LOGGER.info("エンコード不可: レシピ未選択")
            QMessageBox.information(self, "未選択", "レシピが開かれていません。")
            return
        if getattr(self, "_encoding", False):
            APP_LOGGER.info("エンコード多重起動を無視")
            return
        log_user_action("エンコード開始要求", csv_path=self.csv_path)
        output_path = post_output_path_from_csv(self.csv_path)
        log_path = output_path.with_name(f"{output_path.stem}_log.txt")
        result = QMessageBox.question(
            self,
            "エンコード確認",
            self.encode_estimate_text(),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if result != QMessageBox.StandardButton.Ok:
            log_user_action("エンコードキャンセル", csv_path=self.csv_path)
            return
        self._encoding = True
        dialog = PostProgressDialog(self.csv_path, output_path, log_path, self)
        dialog.finished.connect(lambda _result: setattr(self, "_encoding", False))
        dialog.show()
        dialog.set_stage_progress("エンコード前に保存中...", 0, max(1, len(self.data.rows)))
        try:
            self.save_current_recipe_with_progress(
                "エンコード前に保存中...",
                progress_dialog=dialog,
                refresh_table=False,
            )
        except EncodingCancelled as exc:
            self._encoding = False
            dialog.reject()
            QMessageBox.information(self, "キャンセル", str(exc))
            return
        except Exception as exc:
            self._encoding = False
            dialog.reject()
            QMessageBox.critical(self, "Error", f"保存に失敗したためエンコードできません: {exc}")
            return
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
