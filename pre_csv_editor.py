#!/usr/bin/env python3
"""_pre.csv editor for mosaic rectangles."""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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

MAX_MOSAICS = 255
DEFAULT_VISIBLE_MOSAICS = 5
HANDLE_SIZE = 8
DEFAULT_INTENSITY_SLIDER_MAX = 100
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15


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


def refine_rect_with_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray, rect: QRect) -> QRect | None:
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
    deltas = next_points[valid].reshape(-1, 2) - points[valid].reshape(-1, 2)
    dx, dy = np.median(deltas, axis=0)
    return QRect(round(x + float(dx)), round(y + float(dy)), w, h)


def track_rect(prev_frame: np.ndarray, next_frame: np.ndarray, rect: QRect) -> tuple[QRect, float, str] | None:
    template_result = track_rect_template(prev_frame, next_frame, rect)
    flow_rect = refine_rect_with_optical_flow(prev_frame, next_frame, rect)
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
        return template_rect, score, "template+flow"
    if score >= 0.55:
        return template_rect, score, "template"
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


class SequenceSlider(QSlider):
    def __init__(self, frame_numbers: list[int]) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.frame_numbers = frame_numbers
        self.setMinimumHeight(42)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: #a0a0a0;
            }
            QSlider::handle:horizontal {
                width: 5px;
                margin: -8px 0;
                background: #0068c9;
                border: 1px solid #004b91;
            }
            """
        )

    def tick_index(self, frame_no: int) -> int:
        index = bisect.bisect_left(self.frame_numbers, frame_no)
        if index >= len(self.frame_numbers):
            return len(self.frame_numbers) - 1
        if index > 0 and frame_no - self.frame_numbers[index - 1] < self.frame_numbers[index] - frame_no:
            return index - 1
        return index

    def paintEvent(self, event) -> None:
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
        if self.drag_mode == "pan":
            self.unsetCursor()
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
        event.accept()


class EditorWindow(QMainWindow):
    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.data = read_pre_csv(csv_path)
        self.original_rows = [dict(row) for row in self.data.rows]
        self.video_path = source_video_path(self.data, self.csv_path)
        if not self.video_path.is_file():
            raise RuntimeError(f"元動画が見つかりません: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません: {self.video_path}")
        self.current_index = 0
        self.selected_slot = 1
        self.dirty = False
        self.image_width = 0
        self.image_height = 0
        self.source_frame: np.ndarray | None = None
        self.trace_slots: set[int] = set()
        self.auto_track_anchors: dict[int, tuple[int, QRect, np.ndarray, str]] = {}

        self.setWindowTitle(f"pre CSV Editor - {csv_path.name}")
        self.resize(1500, 900)
        self.build_ui()
        self.load_frame()

    def build_ui(self) -> None:
        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_with_confirm)
        self.addAction(save_action)

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
        frame_status_layout = QHBoxLayout()
        frame_status_layout.addWidget(self.frame_label)
        frame_status_layout.addWidget(self.zoom_label)
        frame_status_layout.addWidget(self.preview_checkbox)
        frame_status_layout.addStretch()
        left_layout.addLayout(frame_status_layout)
        frame_numbers = [int(row["frame_no"]) for row in self.data.rows]
        self.sequence_slider = SequenceSlider(frame_numbers)
        self.sequence_slider.setRange(0, max(0, len(self.data.rows) - 1))
        left_layout.addWidget(self.sequence_slider)

        nav = QHBoxLayout()
        self.prev_button = QPushButton("前へ")
        self.next_button = QPushButton("次へ")
        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("frame_no")
        self.go_button = QPushButton("移動")
        for button in (self.prev_button, self.next_button, self.go_button):
            nav.addWidget(button)
        nav.addWidget(QLabel("Frame:"))
        nav.addWidget(self.frame_input)
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
        for key in ("confidence", "pose_model", "yolo_nsfw_model", "interpolate_gap", "no_crotch"):
            meta_layout.addRow(key, QLabel(meta.get(key, "")))
        right_layout.addLayout(meta_layout)

        self.frame_table = QTableWidget(len(self.data.rows), 5)
        self.frame_table.setHorizontalHeaderLabels(
            ["frame_no", "modify", "Mosaic", "Crotch", "comment"]
        )
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(QLabel("CSV行"))
        right_layout.addWidget(self.frame_table, stretch=1)

        self.type_input = QLineEdit()
        self.type_input.setPlaceholderText("mosaic type")
        self.auto_track_status = QLabel("")
        self.auto_track_status.setWordWrap(True)
        self.create_from_nearest_button = QPushButton("直近枠から作成")
        self.save_button = QPushButton("保存")
        self.restore_frame_button = QPushButton("現在フレームを元に戻す")
        right_layout.addWidget(QLabel("Type"))
        right_layout.addWidget(self.type_input)
        right_layout.addWidget(self.auto_track_status)
        right_layout.addWidget(self.create_from_nearest_button)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.restore_frame_button)

        self.mosaic_table = QTableWidget(DEFAULT_VISIBLE_MOSAICS, 8)
        self.mosaic_table.setHorizontalHeaderLabels(["mosaic", "Trace", "type", "score", "x1", "y1", "x2", "y2"])
        self.mosaic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mosaic_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.mosaic_table, stretch=1)
        right_layout.addLayout(nav)
        splitter.addWidget(right)
        splitter.setSizes([1100, 400])

        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.go_button.clicked.connect(self.go_to_frame)
        self.sequence_slider.valueChanged.connect(self.select_sequence_frame)
        self.save_button.clicked.connect(self.save_with_confirm)
        self.restore_frame_button.clicked.connect(self.restore_current_frame)
        self.create_from_nearest_button.clicked.connect(self.create_from_nearest)
        self.preview_checkbox.stateChanged.connect(self.refresh_canvas_frame)
        self.effect_combo.currentTextChanged.connect(self.update_preview_meta)
        self.intensity_slider.valueChanged.connect(self.intensity_spin.setValue)
        self.intensity_spin.valueChanged.connect(self.sync_intensity_slider)
        self.intensity_spin.valueChanged.connect(self.update_preview_meta)
        self.type_input.editingFinished.connect(self.update_selected_type)
        self.mosaic_table.cellClicked.connect(self.select_mosaic)
        self.mosaic_table.itemChanged.connect(self.update_mosaic_from_table)
        self.frame_table.cellClicked.connect(self.select_frame_row)
        self.frame_table.itemChanged.connect(self.update_frame_from_table)
        self.populate_frame_table()

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
            QMessageBox.warning(self, "Error", f"frame_no が不正です: {frame_no_text}")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self.cap.read()
        if not ok:
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(QPixmap.fromImage(qimg), (w, h))

    def update_preview_meta(self, *args) -> None:
        set_meta_value(self.data, "effect", self.effect_combo.currentText())
        set_meta_value(self.data, "intensity", str(self.intensity_spin.value()))
        self.mark_dirty()
        self.refresh_canvas_frame()

    def sync_intensity_slider(self, intensity: int) -> None:
        if intensity > self.intensity_slider.maximum():
            self.intensity_slider.setMaximum(intensity)
        self.intensity_slider.setValue(intensity)

    def populate_frame_table(self) -> None:
        self.frame_table.blockSignals(True)
        for idx, row in enumerate(self.data.rows):
            self.update_frame_table_row(idx, row)
        self.frame_table.blockSignals(False)

    def row_modified(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.original_rows):
            return False
        return self.data.rows[idx] != self.original_rows[idx]

    def update_frame_table_row(self, idx: int, row: dict[str, str]) -> None:
        mosaic_count = enabled_mosaic_count(row)
        nsfw_detection_count = row.get("nsfw_detection_count") or "?"
        values = [
            row.get("frame_no", ""),
            "T" if self.row_modified(idx) else "F",
            f"{mosaic_count}/{nsfw_detection_count}",
            "あり" if is_on(row.get("crotch_detected")) else "none",
            row.get("comment", ""),
        ]
        background = QColor(255, 220, 230) if mosaic_count else QColor(232, 232, 232)
        self.frame_table.blockSignals(True)
        try:
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(background)
                if col != 4:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.frame_table.setItem(idx, col, item)
        finally:
            self.frame_table.blockSignals(False)

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
            values = [
                true_false(row.get(f"mosaic{slot}_on")),
                "T" if slot in self.trace_slots else "F",
                row.get(f"mosaic{slot}_type", ""),
                row.get(f"mosaic{slot}_score", ""),
                row.get(f"mosaic{slot}_x1", ""),
                row.get(f"mosaic{slot}_y1", ""),
                row.get(f"mosaic{slot}_x2", ""),
                row.get(f"mosaic{slot}_y2", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 3):
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
            if is_on(row.get(f"mosaic{slot}_on")) or get_rect(row, slot) is not None or slot in self.trace_slots:
                visible.append(slot)
        if self.selected_slot not in visible:
            visible.append(self.selected_slot)
        return visible

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
            else:
                self.trace_slots.discard(slot)
                self.auto_track_anchors.pop(slot, None)
            self.selected_slot = slot
            self.refresh_mosaic_table()
            return
        keys = {
            0: f"mosaic{slot}_on",
            2: f"mosaic{slot}_type",
            4: f"mosaic{slot}_x1",
            5: f"mosaic{slot}_y1",
            6: f"mosaic{slot}_x2",
            7: f"mosaic{slot}_y2",
        }
        key = keys.get(col)
        if not key:
            return
        value = true_false(item.text()) if col == 0 else item.text().strip()
        self.current_row()[key] = value
        if col >= 4:
            self.current_row()[f"mosaic{slot}_score"] = ""
            set_blank_crotch(self.current_row(), slot)
        self.mark_dirty()
        self.selected_slot = slot
        self.refresh_mosaic_table()

    def update_frame_from_table(self, item: QTableWidgetItem) -> None:
        if item.column() != 4:
            return
        self.data.rows[item.row()]["comment"] = item.text().strip()
        self.mark_dirty()
        self.update_frame_table_row(item.row(), self.data.rows[item.row()])

    def select_frame_row(self, row: int, col: int) -> None:
        if 0 <= row < len(self.data.rows):
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
            self.refresh_mosaic_table()
            return
        if col == 1:
            if self.selected_slot in self.trace_slots:
                self.trace_slots.discard(self.selected_slot)
                self.auto_track_anchors.pop(self.selected_slot, None)
            else:
                self.trace_slots.add(self.selected_slot)
            self.refresh_mosaic_table()
            return
        selected_rect = get_rect(self.current_row(), self.selected_slot)
        if selected_rect is None:
            self.populate_selected_from_nearest(on=False)
        self.refresh_mosaic_table()

    def update_selected_type(self, *args) -> None:
        self.current_row()[f"mosaic{self.selected_slot}_type"] = self.type_input.text()
        self.mark_dirty()
        self.refresh_mosaic_table()

    def disable_selected(self, *args) -> None:
        self.current_row()[f"mosaic{self.selected_slot}_on"] = "0"
        set_blank_crotch(self.current_row(), self.selected_slot)
        self.mark_dirty()
        self.refresh_mosaic_table()

    def restore_current_frame(self, *args) -> None:
        self.data.rows[self.current_index] = dict(self.original_rows[self.current_index])
        self.mark_dirty()
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

    def go_to_frame(self, *args) -> None:
        wanted = self.frame_input.text().strip()
        for idx, row in enumerate(self.data.rows):
            if row.get("frame_no") == wanted:
                self.move_to_index(idx)
                return
        QMessageBox.information(self, "Not found", f"frame_no={wanted} はCSVにありません")

    def move_to_index(self, target_index: int) -> None:
        if target_index == self.current_index:
            return
        if target_index == self.current_index + 1:
            self.auto_track_next_frame(target_index)
        self.current_index = target_index
        self.load_frame()

    def frame_at_index(self, index: int) -> np.ndarray | None:
        frame_no_text = self.data.rows[index].get("frame_no", "")
        try:
            frame_no = int(frame_no_text)
        except ValueError:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self.cap.read()
        return frame if ok else None

    def auto_track_next_frame(self, target_index: int) -> None:
        if not self.trace_slots:
            return
        messages = []
        for slot in sorted(self.trace_slots):
            message = self.auto_track_slot_next_frame(target_index, slot)
            if message:
                messages.append(message)
        if messages:
            self.auto_track_status.setText(" / ".join(messages))

    def auto_track_slot_next_frame(self, target_index: int, slot: int) -> str:
        row = self.current_row()
        rect = get_rect(row, slot)
        prev_frame = self.source_frame if self.source_frame is not None else self.frame_at_index(self.current_index)
        if rect is not None and is_on(row.get(f"mosaic{slot}_on")) and prev_frame is not None:
            label = row.get(f"mosaic{slot}_type", "") or "manual"
            self.auto_track_anchors[slot] = (self.current_index, QRect(rect), prev_frame.copy(), label)

        anchor = self.auto_track_anchors.get(slot)
        if anchor is None:
            return f"mosaic{slot}: 枠なし"
        anchor_index, anchor_rect, anchor_frame, label = anchor
        gap = target_index - anchor_index - 1
        if gap < 0:
            self.auto_track_anchors.pop(slot, None)
            return f"mosaic{slot}: リセット"
        if gap > self.max_interpolate_gap():
            return f"mosaic{slot}: gap超過"
        next_frame = self.frame_at_index(target_index)
        if next_frame is None:
            return f"mosaic{slot}: フレーム読込失敗"
        result = track_rect(anchor_frame, next_frame, anchor_rect)
        if result is None:
            return f"mosaic{slot}: 失敗"
        tracked_rect, score, method = result
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
        return f"mosaic{slot}: frame {frame_no} ({method}, score={score:.3f}{gap_text})"

    def max_interpolate_gap(self) -> int:
        try:
            return max(0, int(self.data.meta_dict.get("interpolate_gap", "0")))
        except ValueError:
            return 0

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

    def save_with_confirm(self, *args) -> None:
        if QMessageBox.question(self, "保存確認", f"{self.csv_path} を上書き保存しますか？") != QMessageBox.StandardButton.Yes:
            return
        write_pre_csv(self.csv_path, self.data)
        self.original_rows = [dict(row) for row in self.data.rows]
        self.dirty = False
        self.setWindowTitle(f"pre CSV Editor - {self.csv_path.name}")
        self.populate_frame_table()

    def closeEvent(self, event) -> None:
        if self.dirty:
            result = QMessageBox.question(self, "未保存", "未保存の変更があります。閉じますか？")
            if result != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.cap.release()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="_pre.csv GUI editor")
    parser.add_argument("csv", nargs="?", help="編集する _pre.csv")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None:
        selected, _ = QFileDialog.getOpenFileName(None, "_pre.csv を選択", "", "CSV (*.csv)")
        if not selected:
            return
        csv_path = Path(selected)
    try:
        window = EditorWindow(csv_path)
    except Exception as exc:
        QMessageBox.critical(None, "Error", str(exc))
        return
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
