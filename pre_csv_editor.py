#!/usr/bin/env python3
"""_pre.csv editor for mosaic rectangles."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MAX_MOSAICS = 255
DEFAULT_VISIBLE_MOSAICS = 5
HANDLE_SIZE = 8


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


def infer_pre_video_path(csv_path: Path) -> Path:
    if csv_path.name.endswith("_pre.csv"):
        return csv_path.with_suffix(".mp4")
    return csv_path.with_name(f"{csv_path.stem}.mp4")


def is_on(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "on", "yes", "y"}


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


class FrameCanvas(QWidget):
    def __init__(self, parent: "EditorWindow") -> None:
        super().__init__()
        self.parent_window = parent
        self.pixmap: QPixmap | None = None
        self.image_size = None
        self.drag_mode: str | None = None
        self.drag_start_img = QPoint()
        self.drag_start_rect = QRect()
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

    def set_frame(self, pixmap: QPixmap, image_size) -> None:
        self.pixmap = pixmap
        self.image_size = image_size
        self.update()

    def image_rect_on_widget(self) -> QRectF:
        if not self.pixmap:
            return QRectF()
        scaled = self.pixmap.size()
        scaled.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        return QRectF(x, y, scaled.width(), scaled.height())

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

        slot = self.parent_window.selected_slot
        rect = get_rect(row, slot)
        if rect:
            mode = self.hit_mode(event.pos(), rect)
            if mode:
                self.drag_mode = mode
                self.drag_start_img = img_pos
                self.drag_start_rect = QRect(rect)
                return
        self.drag_mode = "create"
        self.drag_start_img = img_pos
        self.drag_start_rect = QRect(img_pos, img_pos)
        set_rect(row, slot, QRect(img_pos, img_pos), on=True)
        self.parent_window.mark_dirty()
        self.parent_window.refresh_mosaic_table()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self.drag_mode:
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
        self.drag_mode = None


class EditorWindow(QMainWindow):
    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.data = read_pre_csv(csv_path)
        self.video_path = infer_pre_video_path(csv_path)
        if not self.video_path.is_file():
            raise RuntimeError(f"_pre.mp4 が見つかりません: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません: {self.video_path}")
        self.current_index = 0
        self.selected_slot = 1
        self.dirty = False
        self.image_width = 0
        self.image_height = 0

        self.setWindowTitle(f"pre CSV Editor - {csv_path.name}")
        self.resize(1500, 900)
        self.build_ui()
        self.load_frame()

    def build_ui(self) -> None:
        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_with_confirm)
        self.addAction(save_action)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.canvas = FrameCanvas(self)
        left_layout.addWidget(self.canvas, stretch=1)

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
        left_layout.addLayout(nav)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.info_label = QLabel(f"CSV: {self.csv_path}\nVideo: {self.video_path}")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        self.frame_table = QTableWidget(len(self.data.rows), 3)
        self.frame_table.setHorizontalHeaderLabels(["行", "frame_no", "checked"])
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(QLabel("CSV行"))
        right_layout.addWidget(self.frame_table, stretch=1)

        self.frame_label = QLabel()
        self.on_checkbox = QCheckBox("選択mosaic ON")
        self.type_input = QLineEdit()
        self.type_input.setPlaceholderText("mosaic type")
        self.create_from_nearest_button = QPushButton("直近枠から作成")
        self.delete_button = QPushButton("選択をOFF")
        self.save_button = QPushButton("保存")
        right_layout.addWidget(self.frame_label)
        right_layout.addWidget(self.on_checkbox)
        right_layout.addWidget(QLabel("Type"))
        right_layout.addWidget(self.type_input)
        right_layout.addWidget(self.create_from_nearest_button)
        right_layout.addWidget(self.delete_button)
        right_layout.addWidget(self.save_button)

        self.mosaic_table = QTableWidget(DEFAULT_VISIBLE_MOSAICS, 7)
        self.mosaic_table.setHorizontalHeaderLabels(["No", "ON", "type", "x1", "y1", "x2", "y2"])
        self.mosaic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mosaic_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.mosaic_table, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([1100, 400])

        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.go_button.clicked.connect(self.go_to_frame)
        self.save_button.clicked.connect(self.save_with_confirm)
        self.delete_button.clicked.connect(self.disable_selected)
        self.create_from_nearest_button.clicked.connect(self.create_from_nearest)
        self.on_checkbox.stateChanged.connect(self.toggle_selected)
        self.type_input.editingFinished.connect(self.update_selected_type)
        self.mosaic_table.cellClicked.connect(self.select_mosaic)
        self.mosaic_table.itemChanged.connect(self.update_mosaic_from_table)
        self.frame_table.cellClicked.connect(self.select_frame_row)
        self.frame_table.itemChanged.connect(self.update_frame_from_table)
        self.populate_frame_table()

    def current_row(self) -> dict[str, str]:
        return self.data.rows[self.current_index]

    def mark_dirty(self) -> None:
        self.dirty = True
        if not self.windowTitle().endswith("*"):
            self.setWindowTitle(self.windowTitle() + " *")

    def load_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.warning(self, "Error", f"フレームを読めません: index={self.current_index}")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image_width = w
        self.image_height = h
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(QPixmap.fromImage(qimg), (w, h))
        frame_no = self.current_row().get("frame_no", "")
        self.frame_label.setText(f"{self.current_index + 1}/{len(self.data.rows)}  frame_no={frame_no}")
        self.frame_input.setText(frame_no)
        if get_rect(self.current_row(), self.selected_slot) is None:
            self.populate_selected_from_nearest(on=False)
        self.select_current_frame_row()
        self.refresh_mosaic_table()

    def populate_frame_table(self) -> None:
        self.frame_table.blockSignals(True)
        for idx, row in enumerate(self.data.rows):
            values = [str(idx + 1), row.get("frame_no", ""), row.get("checked", "")]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 1):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.frame_table.setItem(idx, col, item)
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
            values = [
                str(slot),
                "1" if is_on(row.get(f"mosaic{slot}_on")) else "0",
                row.get(f"mosaic{slot}_type", ""),
                row.get(f"mosaic{slot}_x1", ""),
                row.get(f"mosaic{slot}_y1", ""),
                row.get(f"mosaic{slot}_x2", ""),
                row.get(f"mosaic{slot}_y2", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if slot == self.selected_slot:
                    item.setBackground(QColor(255, 245, 200))
                self.mosaic_table.setItem(table_row, col, item)
        self.mosaic_table.blockSignals(False)
        selected_row = visible_slots.index(self.selected_slot) if self.selected_slot in visible_slots else 0
        self.mosaic_table.selectRow(selected_row)
        self.mosaic_table.scrollToItem(self.mosaic_table.item(selected_row, 0))
        self.on_checkbox.blockSignals(True)
        self.on_checkbox.setChecked(is_on(row.get(f"mosaic{self.selected_slot}_on")))
        self.on_checkbox.blockSignals(False)
        self.type_input.setText(row.get(f"mosaic{self.selected_slot}_type", ""))
        self.canvas.update()

    def visible_mosaic_slots(self, row: dict[str, str]) -> list[int]:
        visible = list(range(1, DEFAULT_VISIBLE_MOSAICS + 1))
        for slot in range(DEFAULT_VISIBLE_MOSAICS + 1, MAX_MOSAICS + 1):
            if is_on(row.get(f"mosaic{slot}_on")) or get_rect(row, slot) is not None:
                visible.append(slot)
        if self.selected_slot not in visible:
            visible.append(self.selected_slot)
        return visible

    def update_mosaic_from_table(self, item: QTableWidgetItem) -> None:
        no_item = self.mosaic_table.item(item.row(), 0)
        if no_item is None:
            return
        try:
            slot = int(no_item.text())
        except ValueError:
            return
        col = item.column()
        if col == 0:
            return
        keys = {
            1: f"mosaic{slot}_on",
            2: f"mosaic{slot}_type",
            3: f"mosaic{slot}_x1",
            4: f"mosaic{slot}_y1",
            5: f"mosaic{slot}_x2",
            6: f"mosaic{slot}_y2",
        }
        key = keys.get(col)
        if not key:
            return
        self.current_row()[key] = item.text().strip()
        if col >= 3:
            set_blank_crotch(self.current_row(), slot)
        self.mark_dirty()
        self.selected_slot = slot
        self.refresh_mosaic_table()

    def update_frame_from_table(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        self.data.rows[item.row()]["checked"] = item.text().strip()
        self.mark_dirty()

    def select_frame_row(self, row: int, col: int) -> None:
        if 0 <= row < len(self.data.rows):
            self.current_index = row
            self.load_frame()

    def select_mosaic(self, row: int, col: int) -> None:
        no_item = self.mosaic_table.item(row, 0)
        if no_item is None:
            return
        try:
            self.selected_slot = int(no_item.text())
        except ValueError:
            return
        selected_rect = get_rect(self.current_row(), self.selected_slot)
        if selected_rect is None:
            self.populate_selected_from_nearest(on=False)
        self.refresh_mosaic_table()

    def toggle_selected(self, *args) -> None:
        self.current_row()[f"mosaic{self.selected_slot}_on"] = "1" if self.on_checkbox.isChecked() else "0"
        if self.on_checkbox.isChecked() and get_rect(self.current_row(), self.selected_slot) is None:
            self.populate_selected_from_nearest(on=True)
        self.mark_dirty()
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
            self.current_index -= 1
            self.load_frame()

    def next_frame(self, *args) -> None:
        if self.current_index < len(self.data.rows) - 1:
            self.current_index += 1
            self.load_frame()

    def go_to_frame(self, *args) -> None:
        wanted = self.frame_input.text().strip()
        for idx, row in enumerate(self.data.rows):
            if row.get("frame_no") == wanted:
                self.current_index = idx
                self.load_frame()
                return
        QMessageBox.information(self, "Not found", f"frame_no={wanted} はCSVにありません")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key.Key_Delete:
            self.disable_selected()
        else:
            super().keyPressEvent(event)

    def save_with_confirm(self, *args) -> None:
        if QMessageBox.question(self, "保存確認", f"{self.csv_path} を上書き保存しますか？") != QMessageBox.StandardButton.Yes:
            return
        write_pre_csv(self.csv_path, self.data)
        self.dirty = False
        self.setWindowTitle(f"pre CSV Editor - {self.csv_path.name}")

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
