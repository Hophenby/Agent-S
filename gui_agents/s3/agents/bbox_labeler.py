import sys
import json
import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Win32 helpers (no extra dependencies required)
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32 = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def list_windows() -> List[Dict]:
    """Enumerate visible top-level windows with titles."""
    windows: List[Dict] = []

    def _callback(hwnd, lparam):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
            buffer = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buffer, 512)
            title = buffer.value.strip()
            rect = RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if width > 0 and height > 0:
                    windows.append(
                        {
                            "hwnd": hwnd,
                            "title": title,
                            "rect": {
                                "x": rect.left,
                                "y": rect.top,
                                "width": width,
                                "height": height,
                            },
                        }
                    )
        return True

    cb = EnumWindowsProc(_callback)
    user32.EnumWindows(cb, 0)
    windows.sort(key=lambda w: w["title"].lower())
    return windows


def grab_window_pixmap(hwnd: int) -> QPixmap:
    screen = QApplication.primaryScreen()
    if not screen:
        return QPixmap()
    return screen.grabWindow(hwnd)


class AnnotatorView(QGraphicsView):
    """Graphics view that lets the user draw bounding boxes on a pixmap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(self.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._current_rect_item: Optional[QGraphicsRectItem] = None
        self._annotations: List[Dict] = []
        self._start_pos: Optional[QPointF] = None
        self._label_items: List[QGraphicsSimpleTextItem] = []

    def has_pixmap(self) -> bool:
        return self._pixmap_item is not None

    def set_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._annotations.clear()
        self._label_items.clear()
        self._current_rect_item = None
        self._start_pos = None
        self._fit_view()

    def annotations(self) -> List[Dict]:
        return list(self._annotations)

    def clear_annotations(self):
        if not self._pixmap_item:
            return
        for item in list(self._scene.items()):
            if item is self._pixmap_item:
                continue
            self._scene.removeItem(item)
        self._annotations.clear()
        self._label_items.clear()
        self._current_rect_item = None
        self._start_pos = None

    def mousePressEvent(self, event):
        if not self._pixmap_item or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self._start_pos = self.mapToScene(event.pos())
        rect = QRectF(self._start_pos, self._start_pos)
        pen = QPen(QColor(255, 0, 0), 2)
        self._current_rect_item = self._scene.addRect(rect, pen)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._current_rect_item and self._start_pos:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, current_pos).normalized()
            self._current_rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._current_rect_item or not self._start_pos:
            return super().mouseReleaseEvent(event)
        rect = self._current_rect_item.rect().toRect()
        if rect.width() < 2 or rect.height() < 2:
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
            self._start_pos = None
            return super().mouseReleaseEvent(event)

        name, ok = QInputDialog.getText(self, "Add label", "Name for this element:")
        if not ok or not name.strip():
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
            self._start_pos = None
            return super().mouseReleaseEvent(event)

        label = name.strip()
        self._annotations.append(
            {
                "name": label,
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height(),
            }
        )

        text_item = QGraphicsSimpleTextItem(label)
        text_item.setBrush(QColor(255, 0, 0))
        text_item.setPos(rect.x(), rect.y() - 18)
        self._scene.addItem(text_item)
        self._label_items.append(text_item)

        self._current_rect_item = None
        self._start_pos = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        self._fit_view()
        super().resizeEvent(event)

    def _fit_view(self):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Bounding Box Labeler")
        self.view = AnnotatorView(self)
        self.window_combo = QComboBox(self)
        self.status_label = QLabel("Ready", self)
        self.current_window: Optional[Dict] = None

        refresh_btn = QPushButton("Refresh windows")
        capture_btn = QPushButton("Capture selected window")
        desktop_btn = QPushButton("Capture desktop")
        clear_btn = QPushButton("Clear boxes")
        export_btn = QPushButton("Export JSON")

        refresh_btn.clicked.connect(self.refresh_windows)
        capture_btn.clicked.connect(self.capture_selected_window)
        desktop_btn.clicked.connect(self.capture_desktop)
        clear_btn.clicked.connect(self.view.clear_annotations)
        export_btn.clicked.connect(self.export_json)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Window:"))
        top_row.addWidget(self.window_combo, stretch=1)
        top_row.addWidget(refresh_btn)
        top_row.addWidget(capture_btn)
        top_row.addWidget(desktop_btn)

        mid_row = QHBoxLayout()
        mid_row.addWidget(clear_btn)
        mid_row.addWidget(export_btn)
        mid_row.addWidget(self.status_label, stretch=1)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(mid_row)

        container = QWidget(self)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.refresh_windows()
        self.resize(1200, 800)

    def refresh_windows(self):
        self.window_combo.clear()
        windows = list_windows()
        for win in windows:
            label = f"{win['title']} (hwnd={win['hwnd']})"
            self.window_combo.addItem(label, win)
        if windows:
            self.window_combo.setCurrentIndex(0)
        self.status_label.setText(f"Found {len(windows)} windows")

    def capture_selected_window(self):
        data = self.window_combo.currentData()
        if not data:
            QMessageBox.warning(self, "No window", "No window selected.")
            return
        hwnd = int(data["hwnd"])
        pixmap = grab_window_pixmap(hwnd)
        if pixmap.isNull():
            QMessageBox.critical(self, "Capture failed", "Could not capture the selected window.")
            return
        self.view.set_pixmap(pixmap)
        self.current_window = data
        self.status_label.setText(f"Captured: {data['title']}")

    def capture_desktop(self):
        pixmap = grab_window_pixmap(0)
        if pixmap.isNull():
            QMessageBox.critical(self, "Capture failed", "Could not capture the desktop.")
            return
        self.view.set_pixmap(pixmap)
        self.current_window = {
            "hwnd": 0,
            "title": "Desktop",
            "rect": {
                "x": 0,
                "y": 0,
                "width": pixmap.width(),
                "height": pixmap.height(),
            },
        }
        self.status_label.setText("Captured desktop")

    def export_json(self):
        if not self.view.has_pixmap():
            QMessageBox.warning(self, "Nothing to export", "Capture a window first.")
            return
        annotations = self.view.annotations()
        payload = {
            "window": self.current_window,
            "annotations": annotations,
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save annotations", "annotations.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.status_label.setText(f"Saved to {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save file: {exc}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
