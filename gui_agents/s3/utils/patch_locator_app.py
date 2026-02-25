"""
PyQt6-based visual tester for patch locator.

Provides a GUI for:
- Loading a full screenshot
- Drawing a crop region to extract a patch
- Running patch matching
- Visualizing and exporting results
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QScrollArea,
    QComboBox,
)

from patch_locator import locate_patch, draw_match_box, MatchResult


class ImageCanvas(QLabel):
    """Canvas for drawing and interacting with images."""

    crop_started = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 2px solid #ccc; background: #f5f5f5;")
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pixmap = None
        self.pil_image = None
        self.crop_rect = None
        self.is_drawing = False
        self.start_point = None
        self.setMinimumSize(400, 300)
        self.zoom_level = 1.0

    def set_image(self, pil_image: Image.Image):
        """Load a PIL Image and display it."""
        self.pil_image = pil_image
        self.crop_rect = None
        self.display_image()

    def display_image(self):
        """Refresh the canvas display."""
        if self.pil_image is None:
            self.clear()
            return

        rgb_image = self.pil_image.convert("RGB")
        data = rgb_image.tobytes("raw", "RGB")
        w, h = rgb_image.size
        q_image = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888)

        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        self.pixmap = QPixmap.fromImage(q_image).scaled(
            scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(self.pixmap)

    def mousePressEvent(self, event):
        """Start crop region."""
        if self.pixmap is None:
            return
        self.is_drawing = True
        self.start_point = event.pos()
        self.crop_rect = QRect(self.start_point, self.start_point)

    def mouseMoveEvent(self, event):
        """Update crop region during drag."""
        if not self.is_drawing or self.start_point is None:
            return
        self.crop_rect = QRect(self.start_point, event.pos()).normalized()
        self.display_image()
        self.draw_crop_rect()

    def mouseReleaseEvent(self, event):
        """Finalize crop region."""
        if not self.is_drawing:
            return
        self.is_drawing = False
        if self.crop_rect and self.crop_rect.width() > 5 and self.crop_rect.height() > 5:
            self.crop_started.emit(self.crop_rect)
        self.display_image()
        if self.crop_rect:
            self.draw_crop_rect()

    def draw_crop_rect(self):
        """Draw the crop rectangle on the displayed pixmap."""
        if self.crop_rect is None or self.pixmap is None:
            return
        pixmap_copy = self.pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.crop_rect)
        painter.end()
        self.setPixmap(pixmap_copy)

    def set_zoom(self, level: float):
        """Set zoom level and refresh display."""
        self.zoom_level = max(0.2, min(3.0, level))
        self.display_image()
        if self.crop_rect:
            self.draw_crop_rect()


class PatchLocatorApp(QMainWindow):
    """Main PyQt6 application for patch locator visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patch Locator - Visual Tester")
        self.setGeometry(100, 100, 1400, 900)

        self.full_image_path = None
        self.full_image: Optional[Image.Image] = None
        self.patch_image: Optional[Image.Image] = None
        self.last_match: Optional[MatchResult] = None

        self.setup_ui()

    def setup_ui(self):
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        # Left panel: full image canvas
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("<b>Full Image (draw crop)</b>"))
        self.canvas = ImageCanvas()
        self.canvas.crop_started.connect(self.on_crop_drawn)
        left_panel.addWidget(self.canvas)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(20)
        self.zoom_slider.setMaximum(300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.sliderMoved.connect(
            lambda: self.canvas.set_zoom(self.zoom_slider.value() / 100.0)
        )
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addStretch()
        left_panel.addLayout(zoom_layout)

        # Right panel: controls and results
        right_panel = QVBoxLayout()

        # File loading
        file_layout = QHBoxLayout()
        self.full_btn = QPushButton("Load Full Image")
        self.full_btn.clicked.connect(self.load_full_image)
        self.patch_btn = QPushButton("Load Patch Image")
        self.patch_btn.clicked.connect(self.load_patch_image)
        file_layout.addWidget(self.full_btn)
        file_layout.addWidget(self.patch_btn)
        right_panel.addLayout(file_layout)

        # Matching options
        opts_layout = QVBoxLayout()
        opts_layout.addWidget(QLabel("<b>Matching Options</b>"))

        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step (NumPy only):"))
        self.step_spin = QSpinBox()
        self.step_spin.setMinimum(1)
        self.step_spin.setMaximum(20)
        self.step_spin.setValue(1)
        step_layout.addWidget(self.step_spin)
        step_layout.addStretch()
        opts_layout.addLayout(step_layout)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Score Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setMinimum(0)
        self.threshold_spin.setMaximum(100)
        self.threshold_spin.setValue(0)
        self.threshold_spin.setSuffix("%")
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        opts_layout.addLayout(threshold_layout)

        self.cv2_check = QCheckBox("Prefer OpenCV (if available)")
        self.cv2_check.setChecked(True)
        opts_layout.addWidget(self.cv2_check)
        opts_layout.addStretch()

        right_panel.addLayout(opts_layout)

        # Patch preview
        right_panel.addWidget(QLabel("<b>Patch Preview</b>"))
        self.patch_preview = QLabel()
        self.patch_preview.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        self.patch_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.patch_preview.setMinimumHeight(150)
        self.patch_preview.setMaximumHeight(150)
        right_panel.addWidget(self.patch_preview)

        # Match button
        self.match_btn = QPushButton("Find Patch in Full Image")
        self.match_btn.clicked.connect(self.run_matching)
        self.match_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        right_panel.addWidget(self.match_btn)

        # Results
        right_panel.addWidget(QLabel("<b>Results</b>"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        right_panel.addWidget(self.results_text)

        # Export buttons
        export_layout = QHBoxLayout()
        self.save_preview_btn = QPushButton("Save Preview")
        self.save_preview_btn.clicked.connect(self.save_preview)
        self.save_crop_btn = QPushButton("Save Crop")
        self.save_crop_btn.clicked.connect(self.save_crop)
        export_layout.addWidget(self.save_preview_btn)
        export_layout.addWidget(self.save_crop_btn)
        export_layout.addStretch()
        right_panel.addLayout(export_layout)

        # Assemble layout
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(400)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 0)
        central.setLayout(main_layout)

    def load_full_image(self):
        """Load the full screenshot image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Full Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            try:
                self.full_image = Image.open(path)
                self.full_image_path = path
                self.canvas.set_image(self.full_image)
                self.results_text.setText(
                    f"✓ Full image loaded\nSize: {self.full_image.size}\nPath: {path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def load_patch_image(self):
        """Load the patch image for matching."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Patch Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            try:
                self.patch_image = Image.open(path)
                self.display_patch_preview()
                self.results_text.setText(
                    f"✓ Patch image loaded\nSize: {self.patch_image.size}\nPath: {path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load patch: {e}")

    def display_patch_preview(self):
        """Show a thumbnail of the patch image."""
        if self.patch_image is None:
            return
        thumb = self.patch_image.copy()
        thumb.thumbnail((150, 150), Image.Resampling.LANCZOS)
        data = thumb.convert("RGB").tobytes("raw", "RGB")
        w, h = thumb.size
        q_image = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.patch_preview.setPixmap(QPixmap.fromImage(q_image))

    def on_crop_drawn(self, rect: QRect):
        """Extract patch when user finishes drawing crop region."""
        if self.full_image is None:
            QMessageBox.warning(self, "Warning", "Load a full image first!")
            return

        # Convert from canvas coordinates to image coordinates
        scale = self.canvas.zoom_level
        x1 = int(rect.left() / scale)
        y1 = int(rect.top() / scale)
        x2 = int(rect.right() / scale)
        y2 = int(rect.bottom() / scale)

        x1, x2 = max(0, x1), min(self.full_image.width, x2)
        y1, y2 = max(0, y1), min(self.full_image.height, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        self.patch_image = self.full_image.crop((x1, y1, x2, y2))
        self.display_patch_preview()

    def run_matching(self):
        """Execute patch matching and display results."""
        if self.full_image is None or self.patch_image is None:
            QMessageBox.warning(self, "Warning", "Load both full and patch images!")
            return

        try:
            threshold = self.threshold_spin.value() / 100.0
            step = self.step_spin.value()
            prefer_cv2 = self.cv2_check.isChecked()

            self.last_match = locate_patch(
                self.full_image,
                self.patch_image,
                prefer_cv2=prefer_cv2,
                step=step,
                score_threshold=threshold,
            )

            if self.last_match:
                result_text = (
                    f"✓ Match found!\n\n"
                    f"Position: ({self.last_match.x}, {self.last_match.y})\n"
                    f"Size: {self.last_match.width}×{self.last_match.height}\n"
                    f"Score: {self.last_match.score:.4f}\n"
                    f"Method: {self.last_match.method}"
                )
                self.results_text.setText(result_text)
                self.visualize_match()
            else:
                self.results_text.setText("✗ No match found or score below threshold")
                self.canvas.set_image(self.full_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Matching failed: {e}")
            self.results_text.setText(f"✗ Error: {e}")

    def visualize_match(self):
        """Draw the match box on the canvas."""
        if self.last_match is None or self.full_image is None:
            return
        result_img = draw_match_box(self.full_image, self.last_match)
        self.canvas.set_image(result_img)

    def save_preview(self):
        """Save the current canvas (with match box) to a file."""
        if self.canvas.pixmap is None:
            QMessageBox.warning(self, "Warning", "Nothing to save!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Preview", "", "PNG (*.png)")
        if path:
            try:
                self.canvas.pixmap.save(path)
                QMessageBox.information(self, "Success", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def save_crop(self):
        """Save the extracted patch image."""
        if self.patch_image is None:
            QMessageBox.warning(self, "Warning", "No patch to save!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Patch", "", "PNG (*.png)")
        if path:
            try:
                self.patch_image.save(path)
                QMessageBox.information(self, "Success", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")


def main():
    """Launch the application."""
    app = QApplication(sys.argv)
    window = PatchLocatorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
