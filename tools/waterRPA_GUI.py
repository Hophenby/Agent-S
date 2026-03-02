import sys
import os
import json
from typing import Dict, List
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QLineEdit, QScrollArea, 
                               QFileDialog, QTextEdit, QMessageBox, QFrame)
from PySide6.QtCore import Qt, Signal, QEvent, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF

from core import WorkflowConfig, WorkflowNode, NodeCategory, CommandType
from worker import RPAEngine, WorkerThread

# --------------------------
# GUI 界面 (原 rpa_gui.py)
# --------------------------

class TaskRow(QFrame):
    changed = Signal()

    def __init__(self, parent_widget, delete_callback):
        super().__init__(parent_widget)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid "+"#8f9faf"+"; border-radius: 8px; background: "+"#cfdfef"+"; }")
        self.setMinimumHeight(120)

        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(10, 10, 10, 10)
        self.layout_.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("节点"))

        self.node_id_input = QLineEdit()
        self.node_id_input.setPlaceholderText("ID")
        self.node_id_input.setFixedWidth(80)
        header_layout.addWidget(self.node_id_input)

        self.category_label = QLabel(NodeCategory.MIDDLE.value)
        self.category_label.setStyleSheet("color: #2d4f7c; background: #d9ecff; border-radius: 6px; padding: 2px 8px;")
        self.category_label.setFixedHeight(24)
        header_layout.addWidget(self.category_label)
        header_layout.addStretch()

        self.del_btn = QPushButton("删除")
        self.del_btn.setStyleSheet("color: #ff8080; font-weight: bold;")
        self.del_btn.setFixedWidth(55)
        self.del_btn.clicked.connect(lambda: delete_callback(self))
        header_layout.addWidget(self.del_btn)
        self.layout_.addLayout(header_layout)

        action_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems([cmd.label for cmd in CommandType.ordered()])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        action_layout.addWidget(self.type_combo)

        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("参数值 (如图片路径、文本、时间)")
        action_layout.addWidget(self.value_input)

        self.file_btn = QPushButton("选择图片")
        self.file_btn.clicked.connect(self.select_file)
        self.file_btn.setVisible(True)
        action_layout.addWidget(self.file_btn)
        self.layout_.addLayout(action_layout)

        route_layout = QHBoxLayout()
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("超时(秒,0禁用)")
        self.timeout_input.setText("0")
        self.timeout_input.setFixedWidth(100)
        self.timeout_input.setVisible(True)
        route_layout.addWidget(self.timeout_input)

        self.next_input = QLineEdit()
        self.next_input.setPlaceholderText("成功 -> 节点")
        self.next_input.setFixedWidth(140)
        route_layout.addWidget(self.next_input)

        self.fallback_input = QLineEdit()
        self.fallback_input.setPlaceholderText("失败 -> 节点")
        self.fallback_input.setFixedWidth(140)
        route_layout.addWidget(self.fallback_input)
        route_layout.addStretch()
        self.layout_.addLayout(route_layout)

        self.node_id_input.textChanged.connect(lambda _: self.changed.emit())
        self.value_input.textChanged.connect(lambda _: self.changed.emit())
        self.timeout_input.textChanged.connect(lambda _: self.changed.emit())
        self.next_input.textChanged.connect(lambda _: self.changed.emit())
        self.fallback_input.textChanged.connect(lambda _: self.changed.emit())
        self.type_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        
        self.show()

    def on_type_changed(self, text):
        cmd_type = CommandType.from_label(text)
        
        # 图片相关操作 (1, 2, 3, 8)
        if cmd_type in [CommandType.LEFT_CLICK, CommandType.LEFT_DOUBLE_CLICK, CommandType.RIGHT_CLICK, CommandType.HOVER]:
            self.file_btn.setVisible(True)
            self.file_btn.setText("选择图片")
            self.timeout_input.setVisible(True)
            self.value_input.setPlaceholderText("图片路径")
        # 输入 (4)
        elif cmd_type == CommandType.INPUT_TEXT:
            self.file_btn.setVisible(False)
            self.timeout_input.setVisible(False)
            self.value_input.setPlaceholderText("请输入要发送的文本")
        # 等待 (5)
        elif cmd_type == CommandType.WAIT:
            self.file_btn.setVisible(False)
            self.timeout_input.setVisible(False)
            self.value_input.setPlaceholderText("等待秒数 (如 1.5)")
        # 滚轮 (6)
        elif cmd_type == CommandType.SCROLL:
            self.file_btn.setVisible(False)
            self.timeout_input.setVisible(False)
            self.value_input.setPlaceholderText("滚动距离 (正数向上，负数向下)")
        # 系统按键 (7)
        elif cmd_type == CommandType.HOTKEY:
            self.file_btn.setVisible(False)
            self.timeout_input.setVisible(False)
            self.value_input.setPlaceholderText("组合键 (如 ctrl+s, alt+tab)")
        # 截图保存 (9)
        elif cmd_type == CommandType.SCREENSHOT:
            self.file_btn.setVisible(True)
            self.file_btn.setText("选择保存文件夹")
            self.timeout_input.setVisible(False)
            self.value_input.setPlaceholderText("保存目录 (如 D:\\Screenshots)")

        self.changed.emit()

    def set_data(self, data):
        """用于回填数据"""
        if isinstance(data, WorkflowNode):
            node_data = data
        elif isinstance(data, dict):
            node_data = WorkflowNode.from_dict(data)
        else:
            return

        cmd_type = CommandType.from_raw(node_data.type)
        value = node_data.value
        timeout_second = node_data.timeout_second
        node_id = str(node_data.node_id).strip()
        next_node = str(node_data.next).strip()
        fallback_next = str(node_data.fallback_next).strip()

        self.type_combo.setCurrentText(cmd_type.label)
        
        # 设置值
        self.value_input.setText(str(value))

        self.node_id_input.setText(node_id)
        self.next_input.setText(next_node)
        self.fallback_input.setText(fallback_next)
        
        # 设置超时秒数
        self.timeout_input.setText(str(timeout_second))
        self.set_category(NodeCategory.MIDDLE)

    def set_category(self, category: NodeCategory):
        tag = category.value
        self.category_label.setText(tag)

        if category == NodeCategory.START:
            self.category_label.setStyleSheet("color: "+"#1e5d2f"+"; background: "+"#d7f5df"+"; border-radius: 6px; padding: 2px 8px;")
        elif category == NodeCategory.END:
            self.category_label.setStyleSheet("color: "+"#7a3b1f"+"; background: "+"#ffe8d8"+"; border-radius: 6px; padding: 2px 8px;")
        else:
            self.category_label.setStyleSheet("color: "+"#2d4f7c"+"; background: "+"#d9ecff"+"; border-radius: 6px; padding: 2px 8px;")

    def select_file(self):
        cmd_type = CommandType.from_label(self.type_combo.currentText())
        
        # 截图保存 (9.0) -> 选择文件夹
        if cmd_type == CommandType.SCREENSHOT:
            folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹", os.getcwd())
            if folder:
                self.value_input.setText(folder)
        
        # 其他图片操作 (1, 2, 3, 8) -> 打开文件对话框
        else:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图片", os.getcwd(), "Image Files (*.png *.jpg *.bmp)")
            if filename:
                self.value_input.setText(filename)

    def get_data(self):
        cmd_type = CommandType.from_label(self.type_combo.currentText())
        value = self.value_input.text()
        
        # 数据校验与转换
        try:
            if cmd_type in [CommandType.WAIT, CommandType.SCROLL]:
                # 尝试转换为数字，如果失败可能会在运行时报错，这里简单处理
                if not value: value = "0"
            
            timeout_second = 0.0
            if self.timeout_input.isVisible():
                timeout_text = self.timeout_input.text().strip()
                if timeout_text:
                    timeout_second = float(timeout_text)
                if timeout_second < 0:
                    timeout_second = 0.0
        except ValueError:
            timeout_second = 0.0

        return WorkflowNode(
            node_id=self.node_id_input.text().strip(),
            type=cmd_type,
            value=value,
            timeout_second=timeout_second,
            next=self.next_input.text().strip(),
            fallback_next=self.fallback_input.text().strip()
        )

class FlowArrowOverlay(QWidget):
    def __init__(self, parent, get_rows_callback):
        super().__init__(parent)
        self.get_rows_callback = get_rows_callback
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent;")

    def _draw_arrow(self, painter:QPainter, start, end, color, dashed=False):
        pen = QPen(color, 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(start, end)

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            return

        ux = dx / length
        uy = dy / length
        arrow_size = 8

        tip = QPointF(end.x(), end.y())
        left = QPointF(
            end.x() - ux * arrow_size - uy * (arrow_size * 0.6),
            end.y() - uy * arrow_size + ux * (arrow_size * 0.6)
        )
        right = QPointF(
            end.x() - ux * arrow_size + uy * (arrow_size * 0.6),
            end.y() - uy * arrow_size - ux * (arrow_size * 0.6)
        )

        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([tip, left, right]))

    def paintEvent(self, event):
        super().paintEvent(event)

        rows = self.get_rows_callback()
        if not rows:
            return

        node_pos = {}
        for row in rows:
            node_id = row.node_id_input.text().strip()
            if node_id:
                node_pos[node_id] = row.geometry()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        for row in rows:
            src_id = row.node_id_input.text().strip()
            if not src_id:
                continue

            src_rect = row.geometry()
            success_target = row.next_input.text().strip()
            fail_target = row.fallback_input.text().strip()

            if success_target in node_pos:
                dst_rect = node_pos[success_target]
                start = QPointF(src_rect.center().x(), src_rect.bottom() - 4)
                end = QPointF(dst_rect.center().x(), dst_rect.top() + 4)
                self._draw_arrow(painter, start, end, QColor("#57c45a"), dashed=False)

            if fail_target in node_pos:
                dst_rect = node_pos[fail_target]
                start = QPointF(src_rect.right() - 4, src_rect.center().y())
                end = QPointF(dst_rect.left() + 4, dst_rect.center().y())
                self._draw_arrow(painter, start, end, QColor("#ff7a7a"), dashed=True)

class RPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("不高兴就喝水 RPA 配置工具")
        self.resize(800, 600)
        
        self.engine = RPAEngine()
        self.worker = None
        self.rows: List["TaskRow"] = []
        self._refreshing_arrows = False
        self.node_width = 560
        self.node_height = 130
        self.h_gap = 90
        self.v_gap = 40
        self.canvas_margin = 20

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制栏
        top_bar = QHBoxLayout()
        
        self.add_btn = QPushButton("+ 新增节点")
        self.add_btn.clicked.connect(self.add_row)
        top_bar.addWidget(self.add_btn)

        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        top_bar.addWidget(self.save_btn)

        self.load_btn = QPushButton("导入配置")
        self.load_btn.clicked.connect(self.load_config)
        top_bar.addWidget(self.load_btn)
        
        top_bar.addStretch()
        
        self.loop_check = QComboBox()
        self.loop_check.addItems(["执行一次", "循环执行"])
        top_bar.addWidget(self.loop_check)
        
        self.start_btn = QPushButton("开始运行")
        self.start_btn.setStyleSheet("background-color: "+"#4CAF50"+"; color: white;")
        self.start_btn.clicked.connect(self.start_task)
        top_bar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet("background-color: "+"#f44336"+"; color: white;")
        self.stop_btn.clicked.connect(self.stop_task)
        self.stop_btn.setEnabled(False)
        top_bar.addWidget(self.stop_btn)
        
        main_layout.addLayout(top_bar)

        # 节点画布区域 (滚动)
        self.scroll_ = QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.task_container = QWidget()
        self.task_container.setMinimumSize(900, 500)
        self.scroll_.setWidget(self.task_container)
        main_layout.addWidget(self.scroll_)

        self.flow_hint = QLabel("绿色实线=成功跳转    红色虚线=失败跳转")
        main_layout.addWidget(self.flow_hint)

        self.arrow_overlay = FlowArrowOverlay(self.task_container, lambda: self.rows)
        self.arrow_overlay.raise_()
        self.task_container.installEventFilter(self)
        self.scroll_.viewport().installEventFilter(self)

        # 日志区域
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        main_layout.addWidget(QLabel("运行日志:"))
        main_layout.addWidget(self.log_area)

        # 初始添加一行
        self.add_row()
        self.refresh_arrows()

    def eventFilter(self, obj, event:QEvent):
        if obj in [self.task_container, self.scroll_.viewport()] and event.type() in [QEvent.Type.Resize, QEvent.Type.LayoutRequest]:
            self.refresh_arrows()
        return super().eventFilter(obj, event)

    def auto_layout_nodes(self):
        if not self.rows:
            if self.task_container.minimumSize().width() != 900 or self.task_container.minimumSize().height() != 500:
                self.task_container.setMinimumSize(900, 500)
            return

        row_by_id: Dict[str, TaskRow] = {}
        for row in self.rows:
            node_id = row.node_id_input.text().strip()
            if node_id and node_id not in row_by_id:
                row_by_id[node_id] = row

        occupied = set()
        placed = {}

        def alloc_down(x, y):
            ny = y
            while (x, ny) in occupied:
                ny += 1
            return x, ny

        def alloc_right(x, y):
            nx = x
            while (nx, y) in occupied:
                nx += 1
            return nx, y

        root_row = self.rows[0]
        root_id = root_row.node_id_input.text().strip() or "1"
        placed[root_id] = (0, 0)
        occupied.add((0, 0))

        queue = [root_id]
        visited = set()

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            src_row = row_by_id.get(node_id)
            if not src_row:
                continue

            sx, sy = placed[node_id]
            succ = src_row.next_input.text().strip()
            fail = src_row.fallback_input.text().strip()

            if succ and succ in row_by_id and succ not in placed:
                px, py = alloc_down(sx, sy + 1)
                placed[succ] = (px, py)
                occupied.add((px, py))
                queue.append(succ)
            elif succ and succ in row_by_id:
                queue.append(succ)

            if fail and fail in row_by_id and fail not in placed:
                px, py = alloc_right(sx + 1, sy)
                placed[fail] = (px, py)
                occupied.add((px, py))
                queue.append(fail)
            elif fail and fail in row_by_id:
                queue.append(fail)

        max_y = max(y for _, y in occupied) if occupied else 0
        for row in self.rows:
            node_id = row.node_id_input.text().strip()
            if not node_id:
                continue
            if node_id not in placed:
                px, py = alloc_down(0, max_y + 1)
                placed[node_id] = (px, py)
                occupied.add((px, py))
                max_y = max(max_y, py)

        max_x = 0
        max_y = 0
        fallback_index = 0
        for row in self.rows:
            node_id = row.node_id_input.text().strip()
            if node_id:
                gx, gy = placed.get(node_id, (0, fallback_index))
            else:
                gx, gy = (0, fallback_index)
            fallback_index += 1
            px = self.canvas_margin + gx * (self.node_width + self.h_gap)
            py = self.canvas_margin + gy * (self.node_height + self.v_gap)
            row.setFixedWidth(self.node_width)
            row.setFixedHeight(self.node_height)
            row.move(px, py)
            max_x = max(max_x, px + self.node_width)
            max_y = max(max_y, py + self.node_height)

        view_w = self.scroll_.viewport().width()
        min_w = max(view_w - 4, max_x + self.canvas_margin)
        min_h = max(500, max_y + self.canvas_margin)
        if self.task_container.width() != min_w or self.task_container.height() != min_h:
            self.task_container.resize(min_w, min_h)
        if self.task_container.minimumSize().width() != min_w or self.task_container.minimumSize().height() != min_h:
            self.task_container.setMinimumSize(min_w, min_h)

    def refresh_arrows(self):
        if self._refreshing_arrows:
            return

        self._refreshing_arrows = True
        try:
            self.update_node_categories()
            self.auto_layout_nodes()
            self.arrow_overlay.setGeometry(self.task_container.rect())
            self.arrow_overlay.raise_()
            self.arrow_overlay.update()
        finally:
            self._refreshing_arrows = False

    def update_node_categories(self):
        if not self.rows:
            return

        node_ids = {
            row.node_id_input.text().strip()
            for row in self.rows
            if row.node_id_input.text().strip()
        }

        in_degree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
        out_degree: Dict[str, int] = {node_id: 0 for node_id in node_ids}

        for row in self.rows:
            src_id = row.node_id_input.text().strip()
            if not src_id or src_id not in node_ids:
                continue

            targets = {
                target
                for target in [row.next_input.text().strip(), row.fallback_input.text().strip()]
                if target and target in node_ids
            }

            out_degree[src_id] += len(targets)
            for target in targets:
                in_degree[target] += 1

        for row in self.rows:
            node_id = row.node_id_input.text().strip()
            if not node_id or node_id not in node_ids:
                row.set_category(NodeCategory.MIDDLE)
                continue

            indeg = in_degree.get(node_id, 0)
            outdeg = out_degree.get(node_id, 0)

            if indeg == 0:
                row.set_category(NodeCategory.START)
            elif outdeg == 0:
                row.set_category(NodeCategory.END)
            else:
                row.set_category(NodeCategory.MIDDLE)

    def add_row(self, data=None):
        row = TaskRow(self.task_container, self.delete_row)
        if data:
            row.set_data(data)
        else:
            row.node_id_input.setText(str(len(self.rows) + 1))
            row.next_input.setText(str(len(self.rows) + 2))

        row.changed.connect(self.refresh_arrows)
        self.rows.append(row)
        
        self.refresh_arrows()

    def delete_row(self, row_widget):
        if row_widget in self.rows:
            self.rows.remove(row_widget)
            row_widget.deleteLater()
            self.refresh_arrows()
            
    def save_config(self):
        nodes: list[WorkflowNode] = []
        for row in self.rows:
            data = row.get_data()
            nodes.append(data)
            
        if not nodes:
            QMessageBox.warning(self, "警告", "没有可保存的配置")
            return

        start_node = str(nodes[0].node_id).strip() or "1"
        payload = WorkflowConfig(start_node=start_node, nodes=nodes).to_dict()

        filename, _ = QFileDialog.getSaveFileName(self, "保存配置", os.getcwd(), "JSON Files (*.json);;Text Files (*.txt)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "成功", "配置已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "导入配置", os.getcwd(), "JSON Files (*.json);;Text Files (*.txt)")
        if not filename:
            return
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            workflow = WorkflowConfig.from_raw(config)

            # 清空现有行
            for row in self.rows:
                row.deleteLater()
            self.rows.clear()
            
            # 重新添加行
            for task in workflow.nodes:
                self.add_row(task)

            self.refresh_arrows()
                
            QMessageBox.information(self, "成功", f"成功导入 {len(workflow.nodes)} 条指令！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")

    def validate_flow(self, nodes):
        node_ids = []
        for idx, node in enumerate(nodes):
            node_id = str(node.node_id).strip()
            if not node_id:
                return f"第 {idx+1} 行缺少节点ID"
            node_ids.append(node_id)

        if len(node_ids) != len(set(node_ids)):
            return "节点ID存在重复"

        node_id_set = set(node_ids)
        for idx, node in enumerate(nodes):
            next_node = str(node.next).strip()
            fallback_next = str(node.fallback_next).strip()

            if next_node and next_node not in node_id_set:
                return f"第 {idx+1} 行的成功跳转节点不存在: {next_node}"
            if fallback_next and fallback_next not in node_id_set:
                return f"第 {idx+1} 行的失败跳转节点不存在: {fallback_next}"

        return ""

    def start_task(self):
        nodes: list[WorkflowNode] = []
        for row in self.rows:
            data = row.get_data()
            if not data.value:
                QMessageBox.warning(self, "警告", "请检查有空参数的指令！")
                return
            nodes.append(data)
            
        if not nodes:
            QMessageBox.warning(self, "警告", "请至少添加一条指令！")
            return

        error_msg = self.validate_flow(nodes)
        if error_msg:
            QMessageBox.warning(self, "警告", error_msg)
            return

        start_node = str(nodes[0].node_id).strip()
        workflow = WorkflowConfig(start_node=start_node, nodes=nodes)

        self.log_area.clear()
        self.log("任务开始...")
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_btn.setEnabled(False)
        
        loop = (self.loop_check.currentText() == "循环执行")
        
        self.worker = WorkerThread(self.engine, workflow, loop)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

        # 最小化窗口
        self.showMinimized()

    def stop_task(self):
        self.engine.stop()
        self.log("正在停止...")

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_btn.setEnabled(True)
        self.log("任务已结束")
        
        # 恢复窗口并置顶
        self.showNormal()
        self.activateWindow()

    def log(self, msg):
        self.log_area.append(msg)

    def closeEvent(self, event):
        """窗口关闭事件：确保线程停止，防止残留"""
        if self.worker and self.worker.isRunning():
            self.engine.stop()
            self.worker.quit()
            self.worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RPAWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
