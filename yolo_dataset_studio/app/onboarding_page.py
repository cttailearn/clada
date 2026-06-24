"""配置/引导页：选任务、选模型、选目录、配置类别。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.inference import list_input_images
from app.models import TaskType, YoloClass
from app.state import AppState, color_for_index


_TASK_INFO = {
    "detect": ("检测 (Detect)", "为图像中物体绘制矩形框，输出 cx cy w h", "📦"),
    "segment": ("分割 (Segment)", "为每个物体绘制多边形掩码", "🎯"),
    "classify": ("分类 (Classify)", "为整张图像分配一个类别", "🏷"),
}

_TASK_MODEL_HINT = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
    "classify": "yolo26n-cls.pt",
}


class _TaskCard(QFrame):
    """单个任务选择卡片。"""

    clicked = pyqtSignal(str)  # task key

    def __init__(self, task: TaskType, parent=None) -> None:
        super().__init__(parent)
        title, desc, icon = _TASK_INFO[task]
        self.task = task
        self._selected = False
        self.setObjectName("taskCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(140)
        self._update_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        icon_label = QLabel(icon)
        f = icon_label.font()
        f.setPointSize(22)
        icon_label.setFont(f)

        title_label = QLabel(title)
        f2 = title_label.font()
        f2.setBold(True)
        f2.setPointSize(13)
        title_label.setFont(f2)

        desc_label = QLabel(desc)
        desc_label.setObjectName("caption")
        desc_label.setWordWrap(True)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch(1)

    def mousePressEvent(self, ev) -> None:
        self.clicked.emit(self.task)
        super().mousePressEvent(ev)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "#taskCard { background-color:#1E3A5F; border:2px solid #22D3EE;"
                " border-radius:10px; }"
            )
        else:
            self.setStyleSheet(
                "#taskCard { background-color:#16213A; border:1px solid #1E293B;"
                " border-radius:10px; } "
                "#taskCard:hover { border-color:#334155; }"
            )


class OnboardingPage(QWidget):
    """步骤 1：配置任务/模型/目录/类别。"""

    request_next = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(20)

        heading = QLabel("配置数据集准备工作流")
        heading.setObjectName("heading")
        sub = QLabel(
            "选择任务类型、加载 YOLO26 预训练模型、设置输入与输出目录，并配置类别。"
        )
        sub.setObjectName("subheading")
        root.addWidget(heading)
        root.addWidget(sub)

        # ===== 任务卡片 =====
        task_row = QHBoxLayout()
        task_row.setSpacing(16)
        self.task_cards: dict[str, _TaskCard] = {}
        for t in ("detect", "segment", "classify"):
            card = _TaskCard(t)
            card.clicked.connect(self._on_task_clicked)
            self.task_cards[t] = card
            task_row.addWidget(card, 1)
        root.addLayout(task_row)

        # ===== 模型 + 目录 =====
        io_group = QGroupBox("模型与目录")
        io_form = QFormLayout(io_group)
        io_form.setVerticalSpacing(10)
        io_form.setHorizontalSpacing(12)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择 .pt 模型文件（如 yolo26n-seg.pt）")
        self.model_edit.setReadOnly(True)
        model_btn = QPushButton("浏览…")
        model_btn.clicked.connect(self._pick_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(model_btn)
        io_form.addRow(QLabel("模型 (.pt)"), self._wrap(model_row))

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("含待标注图片的目录")
        self.input_edit.setReadOnly(True)
        input_btn = QPushButton("浏览…")
        input_btn.clicked.connect(self._pick_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(input_btn)
        io_form.addRow(QLabel("输入目录"), self._wrap(input_row))

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("数据集与标签的输出目录")
        self.output_edit.setReadOnly(True)
        output_btn = QPushButton("浏览…")
        output_btn.clicked.connect(self._pick_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(output_btn)
        io_form.addRow(QLabel("输出目录"), self._wrap(output_row))

        self.count_label = QLabel("尚未扫描")
        self.count_label.setObjectName("caption")
        io_form.addRow(QLabel(""), self.count_label)

        root.addWidget(io_group)

        # ===== 类别 + 参数 =====
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        cls_group = QGroupBox("类别列表")
        cls_layout = QVBoxLayout(cls_group)
        self.class_list = QListWidget()
        cls_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ 添加")
        rename_btn = QPushButton("重命名")
        del_btn = QPushButton("删除")
        del_btn.setObjectName("dangerBtn")
        add_btn.clicked.connect(self._add_class)
        rename_btn.clicked.connect(self._rename_class)
        del_btn.clicked.connect(self._remove_class)
        cls_btn_row.addWidget(add_btn)
        cls_btn_row.addWidget(rename_btn)
        cls_btn_row.addWidget(del_btn)
        cls_btn_row.addStretch(1)
        self.single_cls_chk = QCheckBox("single_cls（训练时合并为单类）")
        self.single_cls_chk.toggled.connect(self.state.set_single_cls)
        cls_layout.addWidget(self.class_list)
        cls_layout.addLayout(cls_btn_row)
        cls_layout.addWidget(self.single_cls_chk)
        bottom_row.addWidget(cls_group, 2)

        param_group = QGroupBox("推理参数")
        param_form = QFormLayout(param_group)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.state.conf)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.state.iou)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.state.imgsz)
        self.device_edit = QLineEdit(self.state.device)
        self.device_edit.setPlaceholderText("auto / cuda / cpu / 0")
        param_form.addRow("conf 阈值", self.conf_spin)
        param_form.addRow("iou 阈值", self.iou_spin)
        param_form.addRow("imgsz", self.imgsz_spin)
        param_form.addRow("device", self.device_edit)
        bottom_row.addWidget(param_group, 1)

        root.addLayout(bottom_row)

        # ===== 下一步 =====
        next_row = QHBoxLayout()
        next_row.addStretch(1)
        self.next_btn = QPushButton("下一步：开始推理 →")
        self.next_btn.setObjectName("primaryBtn")
        self.next_btn.clicked.connect(self._go_next)
        next_row.addWidget(self.next_btn)
        root.addLayout(next_row)

        # 监听状态变化
        self.state.task_changed.connect(self._sync_task_cards)
        self.state.input_dir_changed.connect(self._refresh_count)
        self.state.classes_changed.connect(self._sync_class_list)

        # 默认值
        self.state.set_task("detect")
        self._sync_task_cards("detect")
        # 默认提供两个类别便于演示
        if not self.state.classes:
            self.state.set_classes([
                YoloClass(0, "object", color_for_index(0)),
                YoloClass(1, "other", color_for_index(1)),
            ])

    # ----- helpers -----
    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _on_task_clicked(self, task: str) -> None:
        self.state.set_task(task)  # type: ignore[arg-type]

    def _sync_task_cards(self, task: str) -> None:
        for k, card in self.task_cards.items():
            card.set_selected(k == task)

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", "", "PyTorch 权重 (*.pt)"
        )
        if path:
            p = Path(path)
            self.state.set_model_path(p)
            self.model_edit.setText(str(p))
            # 提示任务后缀
            name = p.name.lower()
            if name.endswith("-seg.pt") and self.state.task != "segment":
                QMessageBox.information(self, "提示",
                                        "模型名带 -seg 后缀，建议任务选择"
                                        "「分割」。")
            elif name.endswith("-cls.pt") and self.state.task != "classify":
                QMessageBox.information(self, "提示",
                                        "模型名带 -cls 后缀，建议任务选择"
                                        "「分类」。")

    def _pick_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输入图像目录")
        if path:
            self.state.set_input_dir(Path(path))
            self.input_edit.setText(path)

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出数据集目录")
        if path:
            self.state.set_output_dir(Path(path))
            self.output_edit.setText(path)

    def _refresh_count(self, _path: str) -> None:
        if not self.state.input_dir:
            self.count_label.setText("尚未扫描")
            return
        files = list_input_images(self.state.input_dir)
        self.count_label.setText(
            f"已识别 {len(files)} 张图片（递归扫描，含 jpg/png/bmp/webp/tif）"
        )

    def _sync_class_list(self, classes: list[YoloClass]) -> None:
        self.class_list.clear()
        for c in classes:
            it = QListWidgetItem(f"  ●  {c.id}  {c.name}")
            from PyQt6.QtGui import QColor
            it.setForeground(QColor(c.color))
            self.class_list.addItem(it)

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "新增类别", "类别名：")
        if ok and name.strip():
            self.state.add_class(name.strip())

    def _rename_class(self) -> None:
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.state.classes):
            return
        current = self.state.classes[row]
        name, ok = QInputDialog.getText(
            self, "重命名类别", "新名称：", text=current.name
        )
        if ok and name.strip():
            current.name = name.strip()
            self.state.set_classes(self.state.classes)

    def _remove_class(self) -> None:
        row = self.class_list.currentRow()
        if row >= 0:
            self.state.remove_class(row)

    def _go_next(self) -> None:
        # 把参数同步到 state
        self.state.conf = float(self.conf_spin.value())
        self.state.iou = float(self.iou_spin.value())
        self.state.imgsz = int(self.imgsz_spin.value())
        self.state.device = self.device_edit.text().strip() or "auto"
        ok, msg = self.state.is_ready_for_inference()
        if not ok:
            QMessageBox.warning(self, "无法继续", msg)
            return
        files = list_input_images(self.state.input_dir)  # type: ignore[arg-type]
        if not files:
            QMessageBox.warning(self, "无法继续", "输入目录未发现任何图片。")
            return
        self.request_next.emit()