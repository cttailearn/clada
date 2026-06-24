"""导出页 — 统计、划分、生成 data.yaml + 数据集目录结构。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.exporter import DatasetExporter
from app.state import AppState


class _StatCard(QFrame):
    def __init__(self, title: str, value: str = "0") -> None:
        super().__init__()
        self.setStyleSheet(
            "QFrame{background:#16213A;border:1px solid #1E293B;border-radius:10px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        title_label = QLabel(title)
        title_label.setObjectName("subheading")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statNumber")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ExportPage(QWidget):
    """步骤 4：生成数据集 + data.yaml。"""

    request_prev = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(20)

        heading = QLabel("导出 YOLO 训练数据集")
        heading.setObjectName("heading")
        sub = QLabel("按比例划分 train / val / test，生成 data.yaml 与训练命令说明。")
        sub.setObjectName("subheading")
        root.addWidget(heading)
        root.addWidget(sub)

        # 统计卡片
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.stat_total = _StatCard("图片总数")
        self.stat_reviewed = _StatCard("已审")
        self.stat_modified = _StatCard("已修改")
        self.stat_classes = _StatCard("类别数")
        stats_row.addWidget(self.stat_total)
        stats_row.addWidget(self.stat_reviewed)
        stats_row.addWidget(self.stat_modified)
        stats_row.addWidget(self.stat_classes)
        root.addLayout(stats_row)

        # 划分 + YAML
        body = QHBoxLayout()
        body.setSpacing(16)

        split_group = QGroupBox("数据集划分（百分比，自动归一化）")
        split_form = QFormLayout(split_group)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(self.state.split_train)
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(self.state.split_val)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(self.state.split_test)
        for s in (self.train_spin, self.val_spin, self.test_spin):
            s.valueChanged.connect(self._refresh_yaml)
        split_form.addRow("train (%)", self.train_spin)
        split_form.addRow("val (%)", self.val_spin)
        split_form.addRow("test (%)", self.test_spin)
        body.addWidget(split_group, 1)

        yaml_group = QGroupBox("data.yaml 预览")
        yaml_layout = QVBoxLayout(yaml_group)
        self.yaml_view = QTextEdit()
        self.yaml_view.setReadOnly(True)
        self.yaml_view.setStyleSheet(
            "QTextEdit{background:#0B1020;border:1px solid #1E293B;"
            "border-radius:8px;font-family:Consolas,monospace;}"
        )
        yaml_layout.addWidget(self.yaml_view)
        body.addWidget(yaml_group, 2)

        root.addLayout(body, 1)

        # 底部
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("← 上一步")
        self.prev_btn.clicked.connect(self.request_prev.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addStretch(1)
        self.export_btn = QPushButton("生成数据集")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.clicked.connect(self._do_export)
        bottom.addWidget(self.export_btn)
        root.addLayout(bottom)

        # 状态联动
        self.state.images_changed.connect(self.refresh)
        self.state.classes_changed.connect(lambda _=None: self.refresh())

    # ----- public -----
    def refresh(self) -> None:
        imgs = self.state.images
        self.stat_total.set_value(str(len(imgs)))
        self.stat_reviewed.set_value(
            str(sum(1 for it in imgs if it.reviewed))
        )
        self.stat_modified.set_value(
            str(sum(1 for it in imgs if it.modified))
        )
        self.stat_classes.set_value(str(len(self.state.classes)))
        self._refresh_yaml()

    def _refresh_yaml(self) -> None:
        # 把当前值同步到 state
        self.state.split_train = int(self.train_spin.value())
        self.state.split_val = int(self.val_spin.value())
        self.state.split_test = int(self.test_spin.value())
        if not self.state.output_dir or not self.state.classes:
            self.yaml_view.setPlainText("# 请先在前面步骤设置输出目录与类别")
            return
        exporter = DatasetExporter(
            output_dir=self.state.output_dir,
            images=self.state.images,
            classes=self.state.classes,
            task=self.state.task,
            split_train=self.state.split_train,
            split_val=self.state.split_val,
            split_test=self.state.split_test,
        )
        self.yaml_view.setPlainText(exporter.build_yaml())

    def _do_export(self) -> None:
        if not self.state.output_dir:
            QMessageBox.warning(self, "失败", "未设置输出目录。")
            return
        if not self.state.images:
            QMessageBox.warning(self, "失败", "没有可导出的图像。")
            return
        if not self.state.classes:
            QMessageBox.warning(self, "失败", "请先配置类别。")
            return

        confirm = QMessageBox.question(
            self,
            "确认导出",
            f"将复制 {len(self.state.images)} 张图片到\n"
            f"{self.state.output_dir}\n并生成 data.yaml。是否继续？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            exporter = DatasetExporter(
                output_dir=self.state.output_dir,
                images=self.state.images,
                classes=self.state.classes,
                task=self.state.task,
                split_train=self.state.split_train,
                split_val=self.state.split_val,
                split_test=self.state.split_test,
            )
            exporter.run()
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        QMessageBox.information(
            self,
            "完成",
            f"数据集已生成于：\n{self.state.output_dir}\n\n"
            f"可执行：\n  yolo {self.state.task} train data="
            f"{'.' if self.state.task == 'classify' else 'data.yaml'} "
            f"model=yolo26n"
            f"{'-seg' if self.state.task == 'segment' else ('-cls' if self.state.task == 'classify' else '')}.pt "
            f"epochs=100",
        )