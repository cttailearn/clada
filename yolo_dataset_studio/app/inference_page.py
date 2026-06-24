"""推理页 — 启动 InferenceWorker，显示进度与日志。"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.inference import InferenceWorker, list_input_images
from app.models import ImageItem
from app.state import AppState


class InferencePage(QWidget):
    """步骤 2：批量预标注。"""

    request_next = pyqtSignal()
    request_prev = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: InferenceWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(20)

        heading = QLabel("预标注推理")
        heading.setObjectName("heading")
        sub = QLabel("在后台线程对全部图片执行 YOLO 推理，可随时取消。")
        sub.setObjectName("subheading")
        root.addWidget(heading)
        root.addWidget(sub)

        # 进度
        self.percent_label = QLabel("0%")
        f = self.percent_label.font()
        f.setPointSize(36)
        f.setBold(True)
        self.percent_label.setFont(f)
        self.percent_label.setStyleSheet("color:#22D3EE;")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.percent_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.status_label = QLabel("尚未开始")
        self.status_label.setObjectName("caption")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        # 日志
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "QTextEdit{background:#0B1020;border:1px solid #1E293B;"
            "border-radius:8px;font-family:Consolas,monospace;}"
        )
        self.log_box.setMinimumHeight(180)
        root.addWidget(self.log_box, 1)

        # 控制按钮
        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("← 上一步")
        self.start_btn = QPushButton("开始推理")
        self.start_btn.setObjectName("primaryBtn")
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("dangerBtn")
        self.next_btn = QPushButton("下一步：检视与编辑 →")
        self.next_btn.setObjectName("primaryBtn")
        self.next_btn.setEnabled(False)

        self.prev_btn.clicked.connect(self.request_prev.emit)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        self.next_btn.clicked.connect(self.request_next.emit)
        self.cancel_btn.setEnabled(False)

        btn_row.addWidget(self.prev_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.next_btn)
        root.addLayout(btn_row)

    # ----- public -----
    def reset(self) -> None:
        self.progress.setValue(0)
        self.percent_label.setText("0%")
        self.status_label.setText("准备就绪")
        self.log_box.clear()
        self.start_btn.setEnabled(True)
        self.next_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    # ----- worker -----
    def _start(self) -> None:
        if not self.state.input_dir:
            return
        files = list_input_images(self.state.input_dir)
        if not files:
            self._log("输入目录无图片，无法开始。")
            return

        # 初始化 image items 列表（先填占位，让 review 页能立即用到）
        self.state.set_images([ImageItem(path=p) for p in files])

        self._log(
            f"开始推理，共 {len(files)} 张图片，任务 = {self.state.task}，"
            f"模型 = {self.state.model_path or '(无 — 演示模式)'}"
        )

        self.worker = InferenceWorker(
            model_path=self.state.model_path,
            images=files,
            task=self.state.task,
            conf=self.state.conf,
            iou=self.state.iou,
            imgsz=self.state.imgsz,
            device=self.state.device,
            num_classes=max(1, len(self.state.classes)),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.item_done.connect(self._on_item_done)
        self.worker.error.connect(self._log)
        self.worker.finished_all.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self._log("已请求取消…")

    def _on_progress(self, done: int, total: int, name: str) -> None:
        pct = int(done * 100 / max(1, total))
        self.progress.setValue(pct)
        self.percent_label.setText(f"{pct}%")
        self.status_label.setText(f"{done}/{total} — 正在处理：{name}")

    def _on_item_done(self, idx: int, item: ImageItem) -> None:
        self.state.update_image(idx, item)
        n_box = len(item.bboxes)
        n_poly = len(item.polygons)
        cls_txt = f" class={item.class_label.class_id}" if item.class_label else ""
        if item.error:
            self._log(f"  ✗ {item.path.name}：{item.error.splitlines()[0]}")
        else:
            self._log(f"  ✓ {item.path.name}  bboxes={n_box} polys={n_poly}{cls_txt}")

    def _on_finished(self) -> None:
        self._log("推理完成。")
        self.cancel_btn.setEnabled(False)
        self.next_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

    def _log(self, text: str) -> None:
        self.log_box.append(text)