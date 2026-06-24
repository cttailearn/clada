"""主窗口 — Stepper + QStackedWidget。"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.export_page import ExportPage
from app.inference_page import InferencePage
from app.onboarding_page import OnboardingPage
from app.review_page import ReviewPage
from app.state import AppState


STEP_NAMES = ("配置", "推理", "检视", "导出")


class _Stepper(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(72)
        self._current = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(36, 12, 36, 12)
        layout.setSpacing(0)

        self._labels: list[QLabel] = []
        self._dots: list[QLabel] = []
        for i, name in enumerate(STEP_NAMES):
            dot = QLabel(str(i + 1))
            dot.setFixedSize(32, 32)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._dots.append(dot)
            txt = QLabel(name)
            self._labels.append(txt)
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(dot)
            row.addWidget(txt)
            wrap = QWidget()
            wrap.setLayout(row)
            layout.addWidget(wrap)
            if i < len(STEP_NAMES) - 1:
                bar = QFrame()
                bar.setFrameShape(QFrame.Shape.HLine)
                bar.setStyleSheet("color:#334155;background:#1E293B;")
                bar.setFixedHeight(2)
                layout.addWidget(bar, 1)
        self.set_current(0)

    def set_current(self, index: int) -> None:
        self._current = index
        for i, (dot, txt) in enumerate(zip(self._dots, self._labels)):
            if i < index:
                dot.setStyleSheet(
                    "background:#22C55E;border-radius:16px;color:#0E1422;"
                    "font-weight:700;"
                )
                txt.setStyleSheet("color:#22C55E;font-weight:600;")
            elif i == index:
                dot.setStyleSheet(
                    "background:#22D3EE;border-radius:16px;color:#0E1422;"
                    "font-weight:700;"
                )
                txt.setStyleSheet("color:#22D3EE;font-weight:700;")
            else:
                dot.setStyleSheet(
                    "background:#1E293B;border-radius:16px;color:#64748B;"
                    "font-weight:500;border:1px solid #334155;"
                )
                txt.setStyleSheet("color:#64748B;")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Dataset Studio — YOLO26 数据集准备工具")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)

        self.state = AppState(self)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.setCentralWidget(central)

        # 顶部 banner
        banner = QFrame()
        banner.setStyleSheet("background:#0B1020;border-bottom:1px solid #1E293B;")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(28, 14, 28, 14)
        title = QLabel("⌬  YOLO Dataset Studio")
        f = title.font()
        f.setBold(True)
        f.setPointSize(13)
        title.setFont(f)
        title.setStyleSheet("color:#22D3EE;")
        subtitle = QLabel("YOLO26 预标注 · 校对 · 训练数据集生成")
        subtitle.setObjectName("caption")
        banner_layout.addWidget(title)
        banner_layout.addSpacing(16)
        banner_layout.addWidget(subtitle)
        banner_layout.addStretch(1)
        v.addWidget(banner)

        self.stepper = _Stepper()
        v.addWidget(self.stepper)

        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)

        # 四个页面
        self.page_onboard = OnboardingPage(self.state)
        self.page_inference = InferencePage(self.state)
        self.page_review = ReviewPage(self.state)
        self.page_export = ExportPage(self.state)
        self.stack.addWidget(self.page_onboard)
        self.stack.addWidget(self.page_inference)
        self.stack.addWidget(self.page_review)
        self.stack.addWidget(self.page_export)

        # 路由
        self.page_onboard.request_next.connect(lambda: self._go(1))
        self.page_inference.request_prev.connect(lambda: self._go(0))
        self.page_inference.request_next.connect(lambda: self._go(2))
        self.page_review.request_prev.connect(lambda: self._go(1))
        self.page_review.request_next.connect(lambda: self._go(3))
        self.page_export.request_prev.connect(lambda: self._go(2))

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪。请在「配置」页选择任务、模型与目录。")
        self.state.task_changed.connect(
            lambda t: self.status.showMessage(f"任务已切换为：{t}", 2000)
        )
        self.state.model_changed.connect(
            lambda p: self.status.showMessage(
                f"模型：{p if p else '(未选)'}", 3000
            )
        )

    def _go(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.stepper.set_current(index)
        if index == 1:
            self.page_inference.reset()
        elif index == 2:
            self.page_review.refresh()
        elif index == 3:
            self.page_export.refresh()