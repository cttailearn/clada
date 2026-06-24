"""YOLO Dataset Studio 启动入口。"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许在源码目录直接运行：python yolo_dataset_studio/run.py
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import APP_STYLE


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Dataset Studio")
    app.setOrganizationName("clada")

    # 字体：优先中文友好字体，回退系统默认
    base_font = QFont()
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Segoe UI"):
        base_font.setFamily(family)
        if base_font.exactMatch():
            break
    base_font.setPointSize(10)
    app.setFont(base_font)

    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
