"""全局 QSS 主题 — 暗色、电光青主色。"""

APP_STYLE = """
/* ===== 全局 ===== */
QMainWindow, QDialog {
    background-color: #0E1422;
    color: #E5E7EB;
}
QWidget {
    background-color: transparent;
    color: #E5E7EB;
    font-size: 10pt;
}

/* ===== 按钮 ===== */
QPushButton {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 20px;
    color: #E5E7EB;
    font-weight: 500;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #273548;
    border-color: #22D3EE;
    color: #22D3EE;
}
QPushButton:pressed {
    background-color: #1A2A40;
}
QPushButton:disabled {
    background-color: #1A1F2E;
    color: #475569;
    border-color: #1E293B;
}
QPushButton#primaryBtn {
    background-color: #0891B2;
    border: none;
    color: #FFFFFF;
}
QPushButton#primaryBtn:hover {
    background-color: #22D3EE;
}
QPushButton#primaryBtn:disabled {
    background-color: #1E3A5F;
    color: #64748B;
}
QPushButton#dangerBtn {
    background-color: #991B1B;
    border: none;
    color: #FCA5A5;
}
QPushButton#dangerBtn:hover {
    background-color: #B91C1C;
}

/* ===== 输入框 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #E5E7EB;
    selection-background-color: #0891B2;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #22D3EE;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1E293B;
    border: 1px solid #334155;
    selection-background-color: #0891B2;
    selection-color: #FFFFFF;
}

/* ===== 列表 ===== */
QListWidget {
    background-color: #16213A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    outline: none;
}
QListWidget::item {
    border-radius: 6px;
    padding: 6px 10px;
    margin: 2px 4px;
}
QListWidget::item:selected {
    background-color: #1E3A5F;
    color: #22D3EE;
}
QListWidget::item:hover {
    background-color: #1E293B;
}

/* ===== 树/表 ===== */
QTreeWidget, QTableWidget {
    background-color: #16213A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    gridline-color: #1E293B;
}
QHeaderView::section {
    background-color: #1E293B;
    border: none;
    padding: 6px 12px;
    color: #94A3B8;
    font-weight: 600;
}

/* ===== 选项卡 ===== */
QTabWidget::pane {
    background-color: #0E1422;
    border: none;
}
QTabBar::tab {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 8px 24px;
    margin-right: 2px;
    color: #94A3B8;
}
QTabBar::tab:selected {
    background-color: #16213A;
    color: #22D3EE;
    border-bottom: 2px solid #22D3EE;
}

/* ===== 进度条 ===== */
QProgressBar {
    background-color: #1E293B;
    border: none;
    border-radius: 6px;
    text-align: center;
    color: #E5E7EB;
    font-weight: 600;
    min-height: 22px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0891B2, stop:1 #22D3EE);
    border-radius: 6px;
}

/* ===== 滑动条 ===== */
QSlider::groove:horizontal {
    background: #1E293B;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #22D3EE;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

/* ===== 分组框 ===== */
QGroupBox {
    border: 1px solid #1E293B;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 12px;
    font-weight: 600;
    color: #94A3B8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ===== 标签/状态栏 ===== */
QLabel {
    color: #E5E7EB;
}
QLabel#heading {
    font-size: 18pt;
    font-weight: 700;
    color: #F1F5F9;
}
QLabel#subheading {
    font-size: 11pt;
    color: #94A3B8;
}
QLabel#caption {
    font-size: 9pt;
    color: #64748B;
}
QLabel#statNumber {
    font-size: 28pt;
    font-weight: 700;
    color: #22D3EE;
}
QStatusBar {
    background-color: #0B1020;
    color: #94A3B8;
    font-size: 9pt;
}

/* ===== 滚动条 ===== */
QScrollBar:vertical {
    background: #0E1422;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #0E1422;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""