"""检视页：左缩略图 / 中画布 / 右标注列表。

任意编辑会自动通过 label_io 写入 YOLO 标签 .txt 文件（分类模式只更新内存）。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app import label_io
from app.canvas_widget import AnnotationCanvas
from app.models import ClassLabel, ImageItem
from app.state import AppState


class ReviewPage(QWidget):
    """步骤 3：分页查看并修改预标注结果。"""

    request_next = pyqtSignal()
    request_prev = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        # 顶部工具栏
        outer.addLayout(self._build_top_bar())

        # 三栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        self.thumb_list = QListWidget()
        self.thumb_list.setMaximumWidth(260)
        self.thumb_list.setMinimumWidth(200)
        self.thumb_list.currentRowChanged.connect(self._on_thumb_changed)
        splitter.addWidget(self.thumb_list)

        self.canvas = AnnotationCanvas()
        self.canvas.annotations_changed.connect(self._on_canvas_changed)
        splitter.addWidget(self.canvas)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("当前图标注")
        title.setObjectName("subheading")
        right_layout.addWidget(title)
        self.ann_list = QListWidget()
        right_layout.addWidget(self.ann_list, 1)
        ann_btns = QHBoxLayout()
        self.del_ann_btn = QPushButton("删除选中")
        self.del_ann_btn.setObjectName("dangerBtn")
        self.del_ann_btn.clicked.connect(self._delete_selected_ann)
        ann_btns.addWidget(self.del_ann_btn)
        ann_btns.addStretch(1)
        right_layout.addLayout(ann_btns)
        right_panel.setMaximumWidth(280)
        right_panel.setMinimumWidth(220)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        outer.addWidget(splitter, 1)

        # 底部
        outer.addLayout(self._build_bottom_bar())

        # 监听状态
        self.state.classes_changed.connect(self._sync_classes)
        self.state.images_changed.connect(self._rebuild_thumbs)
        self.state.image_updated.connect(self._on_image_updated)
        self.state.current_index_changed.connect(self._on_current_changed)

        # 快捷键
        QShortcut(QKeySequence("A"), self, activated=self._prev_image)
        QShortcut(QKeySequence("D"), self, activated=self._next_image)
        QShortcut(QKeySequence("1"), self,
                  activated=lambda: self._set_mode(AnnotationCanvas.MODE_SELECT))
        QShortcut(QKeySequence("2"), self,
                  activated=lambda: self._set_mode(AnnotationCanvas.MODE_RECT))
        QShortcut(QKeySequence("3"), self,
                  activated=lambda: self._set_mode(AnnotationCanvas.MODE_POLY))
        QShortcut(QKeySequence("F"), self, activated=self._mark_reviewed)

    # ---------------------------------------------------------------- 顶部
    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        def _make(label: str, mode: str, hotkey: str) -> QPushButton:
            btn = QPushButton(f"{label}  [{hotkey}]")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            self.tool_group.addButton(btn)
            return btn

        self.btn_select = _make("选择", AnnotationCanvas.MODE_SELECT, "1")
        self.btn_rect = _make("矩形", AnnotationCanvas.MODE_RECT, "2")
        self.btn_poly = _make("多边形", AnnotationCanvas.MODE_POLY, "3")
        self.btn_select.setChecked(True)
        bar.addWidget(self.btn_select)
        bar.addWidget(self.btn_rect)
        bar.addWidget(self.btn_poly)

        bar.addSpacing(16)
        bar.addWidget(QLabel("当前类别:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(180)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        bar.addWidget(self.class_combo)

        bar.addStretch(1)

        self.fit_btn = QPushButton("适配视图")
        self.fit_btn.clicked.connect(self.canvas.fit_image)
        bar.addWidget(self.fit_btn)

        self.review_btn = QPushButton("标记已审  [F]")
        self.review_btn.clicked.connect(self._mark_reviewed)
        bar.addWidget(self.review_btn)
        return bar

    def _build_bottom_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.prev_step_btn = QPushButton("← 上一步")
        self.prev_step_btn.clicked.connect(self.request_prev.emit)
        bar.addWidget(self.prev_step_btn)

        bar.addStretch(1)

        self.info_label = QLabel("—")
        self.info_label.setObjectName("caption")
        bar.addWidget(self.info_label)

        bar.addStretch(1)

        self.prev_img_btn = QPushButton("← 上一张 [A]")
        self.next_img_btn = QPushButton("下一张 [D] →")
        self.prev_img_btn.clicked.connect(self._prev_image)
        self.next_img_btn.clicked.connect(self._next_image)
        bar.addWidget(self.prev_img_btn)
        bar.addWidget(self.next_img_btn)

        self.next_step_btn = QPushButton("下一步：导出数据集 →")
        self.next_step_btn.setObjectName("primaryBtn")
        self.next_step_btn.clicked.connect(self.request_next.emit)
        bar.addWidget(self.next_step_btn)
        return bar

    # ---------------------------------------------------------------- 状态同步
    def refresh(self) -> None:
        self._sync_classes(self.state.classes)
        self._rebuild_thumbs()
        if self.state.images and self.state.current_index < 0:
            self.state.set_current_index(0)
        self._update_view()

    def _sync_classes(self, classes) -> None:
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for c in classes:
            self.class_combo.addItem(f"{c.id}  {c.name}", c.id)
        self.class_combo.blockSignals(False)
        self.canvas.set_classes(classes)
        if classes:
            self.canvas.set_current_class(0)

    def _rebuild_thumbs(self) -> None:
        self.thumb_list.blockSignals(True)
        self.thumb_list.clear()
        for i, item in enumerate(self.state.images):
            li = QListWidgetItem(self._thumb_label(item))
            self._style_thumb(li, item)
            self.thumb_list.addItem(li)
        self.thumb_list.blockSignals(False)
        if self.state.current_index >= 0:
            self.thumb_list.setCurrentRow(self.state.current_index)

    def _thumb_label(self, item: ImageItem) -> str:
        badge = "○"
        if item.modified:
            badge = "✎"
        elif item.reviewed:
            badge = "✓"
        if self.state.task == "classify":
            count = 1 if item.class_label is not None else 0
        elif self.state.task == "segment":
            count = len(item.polygons)
        else:
            count = len(item.bboxes)
        return f" {badge}  {item.path.name}    ({count})"

    def _style_thumb(self, li: QListWidgetItem, item: ImageItem) -> None:
        if item.error:
            li.setForeground(QColor("#F87171"))
        elif item.modified:
            li.setForeground(QColor("#FBBF24"))
        elif item.reviewed:
            li.setForeground(QColor("#22C55E"))
        else:
            li.setForeground(QColor("#E5E7EB"))

    def _on_thumb_changed(self, row: int) -> None:
        if row != self.state.current_index:
            self.state.set_current_index(row)

    def _on_current_changed(self, _row: int) -> None:
        self._update_view()
        if self.thumb_list.currentRow() != self.state.current_index:
            self.thumb_list.blockSignals(True)
            self.thumb_list.setCurrentRow(self.state.current_index)
            self.thumb_list.blockSignals(False)

    def _on_image_updated(self, idx: int) -> None:
        if idx < self.thumb_list.count():
            li = self.thumb_list.item(idx)
            li.setText(self._thumb_label(self.state.images[idx]))
            self._style_thumb(li, self.state.images[idx])
        if idx == self.state.current_index:
            self._refresh_ann_list()
            self._update_info()

    # ---------------------------------------------------------------- 编辑
    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        if mode == AnnotationCanvas.MODE_SELECT:
            self.btn_select.setChecked(True)
        elif mode == AnnotationCanvas.MODE_RECT:
            self.btn_rect.setChecked(True)
        else:
            self.btn_poly.setChecked(True)

    def _on_class_changed(self, _index: int) -> None:
        cid = self.class_combo.currentData()
        if cid is not None:
            self.canvas.set_current_class(int(cid))
            # 分类任务：直接更新当前图的类别标签
            item = self.state.current_image
            if self.state.task == "classify" and item is not None:
                item.class_label = ClassLabel(class_id=int(cid), conf=1.0)
                item.modified = True
                self._auto_save(item)
                self.state.image_updated.emit(self.state.current_index)

    def _on_canvas_changed(self) -> None:
        item = self.state.current_image
        if item is None:
            return
        item.modified = True
        self._auto_save(item)
        self.state.image_updated.emit(self.state.current_index)

    def _auto_save(self, item: ImageItem) -> None:
        if self.state.output_dir is None:
            return
        try:
            label_io.write_labels(item, self.state.output_dir, self.state.task)
        except Exception as e:
            print(f"[warn] auto save failed: {e}")

    def _update_view(self) -> None:
        item = self.state.current_image
        self.canvas.set_image(item)
        self._refresh_ann_list()
        self._update_info()

    def _refresh_ann_list(self) -> None:
        self.ann_list.clear()
        item = self.state.current_image
        if item is None:
            return
        if self.state.task == "classify":
            if item.class_label:
                cid = item.class_label.class_id
                color = self.canvas.class_color(cid).name()
                cname = self.canvas.class_name(cid)
                li = QListWidgetItem(f"  ●  {cname}  ({item.class_label.conf:.2f})")
                li.setForeground(QColor(color))
                self.ann_list.addItem(li)
        elif self.state.task == "detect":
            for i, b in enumerate(item.bboxes):
                cname = self.canvas.class_name(b.class_id)
                li = QListWidgetItem(
                    f"  ●  [{i}] {cname}  {int(b.w)}×{int(b.h)}  ({b.conf:.2f})"
                )
                li.setForeground(QColor(self.canvas.class_color(b.class_id).name()))
                li.setData(Qt.ItemDataRole.UserRole, ("bbox", i))
                self.ann_list.addItem(li)
        elif self.state.task == "segment":
            for i, p in enumerate(item.polygons):
                cname = self.canvas.class_name(p.class_id)
                li = QListWidgetItem(
                    f"  ●  [{i}] {cname}  pts={len(p.points)}  ({p.conf:.2f})"
                )
                li.setForeground(QColor(self.canvas.class_color(p.class_id).name()))
                li.setData(Qt.ItemDataRole.UserRole, ("polygon", i))
                self.ann_list.addItem(li)

    def _delete_selected_ann(self) -> None:
        li = self.ann_list.currentItem()
        if li is None:
            return
        data = li.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, idx = data
        self.canvas.delete_annotation(kind, idx)

    def _mark_reviewed(self) -> None:
        idx = self.state.current_index
        if idx < 0:
            return
        self.state.mark_reviewed(idx, True)

    def _prev_image(self) -> None:
        idx = self.state.current_index - 1
        if idx >= 0:
            self.state.set_current_index(idx)

    def _next_image(self) -> None:
        idx = self.state.current_index + 1
        if idx < len(self.state.images):
            self.state.set_current_index(idx)

    def _update_info(self) -> None:
        idx = self.state.current_index
        total = len(self.state.images)
        if idx < 0 or total == 0:
            self.info_label.setText("—")
            return
        item = self.state.images[idx]
        self.info_label.setText(
            f"[{idx + 1}/{total}]  {item.path.name}   "
            f"{item.width}×{item.height}   {'★已审' if item.reviewed else '未审'}"
        )