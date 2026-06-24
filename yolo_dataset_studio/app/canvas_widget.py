"""可缩放/可编辑的标注画布 — 基于 QGraphicsView。

支持：
- 鼠标滚轮缩放、空格+左键拖拽平移；
- 选择工具：点击标注高亮，按 Delete 删除；
- 矩形工具：左键拖拽绘制 BBox（用于检测/分割时作为辅助）；
- 多边形工具：左键依次添加顶点，右键或双击完成；
- 颜色按类别分配，编辑后通过信号回传。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.models import BBox, ImageItem, Polygon, YoloClass


# ---- 自定义图形项 -------------------------------------------------------

class _BBoxItem(QGraphicsItem):
    def __init__(self, bbox: BBox, color: QColor, parent_canvas: "AnnotationCanvas"):
        super().__init__()
        self.bbox = bbox
        self.color = color
        self.parent_canvas = parent_canvas
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(10)

    def boundingRect(self) -> QRectF:
        return QRectF(self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        rect = self.boundingRect()
        pen_color = QColor(self.color)
        pen_color.setAlpha(255 if self.isSelected() else 220)
        pen = QPen(pen_color, 2 if self.isSelected() else 1.5)
        painter.setPen(pen)
        fill = QColor(self.color)
        fill.setAlpha(60 if self.isSelected() else 35)
        painter.setBrush(QBrush(fill))
        painter.drawRect(rect)
        # 类别标签条
        label_h = 18
        label_rect = QRectF(rect.x(), rect.y() - label_h, max(60.0, rect.width()), label_h)
        painter.setBrush(QBrush(self.color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(label_rect)
        painter.setPen(QPen(QColor("#0E1422")))
        cname = self.parent_canvas.class_name(self.bbox.class_id)
        painter.drawText(label_rect.adjusted(6, 0, -6, 0),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         f"{cname}  {self.bbox.conf:.2f}")


class _PolygonItem(QGraphicsItem):
    def __init__(self, poly: Polygon, color: QColor, parent_canvas: "AnnotationCanvas"):
        super().__init__()
        self.poly = poly
        self.color = color
        self.parent_canvas = parent_canvas
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(10)

    def _qpoly(self) -> QPolygonF:
        return QPolygonF([QPointF(x, y) for x, y in self.poly.points])

    def boundingRect(self) -> QRectF:
        if not self.poly.points:
            return QRectF()
        xs = [p[0] for p in self.poly.points]
        ys = [p[1] for p in self.poly.points]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self.poly.points:
            return
        pen_color = QColor(self.color)
        pen_color.setAlpha(255 if self.isSelected() else 220)
        pen = QPen(pen_color, 2 if self.isSelected() else 1.5)
        painter.setPen(pen)
        fill = QColor(self.color)
        fill.setAlpha(70 if self.isSelected() else 40)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(self._qpoly())
        # 顶点
        painter.setBrush(QBrush(self.color))
        for x, y in self.poly.points:
            painter.drawEllipse(QPointF(x, y), 3, 3)


# ---- 主画布 -------------------------------------------------------------

class AnnotationCanvas(QGraphicsView):
    """标注画布 — 支持 select/rect/polygon 三种工具。"""

    MODE_SELECT = "select"
    MODE_RECT = "rect"
    MODE_POLY = "polygon"

    annotations_changed = pyqtSignal()  # 任意编辑后发出

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor("#0B1020")))
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

        self._classes: list[YoloClass] = []
        self._current_class_id: int = 0
        self._item: Optional[ImageItem] = None
        self._mode: str = self.MODE_SELECT
        self._space_pressed = False
        self._panning = False
        self._pan_last: Optional[QPointF] = None
        self._drag_start: Optional[QPointF] = None
        self._drag_rect_item = None
        self._poly_points: list[tuple[float, float]] = []
        self._poly_temp_items: list = []

    # ===== 对外配置 =====
    def set_classes(self, classes: list[YoloClass]) -> None:
        self._classes = classes
        if self._current_class_id >= len(classes):
            self._current_class_id = 0
        self._refresh()

    def set_current_class(self, class_id: int) -> None:
        self._current_class_id = max(0, class_id)

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id].name
        return f"#{class_id}"

    def class_color(self, class_id: int) -> QColor:
        if 0 <= class_id < len(self._classes):
            return QColor(self._classes[class_id].color)
        return QColor("#22D3EE")

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._reset_poly()

    def set_image(self, item: Optional[ImageItem]) -> None:
        self._item = item
        self._reset_poly()
        self._refresh()
        if item and self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ===== 内部刷新 =====
    def _refresh(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        if not self._item:
            return
        # 1) 图片底图
        pix = QPixmap(str(self._item.path))
        if pix.isNull():
            return
        if self._item.width <= 0 or self._item.height <= 0:
            self._item.width = pix.width()
            self._item.height = pix.height()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        # 2) BBox 标注
        for b in self._item.bboxes:
            self._scene.addItem(_BBoxItem(b, self.class_color(b.class_id), self))
        # 3) Polygon 标注
        for p in self._item.polygons:
            self._scene.addItem(_PolygonItem(p, self.class_color(p.class_id), self))

    # ===== 缩放 =====
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    # ===== 键盘 =====
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._reset_poly()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = False
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    # ===== 鼠标 =====
    def mousePressEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._space_pressed and event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_last = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            # 多边形完成
            if self._mode == self.MODE_POLY and len(self._poly_points) >= 3:
                self._commit_polygon()
            else:
                self._reset_poly()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == self.MODE_SELECT:
                super().mousePressEvent(event)
                return
            if self._mode == self.MODE_RECT and self._item:
                self._drag_start = scene_pos
                from PyQt6.QtWidgets import QGraphicsRectItem
                rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                pen = QPen(self.class_color(self._current_class_id), 1.5,
                           Qt.PenStyle.DashLine)
                rect_item.setPen(pen)
                self._scene.addItem(rect_item)
                self._drag_rect_item = rect_item
                return
            if self._mode == self.MODE_POLY and self._item:
                self._poly_points.append((scene_pos.x(), scene_pos.y()))
                self._render_poly_preview()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return

        if self._mode == self.MODE_RECT and self._drag_start and self._drag_rect_item:
            scene_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._drag_start, scene_pos).normalized()
            self._drag_rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_last = None
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self._space_pressed else Qt.CursorShape.ArrowCursor)
            return

        if (
            self._mode == self.MODE_RECT
            and event.button() == Qt.MouseButton.LeftButton
            and self._drag_start
            and self._drag_rect_item
            and self._item
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._drag_start, scene_pos).normalized()
            self._scene.removeItem(self._drag_rect_item)
            self._drag_rect_item = None
            self._drag_start = None
            if rect.width() > 4 and rect.height() > 4:
                # 限制到图片范围
                W, H = self._item.width, self._item.height
                x = max(0, min(rect.x(), W))
                y = max(0, min(rect.y(), H))
                w = min(rect.width(), W - x)
                h = min(rect.height(), H - y)
                bbox = BBox(class_id=self._current_class_id, x=x, y=y, w=w, h=h, conf=1.0)
                self._item.bboxes.append(bbox)
                self._refresh()
                self.annotations_changed.emit()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._mode == self.MODE_POLY and len(self._poly_points) >= 3:
            self._commit_polygon()
            return
        super().mouseDoubleClickEvent(event)

    # ===== 操作 =====
    def _delete_selected(self) -> None:
        if not self._item:
            return
        changed = False
        for it in list(self._scene.selectedItems()):
            if isinstance(it, _BBoxItem) and it.bbox in self._item.bboxes:
                self._item.bboxes.remove(it.bbox)
                changed = True
            elif isinstance(it, _PolygonItem) and it.poly in self._item.polygons:
                self._item.polygons.remove(it.poly)
                changed = True
        if changed:
            self._refresh()
            self.annotations_changed.emit()

    def delete_annotation(self, kind: str, index: int) -> None:
        """从外部（如标注列表）删除一项。"""
        if not self._item:
            return
        if kind == "bbox" and 0 <= index < len(self._item.bboxes):
            self._item.bboxes.pop(index)
        elif kind == "polygon" and 0 <= index < len(self._item.polygons):
            self._item.polygons.pop(index)
        else:
            return
        self._refresh()
        self.annotations_changed.emit()

    def _render_poly_preview(self) -> None:
        for it in self._poly_temp_items:
            self._scene.removeItem(it)
        self._poly_temp_items.clear()
        if not self._poly_points:
            return
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem
        color = self.class_color(self._current_class_id)
        pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        # 顶点
        for x, y in self._poly_points:
            d = QGraphicsEllipseItem(x - 3, y - 3, 6, 6)
            d.setBrush(QBrush(color))
            d.setPen(QPen(Qt.PenStyle.NoPen))
            self._scene.addItem(d)
            self._poly_temp_items.append(d)
        # 线段
        for i in range(len(self._poly_points) - 1):
            x1, y1 = self._poly_points[i]
            x2, y2 = self._poly_points[i + 1]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen)
            self._scene.addItem(line)
            self._poly_temp_items.append(line)

    def _commit_polygon(self) -> None:
        if not self._item or len(self._poly_points) < 3:
            self._reset_poly()
            return
        poly = Polygon(class_id=self._current_class_id,
                       points=list(self._poly_points), conf=1.0)
        self._item.polygons.append(poly)
        self._reset_poly()
        self._refresh()
        self.annotations_changed.emit()

    def _reset_poly(self) -> None:
        self._poly_points = []
        for it in self._poly_temp_items:
            self._scene.removeItem(it)
        self._poly_temp_items.clear()

    def fit_image(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)