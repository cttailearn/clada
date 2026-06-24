"""数据模型 — 标注、类别、图像条目的 dataclass 定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TaskType = Literal["detect", "segment", "classify"]


@dataclass
class YoloClass:
    """YOLO 数据集的类别定义。"""
    id: int
    name: str
    color: str = "#FF6B6B"  # 默认颜色，由调用方分配


@dataclass
class BBox:
    """矩形标注（检测任务）。"""
    class_id: int
    x: float  # 像素坐标
    y: float
    w: float
    h: float
    conf: float = 1.0


@dataclass
class Polygon:
    """多边形标注（分割任务）。"""
    class_id: int
    points: list[tuple[float, float]] = field(default_factory=list)
    conf: float = 1.0


@dataclass
class ClassLabel:
    """整图分类标签（分类任务）。"""
    class_id: int
    conf: float = 1.0


@dataclass
class ImageItem:
    """单张输入图片的全部状态。"""
    path: Path               # 输入目录下的原图完整路径
    width: int = 0
    height: int = 0
    bboxes: list[BBox] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    class_label: ClassLabel | None = None
    reviewed: bool = False
    modified: bool = False   # 相对原始预标注结果是否有改动
    error: str | None = None # 推理失败时的错误信息