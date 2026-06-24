"""全局 AppState — 配置 / 类别 / 图像列表 / 当前选择，通过 pyqtSignal 广播变更。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from app.models import ImageItem, TaskType, YoloClass


# 预设调色板（高对比度循环使用）
_PALETTE = [
    "#22D3EE", "#F472B6", "#FACC15", "#A78BFA", "#34D399",
    "#FB923C", "#60A5FA", "#F87171", "#C084FC", "#10B981",
    "#FCD34D", "#06B6D4", "#EC4899", "#84CC16", "#F59E0B",
]


def color_for_index(index: int) -> str:
    """根据类别 ID 获取一个稳定的颜色。"""
    return _PALETTE[index % len(_PALETTE)]


class AppState(QObject):
    """整个应用的可观察状态。"""

    task_changed = pyqtSignal(str)                       # "detect"/"segment"/"classify"
    model_changed = pyqtSignal(str)                      # 模型路径（或空字符串）
    input_dir_changed = pyqtSignal(str)
    output_dir_changed = pyqtSignal(str)
    classes_changed = pyqtSignal(list)                   # list[YoloClass]
    images_changed = pyqtSignal()                        # 全列表变化
    image_updated = pyqtSignal(int)                      # 单条更新（索引）
    current_index_changed = pyqtSignal(int)
    inference_progress = pyqtSignal(int, int, str)       # done, total, current name
    inference_finished = pyqtSignal()
    log = pyqtSignal(str)                                # 普通文字日志

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._task: TaskType = "detect"
        self._model_path: Optional[Path] = None
        self._input_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._classes: list[YoloClass] = []
        self._single_cls: bool = False
        self._images: list[ImageItem] = []
        self._current_index: int = -1

        # 推理参数（默认）
        self.conf: float = 0.25
        self.iou: float = 0.45
        self.imgsz: int = 640
        self.device: str = "auto"   # "auto" / "cuda" / "cpu"
        self.batch: int = 1

        # 导出参数（百分比）
        self.split_train: int = 80
        self.split_val: int = 10
        self.split_test: int = 10

    # ===== 任务 =====
    @property
    def task(self) -> TaskType:
        return self._task

    def set_task(self, task: TaskType) -> None:
        if task != self._task:
            self._task = task
            self.task_changed.emit(task)

    # ===== 模型路径 =====
    @property
    def model_path(self) -> Optional[Path]:
        return self._model_path

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self.model_changed.emit(str(path) if path else "")

    # ===== 输入/输出目录 =====
    @property
    def input_dir(self) -> Optional[Path]:
        return self._input_dir

    def set_input_dir(self, path: Optional[Path]) -> None:
        self._input_dir = path
        self.input_dir_changed.emit(str(path) if path else "")

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir

    def set_output_dir(self, path: Optional[Path]) -> None:
        self._output_dir = path
        self.output_dir_changed.emit(str(path) if path else "")

    # ===== 类别 =====
    @property
    def classes(self) -> list[YoloClass]:
        return self._classes

    def set_classes(self, classes: list[YoloClass]) -> None:
        # 重新为颜色赋值，保证一致性
        for i, c in enumerate(classes):
            c.id = i
            c.color = color_for_index(i)
        self._classes = classes
        self.classes_changed.emit(classes)

    def add_class(self, name: str) -> None:
        new_id = len(self._classes)
        self._classes.append(
            YoloClass(id=new_id, name=name, color=color_for_index(new_id))
        )
        self.classes_changed.emit(self._classes)

    def remove_class(self, index: int) -> None:
        if 0 <= index < len(self._classes):
            self._classes.pop(index)
            for i, c in enumerate(self._classes):
                c.id = i
                c.color = color_for_index(i)
            self.classes_changed.emit(self._classes)

    @property
    def single_cls(self) -> bool:
        return self._single_cls

    def set_single_cls(self, value: bool) -> None:
        self._single_cls = value

    # ===== 图像列表 =====
    @property
    def images(self) -> list[ImageItem]:
        return self._images

    def set_images(self, items: list[ImageItem]) -> None:
        self._images = items
        self._current_index = 0 if items else -1
        self.images_changed.emit()
        self.current_index_changed.emit(self._current_index)

    def clear_images(self) -> None:
        self._images = []
        self._current_index = -1
        self.images_changed.emit()
        self.current_index_changed.emit(-1)

    def update_image(self, index: int, item: ImageItem) -> None:
        if 0 <= index < len(self._images):
            self._images[index] = item
            self.image_updated.emit(index)

    def mark_modified(self, index: int) -> None:
        if 0 <= index < len(self._images):
            self._images[index].modified = True
            self.image_updated.emit(index)

    def mark_reviewed(self, index: int, reviewed: bool = True) -> None:
        if 0 <= index < len(self._images):
            self._images[index].reviewed = reviewed
            self.image_updated.emit(index)

    # ===== 当前选中 =====
    @property
    def current_index(self) -> int:
        return self._current_index

    def set_current_index(self, index: int) -> None:
        if -1 <= index < len(self._images) and index != self._current_index:
            self._current_index = index
            self.current_index_changed.emit(index)

    @property
    def current_image(self) -> Optional[ImageItem]:
        if 0 <= self._current_index < len(self._images):
            return self._images[self._current_index]
        return None

    # ===== 校验 =====
    def is_ready_for_inference(self) -> tuple[bool, str]:
        if not self._input_dir or not self._input_dir.exists():
            return False, "未设置输入目录"
        if not self._output_dir:
            return False, "未设置输出目录"
        if not self._classes:
            return False, "请至少配置一个类别"
        return True, ""