"""推理工作线程 — 使用 ultralytics YOLO 进行批量预标注。

若环境中没有 ultralytics 包，则回退到一个"演示推理"模式：
- 检测：在图像中心生成一个占图像 60% 的矩形框（类别 0）；
- 分割：在图像中心生成一个八边形（类别 0）；
- 分类：固定为类别 0。

这样即使用户暂时还没安装 PyTorch / ultralytics，也能体验完整的界面流程。
"""
from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Iterable, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.models import BBox, ClassLabel, ImageItem, Polygon, TaskType


SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_input_images(input_dir: Path) -> list[Path]:
    """列出输入目录下所有受支持的图像文件（递归）。"""
    if not input_dir or not input_dir.exists():
        return []
    files: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXT:
            files.append(p)
    return files


def _read_image_size(path: Path) -> tuple[int, int]:
    """读取图像尺寸（先尝试 Pillow，失败返回 0,0）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


class InferenceWorker(QThread):
    """在后台运行批量 YOLO 推理。"""

    progress = pyqtSignal(int, int, str)         # done, total, current_name
    item_done = pyqtSignal(int, object)          # index, ImageItem
    finished_all = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        model_path: Optional[Path],
        images: list[Path],
        task: TaskType,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        num_classes: int,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self.images = images
        self.task = task
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.num_classes = max(1, num_classes)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ----------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401  (QThread API)
        model = self._load_model()
        total = len(self.images)
        for idx, img_path in enumerate(self.images):
            if self._cancelled:
                break
            self.progress.emit(idx, total, img_path.name)
            item = self._build_item(img_path)
            try:
                if model is None:
                    self._fake_inference(item)
                else:
                    self._real_inference(model, item)
            except Exception as e:  # 单图失败不阻断整体
                item.error = f"{e}\n{traceback.format_exc(limit=2)}"
            self.item_done.emit(idx, item)
            self.progress.emit(idx + 1, total, img_path.name)

        self.finished_all.emit()

    # ----------------------------------------------------------------------
    def _build_item(self, img_path: Path) -> ImageItem:
        w, h = _read_image_size(img_path)
        return ImageItem(path=img_path, width=w, height=h)

    def _load_model(self):
        """尝试用 ultralytics 加载 YOLO 模型；失败返回 None（进入演示模式）。"""
        if self.model_path is None or not Path(self.model_path).exists():
            return None
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            return None
        try:
            model = YOLO(str(self.model_path))
            return model
        except Exception as e:  # 模型加载失败
            self.error.emit(f"模型加载失败：{e}\n将回退到演示推理。")
            return None

    # ----- 演示推理（无模型时） ----------------------------------------
    def _fake_inference(self, item: ImageItem) -> None:
        w, h = item.width, item.height
        if w <= 0 or h <= 0:
            return
        cls = 0
        if self.task == "detect":
            bw, bh = w * 0.6, h * 0.6
            x = (w - bw) / 2
            y = (h - bh) / 2
            item.bboxes.append(BBox(class_id=cls, x=x, y=y, w=bw, h=bh, conf=0.5))
        elif self.task == "segment":
            cx, cy = w / 2, h / 2
            rx, ry = w * 0.3, h * 0.3
            pts = []
            for i in range(8):
                ang = 2 * math.pi * i / 8
                pts.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))
            item.polygons.append(Polygon(class_id=cls, points=pts, conf=0.5))
        elif self.task == "classify":
            item.class_label = ClassLabel(class_id=cls, conf=0.5)

    # ----- 真实推理（ultralytics） -------------------------------------
    def _real_inference(self, model, item: ImageItem) -> None:
        device = None if self.device == "auto" else self.device
        kwargs = dict(
            source=str(item.path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if device:
            kwargs["device"] = device
        # 分类任务模型不支持 iou 参数；过滤即可
        if self.task == "classify":
            kwargs.pop("iou", None)

        results = model.predict(**kwargs)
        if not results:
            return
        result = results[0]

        if self.task == "detect":
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            cls_arr = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(float, xyxy[i])
                cls_id = int(cls_arr[i]) if i < len(cls_arr) else 0
                conf = float(conf_arr[i]) if i < len(conf_arr) else 1.0
                cls_id = min(cls_id, self.num_classes - 1)
                item.bboxes.append(
                    BBox(class_id=cls_id, x=x1, y=y1, w=x2 - x1, h=y2 - y1, conf=conf)
                )

        elif self.task == "segment":
            masks = getattr(result, "masks", None)
            boxes = getattr(result, "boxes", None)
            if masks is None or boxes is None:
                return
            xyn = masks.xy if hasattr(masks, "xy") else []  # 像素坐标列表
            cls_arr = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            for i, poly_pts in enumerate(xyn):
                if poly_pts is None or len(poly_pts) < 3:
                    continue
                pts = [(float(p[0]), float(p[1])) for p in poly_pts]
                cls_id = int(cls_arr[i]) if i < len(cls_arr) else 0
                conf = float(conf_arr[i]) if i < len(conf_arr) else 1.0
                cls_id = min(cls_id, self.num_classes - 1)
                item.polygons.append(Polygon(class_id=cls_id, points=pts, conf=conf))

        elif self.task == "classify":
            probs = getattr(result, "probs", None)
            if probs is None:
                return
            top1 = int(getattr(probs, "top1", 0))
            top1conf = float(getattr(probs, "top1conf", 1.0))
            top1 = min(top1, self.num_classes - 1)
            item.class_label = ClassLabel(class_id=top1, conf=top1conf)