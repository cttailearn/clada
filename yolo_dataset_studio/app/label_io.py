"""YOLO 标签 .txt 文件的读写。"""
from __future__ import annotations

from pathlib import Path

from app.models import BBox, ClassLabel, ImageItem, Polygon, TaskType


def write_labels(item: ImageItem, output_dir: Path, task: TaskType) -> None:
    """将 ImageItem 中的标注写入 YOLO 格式 .txt 文件。

    归一化坐标写入：<output_dir>/labels/{item.path.stem}.txt
    """
    if task == "classify":
        # 分类任务无 .txt 标签文件
        return

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    txt_path = labels_dir / f"{item.path.stem}.txt"
    w, h = item.width, item.height
    if w <= 0 or h <= 0:
        return

    lines: list[str] = []
    if task == "detect":
        for bbox in item.bboxes:
            cx = (bbox.x + bbox.w / 2) / w
            cy = (bbox.y + bbox.h / 2) / h
            nw = bbox.w / w
            nh = bbox.h / h
            lines.append(f"{bbox.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    elif task == "segment":
        for poly in item.polygons:
            pts_str = " ".join(
                f"{px / w:.6f} {py / h:.6f}" for px, py in poly.points
            )
            lines.append(f"{poly.class_id} {pts_str}")
    else:
        return

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def read_labels(item: ImageItem, output_dir: Path, task: TaskType) -> None:
    """从 YOLO 格式 .txt 文件读取标注到 ImageItem。

    - 检测：class cx cy w h  → BBox（转换为像素坐标）
    - 分割：class x1 y1 ...  → Polygon
    - 分类：不读取 txt（直接跳过）
    """
    if task == "classify":
        return

    labels_dir = output_dir / "labels"
    txt_path = labels_dir / f"{item.path.stem}.txt"
    if not txt_path.exists():
        return

    w, h = item.width, item.height
    if w <= 0 or h <= 0:
        return

    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue

        if task == "detect":
            cx, cy, nw, nh = map(float, parts[1:5])
            x = (cx - nw / 2) * w
            y = (cy - nh / 2) * h
            bw = nw * w
            bh = nh * h
            item.bboxes.append(BBox(class_id=cls_id, x=x, y=y, w=bw, h=bh))

        elif task == "segment":
            coords = list(map(float, parts[1:]))
            if len(coords) >= 4 and len(coords) % 2 == 0:
                points = [
                    (coords[i] * w, coords[i + 1] * h)
                    for i in range(0, len(coords), 2)
                ]
                item.polygons.append(Polygon(class_id=cls_id, points=points))

    item.modified = False


def ensure_labels_directory(output_dir: Path) -> None:
    """确保输出目录存在 labels 子目录。"""
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)