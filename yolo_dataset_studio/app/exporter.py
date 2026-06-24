"""数据集导出器 — 按比例划分图片+标签，生成 data.yaml。"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import yaml

from app.models import ImageItem, TaskType, YoloClass


class DatasetExporter:
    """把当前会话中的图像与标签导出为 Ultralytics 兼容数据集。"""

    def __init__(
        self,
        output_dir: Path,
        images: list[ImageItem],
        classes: list[YoloClass],
        task: TaskType,
        split_train: int = 80,
        split_val: int = 10,
        split_test: int = 10,
        seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.images = images
        self.classes = classes
        self.task = task
        total = max(1, split_train + split_val + split_test)
        self.split_train = split_train / total
        self.split_val = split_val / total
        self.split_test = split_test / total
        self.seed = seed

    # ------------------------------------------------------------------
    def build_yaml(self) -> str:
        """构建 data.yaml 文本内容。"""
        names = {c.id: c.name for c in self.classes}
        data: dict = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
            "nc": len(self.classes),
        }
        if self.split_test > 0:
            data["test"] = "images/test"
        # 保留键顺序
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    def run(self, on_progress=None) -> None:
        """执行导出（同步）。on_progress(done, total)。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        splits = self._split_images()
        total = sum(len(items) for items in splits.values())
        done = 0

        if self.task == "classify":
            self._export_classify(splits, on_progress, done, total)
        else:
            self._export_detect_or_segment(splits, on_progress, done, total)

        # 写 data.yaml + README
        (self.output_dir / "data.yaml").write_text(self.build_yaml(), encoding="utf-8")
        self._write_readme()

    # ------------------------------------------------------------------
    def _split_images(self) -> dict[str, list[ImageItem]]:
        rng = random.Random(self.seed)
        items = [it for it in self.images if not it.error]
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * self.split_train)
        n_val = int(n * self.split_val)
        # 剩余归 test
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:] if self.split_test > 0 else []
        return {"train": train, "val": val, "test": test}

    def _export_detect_or_segment(self, splits, on_progress, done, total) -> None:
        for split, items in splits.items():
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                # 复制图片
                dst_img = img_dir / it.path.name
                shutil.copy2(it.path, dst_img)
                # 写标签
                txt_path = lbl_dir / f"{it.path.stem}.txt"
                txt_path.write_text(self._label_text(it), encoding="utf-8")
                done += 1
                if on_progress:
                    on_progress(done, total)

    def _export_classify(self, splits, on_progress, done, total) -> None:
        class_names = {c.id: c.name for c in self.classes}
        for split, items in splits.items():
            for it in items:
                if it.class_label is None:
                    cname = "unlabeled"
                else:
                    cname = class_names.get(it.class_label.class_id, "unknown")
                dst_dir = self.output_dir / "images" / split / cname
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(it.path, dst_dir / it.path.name)
                done += 1
                if on_progress:
                    on_progress(done, total)

    # ------------------------------------------------------------------
    def _label_text(self, item: ImageItem) -> str:
        w, h = item.width, item.height
        if w <= 0 or h <= 0:
            return ""
        lines: list[str] = []
        if self.task == "detect":
            for b in item.bboxes:
                cx = (b.x + b.w / 2) / w
                cy = (b.y + b.h / 2) / h
                nw = b.w / w
                nh = b.h / h
                lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        elif self.task == "segment":
            for p in item.polygons:
                pts = " ".join(f"{px / w:.6f} {py / h:.6f}" for px, py in p.points)
                lines.append(f"{p.class_id} {pts}")
        return "\n".join(lines)

    def _write_readme(self) -> None:
        model_hint = {
            "detect": "yolo26n.pt",
            "segment": "yolo26n-seg.pt",
            "classify": "yolo26n-cls.pt",
        }[self.task]
        imgsz = 224 if self.task == "classify" else 640
        cmd_task = "classify" if self.task == "classify" else self.task
        data_arg = "." if self.task == "classify" else "data.yaml"
        content = f"""# YOLO 训练数据集

本数据集由 YOLO Dataset Studio 自动生成，可直接用于 Ultralytics YOLO26 训练：

```bash
yolo {cmd_task} train data={data_arg} model={model_hint} epochs=100 imgsz={imgsz}
```

- 任务类型：`{self.task}`
- 类别数：{len(self.classes)}
- 类别：{', '.join(f'{c.id}:{c.name}' for c in self.classes)}

> 训练时可以用 `device=0` 指定 GPU；详细参数参考 https://docs.ultralytics.com/usage/cfg/
"""
        (self.output_dir / "TRAIN.md").write_text(content, encoding="utf-8")