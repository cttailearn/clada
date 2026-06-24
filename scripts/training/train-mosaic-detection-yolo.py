# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import os

from lada import MODEL_WEIGHTS_DIR
from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

# "accurate" model: YOLO26s-seg (Ultralytics >= 8.4.0)
# YOLO26 drops DFL and uses an end-to-end (NMS-free) head which reduces
# inference latency on CPU by up to 43% compared to YOLO11 while reaching
# a higher mAP on small objects. See docs/yolo26_migration.md for details.
model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo26s-seg.pt'))
model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo26s", augmentations=[])

# "fast" model: YOLO26n-seg
# model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo26n-seg.pt'))
# model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo26n", augmentations=[])
