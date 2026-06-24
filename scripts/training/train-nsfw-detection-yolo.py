# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import os

from lada import MODEL_WEIGHTS_DIR
from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

# YOLO26m-seg (Ultralytics >= 8.4.0). Used only to bootstrap the dataset
# pipeline for mosaic-detection / mosaic-restoration training. Switching
# from YOLO11m-seg brings a +2.5 box AP / +3.7 mask AP gain on COCO at
# similar latency. See docs/yolo26_migration.md.
model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo26m-seg.pt'))
model.train(data='configs/yolo/nsfw_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_nsfw_detection_yolo26m")
