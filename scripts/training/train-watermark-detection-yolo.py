# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import os

from lada import MODEL_WEIGHTS_DIR
from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

# YOLO26s (Ultralytics >= 8.4.0). Replaces YOLO11s for watermark / text
# detection. single_cls=True groups the original "logo" and "text" classes
# into a single detection bucket as required by the dataset config and the
# downstream mosaic-removal dataset filter. See docs/yolo26_migration.md.
model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo26s.pt'))
model.train(data='configs/yolo/watermark_detection_dataset_config.yaml', epochs=100, imgsz=512, single_cls=True, name="train_watermark_detection_yolo26s")
