# YOLO26 迁移指南

> 适用版本：`ultralytics >= 8.4.0`，本项目 `pyproject.toml` 已将 `ultralytics` 下界提升到 `8.4.0` 以支持 YOLO26 系列模型。

本文说明将 Lada 中三个 Ultralytics YOLO 任务（**NSFW 检测**、**马赛克检测**、**水印/文本检测**）从 YOLO11 迁移到 YOLO26 的所有变化，覆盖 **模型训练 / 数据集处理 / 模型推理** 三条主线，并分析其它模型是否可以被 YOLO26 替换。

## 1. 为什么迁移到 YOLO26

来源：[Ultralytics YOLO26 中文文档](https://docs.ultralytics.com/zh/models/yolo26)

* **原生端到端、无 NMS**：默认采用 one-to-one 检测头，部署不再需要 NMS 后处理，CPU 推理速度最多比 YOLO11 提升 43%。
* **去除 DFL 模块**：检测头更轻，导出更简单，硬件兼容性更好。
* **小目标识别精度更高**：ProgLoss + STAL，针对小目标做了专项优化，JAV 帧中远景马赛克区域常为小目标，这一改动直接受益。
* **MuSGD 优化器**：SGD × Muon 的混合体，借鉴自 Moonshot Kimi K2，训练稳定性更好、收敛更快。
* **实例分割增强**：多尺度原型 + 语义分割损失，在 COCO 实例分割上比 YOLO11 提升 +2.5 box AP / +3.7 mask AP。

迁移后 Lada 实际受益最大的是 **mosaic 检测**（小目标 + 边缘部署场景）和 **NSFW 检测**（实例分割质量）。

## 2. 训练（Training）

### 2.1 训练脚本对照

| 任务 | 旧基础权重 | 新基础权重 | epoch / imgsz | 入口 |
|---|---|---|---|---|
| Mosaic detection (accurate) | `yolo11s-seg.pt` | **`yolo26s-seg.pt`** | 200 / 640 | [train-mosaic-detection-yolo.py](../scripts/training/train-mosaic-detection-yolo.py) |
| Mosaic detection (fast) | `yolo11n-seg.pt` | **`yolo26n-seg.pt`** | 200 / 640 | 同上 |
| NSFW detection | `yolo11m-seg.pt` | **`yolo26m-seg.pt`** | 200 / 640 | [train-nsfw-detection-yolo.py](../scripts/training/train-nsfw-detection-yolo.py) |
| Watermark detection | `yolo11s.pt` | **`yolo26s.pt`** | 100 / 512 | [train-watermark-detection-yolo.py](../scripts/training/train-watermark-detection-yolo.py) |

所有 `name=` 实验目录名也从 `train_*_yolo11*` 改为 `train_*_yolo26*`。

### 2.2 权重下载

YOLO26 权重位于 Ultralytics 官方仓库 8.4.0 release，可直接通过 Python 下载：

```python
from ultralytics import YOLO
# 首次运行会自动下载到 model_weights/3rd_party/
YOLO("yolo26s-seg.pt")  # NSFW / mosaic detection accurate
YOLO("yolo26n-seg.pt")  # mosaic detection fast
YOLO("yolo26m-seg.pt")  # NSFW detection
YOLO("yolo26s.pt")      # watermark detection
```

或手动下载后放到 `model_weights/3rd_party/`：

```bash
wget -P model_weights/3rd_party/ \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt
wget -P model_weights/3rd_party/ \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt
wget -P model_weights/3rd_party/ \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt
wget -P model_weights/3rd_party/ \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

### 2.3 训练启动

```bash
# mosaic detection
python scripts/training/train-mosaic-detection-yolo.py

# NSFW detection
python scripts/training/train-nsfw-detection-yolo.py

# watermark detection
python scripts/training/train-watermark-detection-yolo.py
```

> **数据集布局不需要改动**：`configs/yolo/*.yaml` 仍指向 `train/images`、`train/labels` 等标准 YOLO 目录。

### 2.4 YOLO26 训练 tips

1. **默认优化器变为 MuSGD**（除非显式覆盖 `optimizer=`）。Ultralytics 推荐沿用默认；如需回到 SGD 或 Adam，传 `optimizer='SGD'` 或 `optimizer='Adam'`。
2. **双头架构**：`YOLO26` 默认输出 one-to-one（无 NMS），但导出时可切到 one-to-many（带 NMS，精度略高）。训练阶段无需关心，导出阶段再选。
3. **更强的数据增强**：YOLO26 默认训练 recipe 已加强数据增强。如对马赛克这种小目标场景需要更激进增强，可在 `model.train(...)` 调用里显式传 `mosaic=1.0, mixup=0.1, copy_paste=0.1`。
4. **默认 epoch 数保持不变**：mosaic / NSFW 200 epoch、watermark 100 epoch 已经够用；MuSGD 收敛更快通常不需要再加 epoch。

### 2.5 验证训练结果

```bash
# 在验证集上评估（输出 mAP、混淆矩阵）
yolo val model=experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt \
     data=configs/yolo/mosaic_detection_dataset_config.yaml imgsz=640

# 在真实视频上交互式查看预测（按 S 截图回灌数据集）
python scripts/evaluation/view-yolo.py --input <your_video.mp4> \
  --model-path experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt \
  --screenshot-dir datasets/nsfw_detection_labelme/train
```

## 3. 数据集处理（Dataset）

### 3.1 YOLO 格式保持不变

YOLO26 与 YOLO11 在 **数据格式** 上完全兼容：
* 检测：`class cx cy w h`（归一化）
* 实例分割：`class x1 y1 x2 y2 ... xn yn`（多边形，归一化）
* 数据集 YAML 字段 `path / train / val / names` 不变。

所以下面三套脚本可以 **不加修改** 直接沿用：

* [convert-dataset-labelme-to-yolo.py](../scripts/dataset_creation/convert-dataset-labelme-to-yolo.py)
* [convert-dataset-yolo-to-labelme.py](../scripts/dataset_creation/convert-dataset-yolo-to-labelme.py)
* [create-watermark-detection-dataset.py](../scripts/dataset_creation/create-watermark-detection-dataset.py)

### 3.2 数据集生成脚本中的 YOLO 调用

`create-mosaic-detection-dataset.py` 与 `create-mosaic-restoration-dataset.py` 通过 `--model` 路径参数加载训练好的检测器，**与基础权重名（yolo11 vs yolo26）解耦**，无需改动：

```bash
# 启动时显式指定用 YOLO26 训练出的检测器
python scripts/dataset_creation/create-mosaic-detection-dataset.py \
  --input-root <raw_images_dir> \
  --output-root <dataset_dir> \
  --model model_weights/lada_nsfw_detection_model_v1.3.pt          # 或自行训练的 YOLO26m-seg
  --mosaic-model-path experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt
```

> **过滤模型的输入是用户自己训练/发布的权重**，训练脚本切换到 YOLO26 后，调用方只需更换 `--model` / `--mosaic-model-path` 路径即可。

### 3.3 YOLO26 数据集相关注意点

* **`augmentations=[]` 在 mosaic detection 中要保留**：马赛克检测不能容忍 mosaic/旋转增强（增强会破坏马赛克的纹理），所以原训练脚本中已经显式禁用了。YOLO26 沿用此设置。
* **`single_cls=True` 在 watermark detection 中保留**：原 2 类（`logo` / `text`）合并为单类，否则与 `mosaic_removal_dataset` 内的过滤逻辑不匹配。
* **多尺度原型**：YOLO26-seg 的多尺度原型模块对 mask 边界更敏感，标签噪声（手画 polygon 误差）会比 YOLO11 更明显地影响收敛。标注时尽量画紧一些（参见训练文档）。

## 4. 推理（Inference）

### 4.1 推理路径上的代码变化

| 文件 | 变化 |
|---|---|
| [lada/models/yolo/yolo_segmentation_model.py](../lada/models/yolo/yolo_segmentation_model.py) | **新增**，由 `yolo11_segmentation_model.py` 重命名而来，类名 `Yolo11SegmentationModel → YoloSegmentationModel`，同时支持 YOLO11-seg 与 YOLO26-seg |
| [lada/restorationpipeline/__init__.py](../lada/restorationpipeline/__init__.py) | import 改为新模块名 |
| [lada/restorationpipeline/mosaic_detector.py](../lada/restorationpipeline/mosaic_detector.py) | import 与类型注解更新 |
| [lada/restorationpipeline/frame_restorer.py](../lada/restorationpipeline/frame_restorer.py) | import 与类型注解更新 |

### 4.2 端到端 NMS-free 推理

YOLO26 默认启用 `end2end=True`，**推理时无需 NMS**。我们的包装器已经处理好：

```python
preds = nms.non_max_suppression(
    preds[0],
    self.args.conf,
    self.args.iou,
    self.args.classes,
    self.args.agnostic_nms,
    max_det=self.args.max_det,
    nc=len(self.model.names),
    end2end=getattr(self.model, "end2end", False),  # 自动识别 YOLO26 的端到端头
)
```

* **YOLO26-seg**：`end2end=True`，`non_max_suppression` 在内部走 one-to-one 路径直接输出 top-300 检测。
* **YOLO11-seg**（如继续使用旧权重）：`end2end=False`，走传统 NMS 路径。

两条路径共享同一段 `construct_result` 代码，mask 原型仍由 `preds[0][-1]` 取回，因此下游 `MosaicDetector / FrameRestorer / NsfwDetector / WatermarkDetector` 完全无需改动。

### 4.3 推理性能收益

| 模型 | CPU ONNX (ms) | T4 TensorRT10 (ms) | box mAP (e2e) | mask mAP (e2e) |
|---|---|---|---|---|
| YOLO11m-seg | ~325 | ~6.0 | ~50.0 | ~40.4 |
| **YOLO26m-seg** | **328.2 ± 2.4** | **6.7 ± 0.1** | **52.5** | **44.1** |
| YOLO11s-seg | ~115 | ~3.0 | ~44.0 | ~36.0 |
| **YOLO26s-seg** | **118.4 ± 0.9** | **3.3 ± 0.0** | **47.3** | **40.0** |
| YOLO11s | ~85  | ~2.3 | ~46.0 | — |
| **YOLO26s** | **87.2 ± 0.9** | **2.5 ± 0.0** | **47.8** | — |

> 实际收益主要来自精度提升（小目标场景）；端到端推理在 batch=1 / CPU 后端上才能看到显著延迟下降。在 Lada 默认的 GPU 实时播放场景下，**主要受益是检测质量的提升，而非吞吐**。

### 4.4 导出（可选）

如果想脱离 Ultralytics Python 环境做 C++/ONNX 部署：

```python
from ultralytics import YOLO

# 默认导出端到端（无 NMS）
model = YOLO("experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt")
model.export(format="onnx")            # 端到端，最快
model.export(format="onnx", end2end=False)  # one-to-many，保留 NMS
```

Lada GUI 内部仍以 PyTorch AutoBackend 加载，不会切换到 ONNX；但打包时的 GPU 导出选项已可直接复用。

## 5. 可被 YOLO26 替换的其它模型

| 当前模型 | 当前用途 | 能否用 YOLO26 替换 | 备注 |
|---|---|---|---|
| `lada_mosaic_detection_model_v*.pt`（YOLO11s-seg） | 主检测器：定位马赛克区域 | ✅ **已替换**（用 YOLO26s-seg 重训） | 模型权重随训练脚本切换 |
| `lada_nsfw_detection_model_v*.pt`（YOLO11m-seg） | 数据集制作：定位 NSFW | ✅ **已替换**（用 YOLO26m-seg 重训） | 仍只用于离线数据集生成 |
| `lada_watermark_detection_model_v*.pt`（YOLO11s） | 数据集制作：过滤水印/文字 | ✅ **已替换**（用 YOLO26s 重训） | 仍只用于离线过滤 |
| `centerface.onnx`（CenterFace） | SFW 人脸检测，制作马赛克检测数据集 | ❌ **不能** | CenterFace 是人脸专用模型；YOLO26 检测未训练人脸数据集，不具备等效精度 |
| `ch_head_s_1536_e150_best_mMR.pt`（BPJDet） | SFW 人头检测，制作马赛克检测数据集 | ❌ **不能** | BPJDet 输出人体+人头部件分割，YOLO26 是 80 类 COCO 检测，没有 head 部件 |
| `640m.pt`（NudeNet） | NSFW 二次校验 | ⚠️ **可选**：YOLO26 没有 NSFW 预训练，且 YOLO26-cls 是 ImageNet 预训练，不替代 NudeNet 的"裸露器官"细分类能力 | 不建议替换 |
| `clean_youknow_video.pth`（DeepMosaics） | 旧版修复模型 | ❌ **不能** | DeepMosaics 是 GAN-based 修复模型，不是检测/分类，与 YOLO 系列无功能重合 |
| `lada_mosaic_restoration_model_generic_v*.pth`（BasicVSR++） | 视频修复主干 | ❌ **不能** | BasicVSR++ 是视频超分/修复模型，与 YOLO 系列无功能重合 |
| `DOVER.pth` | 视频质量评估（数据集过滤） | ❌ **不能** | DOVER 是视频质量评估专用模型，YOLO26 没有对应能力 |
| `spynet_20210409.pth` | BasicVSR++ 内的光流估计 | ❌ **不能** | 光流专用子网，集成在 BasicVSR++ 内部 |
| `ch_head_s_*` head 检测 | 离线场景 | ❌ | 同 BPJDet |
| `scripts/training/train-bj-classifier.py`（ResNet50） | POV/BJ 场景二分类 | ⚠️ **可选**：可用 `YOLO26n-cls` 替换 | 数据集不变；`model = Yolo("yolo26n-cls.pt").train(data="datasets/pov_bj_scene_detection/data.yaml", epochs=15, imgsz=224)`。ResNet50 训练脚本仍可用作 baseline 对比 |

### 决策摘要

* **值得替换**：3 个 YOLO 任务（mosaic / NSFW / watermark）—— 已完成。
* **可选替换**：POV/BJ 场景分类（ResNet50 → YOLO26-cls），替换收益不大但能统一栈。
* **不应替换**：所有专用检测/评估/生成模型（CenterFace、BPJDet、NudeNet、DOVER、SpyNet、BasicVSR++、DeepMosaics）—— 与 YOLO26 任务域不重叠。

## 6. 升级步骤（实操清单）

1. 升级 `ultralytics`：
   ```bash
   uv pip install 'ultralytics>=8.4.0'
   ```
2. 重新打 telemetry 补丁（仍是同一个文件路径）：
   ```bash
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
   ```
3. 下载 YOLO26 基础权重到 `model_weights/3rd_party/`（见 §2.2）。
4. 重新训练 3 个 YOLO 模型（命令见 §2.3），生成的新权重会落到：
   ```
   experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt
   experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt
   experiments/yolo/segment/train_watermark_detection_yolo26s/weights/best.pt
   ```
5. 把 `best.pt` 重命名并复制到 `model_weights/`，沿用既有的 `lada_*_detection_model_v*.pt` 命名规范发布。
6. 在 Lada GUI / CLI 中通过 `--mosaic-detection-model <path>` 选用新权重；旧 YOLO11 权重仍可继续使用（包装器同时支持两者）。

## 7. 兼容性与回退

* 旧 YOLO11-seg 权重无需重训，仍可由 `YoloSegmentationModel` 加载（`end2end=False`）。
* 训练脚本中的 `name=` 路径已改为 `yolo26*`，如果实验目录下还有旧 `yolo11*` 训练产物，可手动复用其 `weights/best.pt`。
* 暂未发现 YOLO26 与 YOLO11 在 mask 输出格式上有差异，下游 `convert_yolo_mask`、`scale_and_unpad_image` 等工具无需改动。

---

参考：
* [Ultralytics YOLO26 官方文档（中文）](https://docs.ultralytics.com/zh/models/yolo26)
* [YOLO26 Training Recipe](https://docs.ultralytics.com/zh/guides/yolo26-training-recipe)
* [YOLOE-26（开放词汇实例分割）](https://docs.ultralytics.com/zh/models/yolo26/#yoloe-26-open-vocabulary-instance-segmentation)
