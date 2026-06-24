<h1 align="center">
  <img src="assets/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada
</h1>

*Lada* 是一款用于修复像素化成人视频（JAV）的工具，帮助恢复马赛克/像素化区域的视觉质量。

## 功能

- **修复像素化视频**：恢复成人视频中的马赛克或像素化场景。
- **观看/导出视频**：通过 CLI 或 GUI 实时观看或导出修复后的视频。

## 使用

### GUI

打开文件后，可以实时观看修复后的视频，或者导出为新文件稍后观看：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshot_gui_1_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/screenshot_gui_1_light.png">
  <img alt="视频预览截图" src="assets/screenshot_gui_1_dark.png" width="36%">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshot_gui_2_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/screenshot_gui_2_light.png">
  <img alt="视频导出截图" src="assets/screenshot_gui_2_dark.png" width="45%">
</picture>

其他设置可在左侧边栏找到。

### CLI

也可以使用命令行接口（CLI）来修复视频：

```shell
lada-cli --input <输入视频路径>
```

<img src="assets/screenshot_cli_1.png" alt="CLI 截图" width="60%">

更多选项请使用 `--help` 查看。

## 性能要求与硬件需求

修复质量因场景而异。有些场景可能看起来非常逼真，而另一些可能会出现明显的伪影，有时甚至比原始马赛克更糟糕。

运行此应用需要 GPU 和一定耐心。至少 4-6 GB 显存的显卡适合大多数场景。

应用也需要相当的内存进行缓冲以提升性能。对于 1080p 内容，6-8 GB 内存通常足够，但 4K 视频需要显著更多。

要实时观看修复视频，需要性能强劲的设备。否则播放器可能会暂停缓冲。观看时不进行编码，但会使用额外内存进行缓冲。

如果 GPU 不足以支持实时播放，可以导出视频稍后播放（GUI 和 CLI 都支持）。

虽然应用可以在 CPU 上运行，但性能极慢，对大多数用户不实用。

## 安装

### Flatpak（Linux）

在 Linux 上通过 Flathub 安装（CLI + GUI）最为便捷：

<a href='https://flathub.org/apps/details/io.github.ladaapp.lada'><img width='200' alt='Download from Flathub' src='https://flathub.org/api/badge?svg&locale=en'/></a>

> **注意**：Flatpak 仅支持 x86_64 CPU。支持 Nvidia/CUDA（Turing 及以上：RTX 20xx 到 RTX 50xx）和 Intel Arc GPU。请确保 GPU 驱动为最新。无 GPU 也可使用但会非常慢。请务必从 Flathub 安装 Intel 或 Nvidia 插件。

### Docker

也提供了 Docker 镜像（仅 CLI）：

```shell
docker pull ladaapp/lada:latest
```

> 仅支持 x86_64 + Nvidia/CUDA GPU。需安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

```shell
docker run --rm --gpus all --mount type=bind,src=<输入视频路径>,dst=/mnt ladaapp/lada:latest --input "/mnt/<输入视频文件名>"
```

### Windows

Windows 用户请从 [Releases 页面](https://codeberg.org/ladaapp/lada/releases) 下载 `.7z` 独立包。

> 仅支持 x86_64 + Nvidia/CUDA（Turing 及以上）和 Intel Arc GPU。首次启动可能因 Defender 扫描而较慢。

### 从源码构建

见 [Linux 安装指南](docs/linux_install.md)、[macOS 安装指南](docs/macOS_install.md)、[Windows 安装指南](docs/windows_install.md)。

---

## 模型训练完整流程

Lada 依赖三个 YOLO26 检测模型和一个视频修复模型。以下是从零开始训练所有模型的完整流程。如果只想使用预训练权重，可跳过训练部分，直接使用 Lada 发布的模型权重。

### 架构概览

```
                    输入视频
                       │
          ┌────────────┤
          ▼            ▼
   马赛克检测模型 ──► NSFW检测模型 ──► 水印检测模型
   (YOLO26s-seg)   (YOLO26m-seg)     (YOLO26s)
          │              │                 │
          ▼              ▼                 ▼
   定位马赛克区域    定位NSFW区域      过滤水印/文字帧
          │              │
          └──────┬───────┘
                 ▼
         马赛克修复模型
      (RVRT / BasicVSR++)
                 │
                 ▼
           修复后的视频
```

四个模型分工如下：

| 模型 | 架构 | 用途 | 运行阶段 |
|------|------|------|---------|
| NSFW 检测 | YOLO26m-seg | 检测视频帧中的 NSFW 区域（实例分割） | 仅离线数据集制作 |
| 水印检测 | YOLO26s | 过滤被水印/文字/Logo 遮挡的帧 | 仅离线数据集制作 |
| 马赛克检测 | YOLO26s-seg | 实时检测每一帧中的马赛克区域 | 在线推理（GUI/CLI） |
| 马赛克修复 | RVRT + Projected GAN | 对检测到的马赛克区域进行像素级修复 | 在线推理（GUI/CLI） |

### 训练模型一览

| 模型 | 基础权重 | 任务 | Epoch | imgsz | 训练脚本 |
|------|---------|------|-------|-------|---------|
| 马赛克检测 (accurate) | `yolo26s-seg.pt` | 实例分割 | 200 | 640 | `train-mosaic-detection-yolo.py` |
| 马赛克检测 (fast) | `yolo26n-seg.pt` | 实例分割 | 200 | 640 | 同上 |
| NSFW 检测 | `yolo26m-seg.pt` | 实例分割 | 200 | 640 | `train-nsfw-detection-yolo.py` |
| 水印检测 | `yolo26s.pt` | 目标检测 | 100 | 512 | `train-watermark-detection-yolo.py` |
| 马赛克修复 (RVRT) | 从头训练 | 视频修复 | 两阶段 | 256 | `train-mosaic-restoration-basicvsrpp.py` |

### 步骤 1：环境准备

```bash
# 1. 按对应平台安装系统依赖（见上方安装指南）

# 2. 安装 Python 依赖
uv sync --group dev --inexact

# 3. 应用补丁
# 修复 resume training（MMEngine）
patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/adjust_mmengine_resume_dataloader.patch

# 移除 ultralytics telemetry（训练数据不上传）
patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
```

### 步骤 2：下载基础模型权重

YOLO26 权重（Ultralytics >= 8.4.0）可从 [Ultralytics 官方 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 下载：

```bash
# YOLO26 基础权重 — 首次运行 train 脚本时也会自动下载
wget -P model_weights/3rd_party/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt
wget -P model_weights/3rd_party/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt
wget -P model_weights/3rd_party/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt
wget -P model_weights/3rd_party/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

数据集制作必需的辅助模型权重：

```bash
# RVRT generator 的 VGG19 感知损失
wget -P model_weights/3rd_party/ https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

# BasicVSR++ SPyNet 光流（legacy generator）
wget -P model_weights/3rd_party/ https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth

# 视频质量评估（数据集过滤）
wget -P model_weights/3rd_party/ https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth

# NSFW 检测预训练（用于数据集制作脚本）
wget -P model_weights/ https://huggingface.co/ladaapp/lada/resolve/main/lada_nsfw_detection_model_v1.3.pt?download=true

# 水印检测预训练（用于数据集制作脚本）
wget -P model_weights/ https://huggingface.co/ladaapp/lada/resolve/main/lada_watermark_detection_model_v1.3.pt?download=true

# NudeNet（NSFW 二次校验，可选）
wget -P model_weights/3rd_party/ https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt

# CenterFace（人脸检测，SFW mosaic 数据集制作）
wget -P model_weights/3rd_party/ https://github.com/ORB-HD/deface/raw/refs/tags/v1.5.0/deface/centerface.onnx

# BPJDet（人头检测，SFW mosaic 数据集制作）
wget -P model_weights/3rd_party/ https://huggingface.co/HoyerChou/BPJDet/resolve/main/ch_head_s_1536_e150_best_mMR.pt?download=true
```

> NudeNet 项目在 GitHub 上设为限制级，可能需要登录后从 [release 页面](https://github.com/notAI-tech/NudeNet/releases/) 手动下载 `640m.pt`。该模型仅用于数据集制作时为 NSFW 检测结果做二次校验，不影响训练流程。

### 步骤 3：数据集制作

#### 3.1 收集原始素材

```bash
# 使用 yt-dlp 和 gallery-dl 下载 NSFW 视频和图片
# 使用 extract-video-frames.py 从视频抽帧
python scripts/dataset_creation/extract-video-frames.py --input <视频目录> --output <帧输出目录>

# 去重工具推荐：Czkawka、cbird
```

在开始制作数据集之前，请先安装 `labelme` 标注工具到一个独立 venv（避免依赖冲突）：

```bash
python -m venv .venv_labelme
source .venv_labelme/bin/activate      # Linux/macOS
# 或 .venv_labelme\Scripts\Activate.ps1  # Windows
pip install labelme
```

#### 3.2 NSFW 检测数据集（实例分割）

这是整个训练流程的起点，所有后续数据集都依赖 NSFW 检测模型。

**手动标注**（耗时最长的一步）：

```bash
mkdir -p datasets/nsfw_detection_labelme/{train,val}

# 启动 labelme
labelme --flags sfw --labels nsfw --nodata --autosave datasets/nsfw_detection_labelme/train
```

- 使用「Draw Polygon」工具精确描画 NSFW 区域（尽量紧贴物体）。
- 无 NSFW 内容的图像标记 `SFW` flag，不画多边形。
- 标注完成后对 `val` 目录重复同样流程。

**转换为 YOLO 格式**：

切回 Lada 的 venv（`.venv`），执行转换：

```bash
mkdir -p datasets/nsfw_detection/{train,val}/{images,labels}

python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py \
  --dir-in datasets/nsfw_detection_labelme/train \
  --dir-out-images datasets/nsfw_detection/train/images \
  --dir-out-labels datasets/nsfw_detection/train/labels

python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py \
  --dir-in datasets/nsfw_detection_labelme/val \
  --dir-out-images datasets/nsfw_detection/val/images \
  --dir-out-labels datasets/nsfw_detection/val/labels
```

**迭代式改进**：训练首轮后用 `view-yolo.py` 在真实视频上验证，对效果不佳的帧截图回灌到 labelme 重新标注 → 转换 → 再训练。

```bash
python scripts/evaluation/view-yolo.py --input <测试视频> \
  --model-path experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt \
  --screenshot-dir datasets/nsfw_detection_labelme/train
```

#### 3.3 马赛克检测数据集

自动从 NSFW 视频生成。使用已训练的 NSFW 检测模型定位人体区域，用马赛克/像素化图案替换该区域，生成「原图 + 马赛克掩码」对。

```bash
python scripts/dataset_creation/create-mosaic-detection-dataset.py \
  --input-root <NSFW视频/图像目录> \
  --output-root datasets/mosaic_detection \
  --model model_weights/lada_nsfw_detection_model_v1.3.pt
```

同时还需 SFW 马赛克样本（如人脸马赛克），使模型学会区分「需要修复的 NSFW 马赛克」和「不应修复的人脸马赛克」。目前使用 CenterFace + BPJDet 自动生成，但可能需要手动清理。

#### 3.4 马赛克修复数据集

从 NSFW 视频自动裁剪短片用于训练修复模型：

```bash
python scripts/dataset_creation/create-mosaic-restoration-dataset.py \
  --input <NSFW视频目录> \
  --output-root datasets/mosaic_restoration
```

**工作原理**：
1. NSFW 检测模型逐帧定位人体区域。
2. 裁剪出以该区域为中心的短视频片段 + 掩码。
3. 水印检测模型 + DOVER 自动过滤低质量/有水印的片段。
4. 训练时 `MosaicVideoDataset` 会在裁剪的片段上动态叠加马赛克纹理（降低磁盘占用）。

> 先用少量数据试跑，确认参数合理再全量生成。检查生成的 `metadata.json` 可了解水印/质量等过滤信息。
>
> 手动清理提示：打开场景片段目录的缩略图视图 → 删除含水印或不符预期的片段 → 写脚本批量删除对应的 mask 和 metadata JSON 文件。

#### 3.5 水印检测数据集

自动合成——在干净图片上随机叠加 Logo（PNG 透明背景）和系统字体文字：

```bash
python scripts/dataset_creation/create-watermark-detection-dataset.py \
  --train-images-dir <训练图片目录> \
  --val-images-dir <验证图片目录> \
  --logos-dir <Logo PNG 目录> \
  --yolo-dir datasets/watermark_detection
```

- 图片建议以 NSFW 为主，混入少量 COCO 通用图片（无默认水印）。
- Logo 建议收集带 alpha 通道的 PNG，便于精确裁剪边界框。
- 脚本自动随机选取系统已安装的所有 TrueType 字体。建议多安装一些花哨字体以模拟真实水印样式。

#### 数据集格式总结

```
datasets/
├── nsfw_detection/          YOLO 实例分割格式
│   ├── train/{images,labels}/
│   └── val/{images,labels}/
├── mosaic_detection/        YOLO 实例分割格式
│   ├── train/{images,labels}/
│   └── val/{images,labels}/
├── mosaic_restoration/      自定义格式（视频片段 + 掩码）
│   ├── scenes/              裁剪后的 NSFW 片段（mp4）
│   └── masks/               对应的二值掩码（mp4）
├── watermark_detection/     YOLO 目标检测格式
│   ├── train/{images,labels}/
│   └── val/{images,labels}/
└── nsfw_detection_labelme/  labelme 原始标注（非 YOLO）
    ├── train/
    └── val/
```

数据集配置文件在 `configs/yolo/` 下：

| 配置文件 | 用途 |
|---------|------|
| [nsfw_detection_dataset_config.yaml](configs/yolo/nsfw_detection_dataset_config.yaml) | NSFW 检测训练配置（单类 `nsfw`） |
| [mosaic_detection_dataset_config.yaml](configs/yolo/mosaic_detection_dataset_config.yaml) | 马赛克检测训练配置（双类 `mosaic_nsfw` / `mosaic_sfw_head`） |
| [watermark_detection_dataset_config.yaml](configs/yolo/watermark_detection_dataset_config.yaml) | 水印检测训练配置（双类 `logo` / `text`，训练时合并为单类） |

### 步骤 4：YOLO26 模型训练

#### 4.1 训练 NSFW 检测模型

```bash
python scripts/training/train-nsfw-detection-yolo.py
```

内部参数：`YOLO("yolo26m-seg.pt").train(data='...nsfw_detection_dataset_config.yaml', epochs=200, imgsz=640)`

输出路径：`experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt`

#### 4.2 训练马赛克检测模型

```bash
python scripts/training/train-mosaic-detection-yolo.py
```

内部参数：`YOLO("yolo26s-seg.pt" 或 "yolo26n-seg.pt").train(data='...mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, augmentations=[])`

> **注意**：马赛克检测任务显式禁用了 mosaic/mixup/copy_paste 数据增强（`augmentations=[]`），因为这些增强会破坏马赛克的纹理特征。

输出路径：
- accurate: `experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt`
- fast: `experiments/yolo/segment/train_mosaic_detection_yolo26n/weights/best.pt`

#### 4.3 训练水印检测模型

```bash
python scripts/training/train-watermark-detection-yolo.py
```

内部参数：`YOLO("yolo26s.pt").train(data='...watermark_detection_dataset_config.yaml', epochs=100, imgsz=512, single_cls=True)`

> **注意**：`single_cls=True` 将 `logo` 与 `text` 两类合并为单检测桶，与下游马赛克修复数据集的过滤逻辑匹配。

输出路径：`experiments/yolo/segment/train_watermark_detection_yolo26s/weights/best.pt`

#### YOLO26 训练注意事项

- **MuSGD 优化器**：YOLO26 默认使用 MuSGD（借鉴 Moonshot Kimi K2），收敛更快，通常不需要增加 epoch。
- **端到端推理**：YOLO26 默认使用 one-to-one 检测头，推理时无需 NMS，在 CPU / 端侧可获 43% 加速。
- **多尺度原型**：YOLO26-seg 的多尺度原型模块对标签噪声（手画 polygon 误差）更敏感，标注时尽量精确。

### 步骤 5：马赛克修复模型训练

#### 架构选择

| 架构 | 类型 | 核心创新 | 状态 |
|------|------|---------|------|
| **RVRT** (NeurIPS 2022) | Hybrid Recurrent + Transformer | Guided Deformable Attention (GDA)，无需光流 | **推荐** |
| **BasicVSR++** (CVPR 2022) | CNN Recurrent | SPyNet 光流 + 可变形对齐 + 4 分支双向传播 | Legacy（仍支持） |

两种架构的 GAN 阶段均使用 **Projected GAN** 判别器（冻结 EfficientNet-B0 backbone + 可训练 CNN head + Hinge loss + R1 梯度惩罚）。

#### RVRT 架构（推荐）

```
Input: (N, T=16, 3, H=256, W=256) in [0,1]
  │
  ├── feat_extract: Conv2d(3→64) → 全分辨率处理
  │
  ├── Frame Grouping: 重叠分组 (group=3, overlap=1)
  │
  ├── RVRT Blocks (×8, shared, recurrent):
  │     LayerNorm → GDA → LayerNorm → FFN(GELU)
  │
  └── Reconstruction → + 全局残差 (lqs) → Output
```

**与 BasicVSR++ 的主要区别**：
- 不需要预计算光流（SPyNet）
- 全分辨率处理（无 4× 下采样）
- 通过 `hidden.detach()` 实现截断 BPTT，跨组常量内存
- ~1.1M 可训练参数（vs BasicVSR++ 的 ~3.1M）

配置文件在 `configs/rvrt/` 下：

| 配置文件 | 用途 |
|---------|------|
| [mosaic_restoration_stage1.py](configs/rvrt/mosaic_restoration_stage1.py) | Stage 1：纯像素损失预训练 |
| [mosaic_restoration_stage2_projected.py](configs/rvrt/mosaic_restoration_stage2_projected.py) | Stage 2：Projected GAN 微调 |

#### 两阶段训练流程

**Stage 1 — 像素损失预训练**：
```bash
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/rvrt/mosaic_restoration_stage1.py
```

> 中断后恢复：添加 `--resume`。

**权重转换（Stage 1 → Stage 2）**：
```bash
python scripts/training/convert-weights-basicvsrpp-stage1-to-stage2.py
```

**Stage 2 — Projected GAN 微调**：
```bash
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/rvrt/mosaic_restoration_stage2_projected.py \
  --load-from experiments/rvrt/mosaic_restoration_rvrt_stage1/iter_10000_converted.pth
```

**导出推理权重**（去掉判别器，减小模型体积）：
```bash
python scripts/training/export-weights-basicvsrpp-stage2-for-inference.py
```

#### BasicVSR++ 训练（Legacy）

配置文件在 `configs/basicvsrpp/` 下：

```bash
# Stage 1
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/basicvsrpp/mosaic_restoration_generic_stage1.py

# 权重转换（同上脚本）

# Stage 2 — Projected GAN
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/basicvsrpp/mosaic_restoration_generic_stage2_projected.py \
  --load-from experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_10000_converted.pth
```

> 训练框架基于 [MMagic / MMEngine](https://mmengine.readthedocs.io/)。调参建议先阅读 MMEngine 文档。微调用的完整权重可在 [HuggingFace](https://huggingface.co/ladaapp/lada) 找到。

### 步骤 6：模型验证与评估

#### YOLO 模型验证

```bash
# COCO 验证集评估（mAP、混淆矩阵等）
yolo val model=experiments/yolo/segment/train_mosaic_detection_yolo26s/weights/best.pt \
  data=configs/yolo/mosaic_detection_dataset_config.yaml imgsz=640

yolo val model=experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt \
  data=configs/yolo/nsfw_detection_dataset_config.yaml imgsz=640

yolo val model=experiments/yolo/detect/train_watermark_detection_yolo26s/weights/best.pt \
  data=configs/yolo/watermark_detection_dataset_config.yaml imgsz=512
```

#### 可视化验证

在真实视频上交互式查看预测结果，按 `S` 截图回灌标注：

```bash
python scripts/evaluation/view-yolo.py --input <测试视频路径> \
  --model-path experiments/yolo/segment/train_nsfw_detection_yolo26m/weights/best.pt \
  --screenshot-dir datasets/nsfw_detection_labelme/train
```

#### 修复模型验证

```bash
python scripts/evaluation/validate-basicvsrpp.py
```

### 步骤 7：部署到 Lada

训练完成后，将最优权重复制到 `model_weights/` 目录，按 Lada 的命名规范发布：

```
model_weights/
├── lada_mosaic_detection_model_v5.pt          # YOLO26s-seg → 新版本号
├── lada_nsfw_detection_model_v2.0.pt          # YOLO26m-seg → 新版本号
├── lada_watermark_detection_model_v2.0.pt     # YOLO26s → 新版本号
└── lada_mosaic_restoration_model_generic_v2.0.pth  # RVRT → 新版本号
```

> Lada GUI / CLI 通过 `--mosaic-detection-model <path>` 等参数选用自定义权重。旧 YOLO11 权重仍可继续使用（包装器 `YoloSegmentationModel` 同时兼容 YOLO11 和 YOLO26）。

### 完整训练流程速查

```
┌─ 准备 ───────────────────────────────────────────────────────┐
│ 1. 安装依赖 + 应用补丁                                        │
│ 2. 下载 YOLO26 基础权重 + 辅助模型                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 数据集制作 ─────────────────────────────────────────────────┐
│ 3a. labelme 手动标注 NSFW 区域 → convert to YOLO              │
│ 3b. 训练 NSFW 模型 → view-yolo 验证 → 迭代标注                │
│ 3c. create-mosaic-detection-dataset.py (需要 NSFW 模型)      │
│ 3d. create-mosaic-restoration-dataset.py (需要 NSFW + 水印)  │
│ 3e. create-watermark-detection-dataset.py                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ YOLO26 训练 ────────────────────────────────────────────────┐
│ 4a. train-nsfw-detection-yolo.py                             │
│ 4b. train-mosaic-detection-yolo.py                           │
│ 4c. train-watermark-detection-yolo.py                        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 修复模型训练 ───────────────────────────────────────────────┐
│ 5a. Stage1: pixel loss pretraining                           │
│ 5b. 权重转换 (convert-weights-...stage1-to-stage2.py)        │
│ 5c. Stage2: GAN fine-tuning                                  │
│ 5d. 导出推理权重 (export-weights-...for-inference.py)        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 验证与部署 ─────────────────────────────────────────────────┐
│ 6. yolo val / validate-basicvsrpp / view-yolo                │
│ 7. 复制 best.pt → model_weights/, 更新版本号                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 训练与数据集制作参考文档

详细步骤见 [训练与数据集制作](docs/training_and_dataset_creation.md)（英文），或 [YOLO26 迁移指南](docs/yolo26_migration.md)。

## YOLO Dataset Studio（辅助标注工具）

本仓库同时提供一个基于 PyQt6 的 YOLO26 数据集准备 GUI 工具，位于 [yolo_dataset_studio/](yolo_dataset_studio/)。可在模型预标注后进行人工校对并导出 YOLO 格式训练集。详见其 [README](yolo_dataset_studio/README.md)。

## 贡献

Lada 项目主站位于 [Codeberg](https://codeberg.org/ladaapp/lada)，GitHub 为镜像。

贡献代码、想法或 Bug 报告请使用 Codeberg 的 [Pull requests](https://codeberg.org/ladaapp/lada/pulls) 和 [Issue tracker](https://codeberg.org/ladaapp/lada/issues)。

翻译贡献请前往 [Codeberg Translate](https://translate.codeberg.org/projects/lada/lada/)。

[![翻译状态](https://translate.codeberg.org/widget/lada/lada/multi-auto.svg)](https://translate.codeberg.org/engage/lada/)

## 发布

新版本会同时发布在 [GitHub Releases](https://github.com/ladaapp/lada/releases) 和 [Codeberg Releases](https://codeberg.org/ladaapp/lada/releases)。

## 许可证

源码与模型均采用 AGPL-3.0 许可证。详见 [LICENSE.md](LICENSE.md)。

## 致谢

本项目基于以下优秀个人与项目的工作：

* [DeepMosaics](https://github.com/HypoX64/DeepMosaics)：马赛克数据集创建代码，也启发了此项目。
* [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++) / [MMagic](https://github.com/open-mmlab/mmagic)：马赛克修复的基础模型架构。
* [RVRT](https://github.com/JingyunLiang/RVRT) (NeurIPS 2022)：Guided Deformable Attention 生成器，提升时序对齐与生成质量。
* [Projected GAN](https://github.com/autonomousvision/projected-gan) (Sauer et al., NeurIPS 2021)：预训练特征网络判别器 + Hinge loss + R1 正则化。
* [YOLO/Ultralytics](https://github.com/ultralytics/ultralytics)：马赛克与 NSFW 检测模型训练。
* [DOVER](https://github.com/VQAssessment/DOVER)：数据集制作中的视频质量评估。
* [DNN Watermark / PITA Dataset](https://github.com/tgenlis83/dnn-watermark)：水印检测数据集生成。
* [NudeNet](https://github.com/notAI-tech/NudeNet/)：NSFW 二次分类过滤。
* [Twitter Emoji](https://github.com/twitter/twemoji)：应用图标的茄子 emoji 基础。
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)：马赛克检测模型的降质流水线设计。
* [BPJDet](https://github.com/hnuzhy/BPJDet)：人体/人头检测（SFW 马赛克数据集）。
* [CenterFace](https://github.com/Star-Clouds/CenterFace)：人脸检测（SFW 马赛克数据集）。
* PyTorch, FFmpeg, GStreamer, GTK 以及[构建此生态系统的所有人](https://xkcd.com/2347/)。
