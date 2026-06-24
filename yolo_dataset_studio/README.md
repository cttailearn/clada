# YOLO26 数据集准备 GUI 工具（YOLO Dataset Studio）

一款使用 **PyQt6** 编写的桌面工具，结合 [Ultralytics YOLO26](https://docs.ultralytics.com/zh/tasks/segment) 模型，帮助你：

1. 选择任务（**检测 / 分割 / 分类**）并导入一个预训练 `.pt` 模型；
2. 指定 **输入图像目录** 与 **输出数据集目录**；
3. 对输入目录下所有图片进行批量预标注；
4. 通过分页式画布查看并修改 AI 给出的标注；每一次修改都会自动写入 YOLO 标签文件；
5. 一键生成符合 Ultralytics 规范的训练数据集（含 `images/`、`labels/` 子目录与 `data.yaml`），可直接用于训练更精准的 YOLO26 模型。

## 安装

```powershell
# 强烈建议使用一个干净的 venv，避免与 conda 全局的 PyQt5 冲突
# Python 3.10 / 3.11 / 3.12 都已验证；Python 3.13 由于 PyQt6 6.11 在 Windows 上
# 仍有兼容问题，建议改用 3.12。
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r yolo_dataset_studio/requirements.txt
```

> `ultralytics` / `torch` 较大，下载缓慢时可先用国内镜像。仅启动 GUI、不进行推理时也可以暂时不装，工具会回退到「无模型演示模式」（自动生成占位框/多边形/类别，方便先体验完整界面流程）。

> 如果遇到 `ImportError: DLL load failed while importing QtCore`，多半是 **同环境里同时存在 PyQt5 与 PyQt6**。请新建干净的 venv 后再次 `pip install -r requirements.txt`。

## 启动

```powershell
python yolo_dataset_studio/run.py
```

## 工作流

1. **配置页**：选择任务（检测/分割/分类）→ 选择 `.pt` 模型 → 选择输入目录 → 选择输出目录 → 检查/修改类别。
2. **推理页**：点击「开始推理」，所有图片会在后台线程中跑预标注，进度条同步刷新。
3. **检视页**：左侧缩略图列表；中部画布支持滚轮缩放、空格+左键拖拽平移；按 `1` 切到选择工具、`2` 切到矩形（检测）、`3` 切到多边形（分割），按 `A` / `D` 翻页，`F` 标记当前已审。
4. **导出页**：调整 train/val/test 划分 → 预览 `data.yaml` → 点击「生成数据集」。

## 生成的数据集结构

```
<output>/
├── images/{train,val,test}/...
├── labels/{train,val,test}/...        # 分类任务不会生成
├── data.yaml
└── TRAIN.md
```

随后即可：

```bash
yolo segment train data=<output>/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
yolo detect  train data=<output>/data.yaml model=yolo26n.pt     epochs=100 imgsz=640
yolo classify train data=<output> model=yolo26n-cls.pt epochs=100 imgsz=224
```
