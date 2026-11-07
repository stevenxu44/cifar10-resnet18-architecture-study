# ResNet-18 结构改造与 CIFAR-10 分类

独立大作业项目。比较 `baseline`、`depthwise`、`se` 和 `depthwise_se` 四种结构。训练从随机初始化开始，输入为 32×32 RGB 图像；训练/验证集从 CIFAR-10 官方训练集固定划分为 45,000/5,000，官方测试集 10,000 张仅用于最终评价。仓库包含主实验及补充实验的最佳权重、配置、训练历史、指标、图表和示例输入；原始数据与生成的推理输出由脚本获取或生成。

## 快速运行（Windows PowerShell）

在项目目录执行；本机已有上级目录的 `.venv` 时，可直接激活该环境。

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python download_data.py
python verify_checkpoints.py
python run.py predict --weights results/baseline/best.pt --input examples/scene_inputs/cifar10_1_0000_airplane.png --output output/single_prediction.json
python app.py --input examples/scene_inputs --output output/scene_cards
```

`verify_checkpoints.py` 用原评价路径复核六个权重的清晰图测试正确数；最后两条命令分别生成单图 JSON 和四模型逐图卡片。仅用 CPU 时，按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)选择 CPU 版 2.8.0；其余命令相同。`download_data.py` 使用 TorchVision 获取 CIFAR-10，并从 [CIFAR-10.1 官方仓库](https://github.com/modestyachts/CIFAR-10.1)获取 v6 测试集，核对文件哈希。

## 环境

本机已验证：Python 3.9.1、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128、RTX 4060 Laptop GPU。依赖版本见 `requirements.txt`。在 PyCharm 中，项目解释器可选本项目的 `.venv/Scripts/python.exe`，或本机已有的上级目录 `.venv/Scripts/python.exe`；运行配置的工作目录须为本项目根目录。

## 训练

`python download_data.py` 获取数据后，在本目录执行：

```powershell
python run.py train --variant baseline --epochs 80
python run.py train --variant depthwise --epochs 80
python run.py train --variant se --epochs 80
python run.py train --variant depthwise_se --epochs 80
```

每组输出 `results/<variant>/config.json`、`history.json`、`best.pt` 和 `metrics.json`。训练时用验证集选择最佳权重，训练结束后才计算测试集指标。`--seed` 可指定随机种子；`--limit-train` 与 `--tag` 只用于短流程检查。

## 图片推理

```powershell
python run.py predict --weights results/baseline/best.pt --input examples/scene_inputs/cifar10_1_0000_airplane.png --output output/predictions.json
```

`--input` 可为单张图片或文件夹。输出是类别、概率和源图片路径的 JSON。只支持 CIFAR-10 的十个类别；其他类别的输入也会被强制归入其中之一。

阶段 4 的 `app.py` 可对四组模型同时推理，并保存逐图结果卡片：

```powershell
python app.py --input examples/scene_inputs --output output/scene_cards
```

卡片中的百分比为 softmax 输出；它不是经过校准的正确概率。真实照片会先缩放到模型训练使用的 32×32 输入。

## 独立图像与受控退化评价

运行 `python download_data.py` 获取 [CIFAR-10.1 v6](https://github.com/modestyachts/CIFAR-10.1)，然后执行：

```powershell
python scene_eval.py
```

该脚本对 2000 张新测试图及其模糊、低照度版本运行四组模型，并生成 `analysis/scene/` 下的指标与图。变换参数、实测数字及外部照片来源见 `阶段4结果.md`。

## 模糊鲁棒性改进

`train_robust.py` 从已训练的基线权重出发，在 CIFAR-10 训练集上加入随机高斯模糊与亮度增强，用原训练集划出的验证集选择权重。训练结束后评价原 CIFAR-10 测试集与 CIFAR-10.1。运行：

```powershell
python train_robust.py
python extra_blur_eval.py
python analyze_robust.py
python app.py --input examples/blurred_inputs --output output/robust_blurred_cards --include-robust
```

改进模型权重与指标在 `results/robust_baseline/`；对照数据与图在 `analysis/robust/`，见 `模糊鲁棒性改进结果.md`。`--include-robust` 会在分类卡片中加入改进模型。只运行该模型的单图推理可使用：

```powershell
python run.py predict --weights results/robust_baseline/best.pt --input examples/blurred_inputs/cifar10_1_0200_automobile.png --output output/robust_single_prediction.json
```

## 运动模糊专项增强

`train_motion.py` 从高斯模糊增强权重继续训练，随机加入水平、垂直及对角方向的运动模糊（核长 3/5），并保留高斯模糊与清晰图。最佳权重位于 `results/motion_robust/best.pt`。复现实验及生成对照图、示例卡片：

```powershell
python train_motion.py
python analyze_motion.py
python prepare_motion_examples.py
python app.py --input examples/motion_inputs --output output/motion_cards --include-robust --include-motion
```

训练会覆盖 `results/motion_robust/` 中已有权重和指标；仅复核现有结果时运行后三条命令。`app.py --include-motion` 会增加专项模型，`--include-robust` 可同时加入高斯增强模型。单图推理示例：

```powershell
python run.py predict --weights results/motion_robust/best.pt --input examples/motion_inputs/cifar10_1_1401_horse_shallow7.png --output output/motion_single_prediction.json
```

数据对比见 `运动模糊增强结果.md` 与 `analysis/motion/motion_metrics.json`；图在 `analysis/motion/motion_comparison.png`；20 张示例输入与结果卡片分别在 `examples/motion_inputs/`、`output/motion_cards/`。

## 指标说明

准确率是官方测试集 Top-1 准确率；MACs 按单张 32×32 输入由 `thop` 统计，FLOPs 如需报告按 `2 × MACs` 定义。延迟是本机 GPU 上预热后单张图片连续 100 次前向计算的平均值；它不包含磁盘读取与图像预处理。硬件延迟和理论运算量分别报告。

## 已保存的实验结果

| 结构 | CIFAR-10 测试准确率 | 参数量 | MACs/图 |
| --- | ---: | ---: | ---: |
| ResNet-18 基线 | 94.64% | 11,173,962 | 557,889,024 |
| 深度可分离卷积 | 93.27% | 1,439,626 | 75,515,392 |
| SE 注意力 | 94.46% | 11,263,042 | 558,223,744 |
| 深度可分离卷积 + SE | 93.67% | 1,528,706 | 75,850,112 |

数值来自 `results/<variant>/metrics.json`，汇总在 `analysis/summary.json`。场景适配的高斯增强和运动增强结果分别见 `模糊鲁棒性改进结果.md`、`运动模糊增强结果.md`。`results/` 包含可直接推理的权重；`data/` 和 `output/` 不纳入仓库，前者可由下载脚本获取，后者由推理命令生成。
