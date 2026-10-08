# Unet++ 语义分割 — conda 运行环境配置

本项目基于 PyTorch 实现 UNet++ 语义分割，数据集为 **Cityscapes（19 类）**。
本文档用于在本地重新部署时快速搭建运行环境。

---

## 1. 环境概览（当前已验证版本）

| 组件 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Linux | 已验证 |
| Python | 3.11.1 | |
| PyTorch | 2.10.0+cu128 | CUDA 12.8 构建 |
| torchvision | 0.25.0+cu128 | |
| numpy | 2.3.5 | |
| matplotlib | 3.10.8 | 训练曲线可视化 |
| Pillow | 12.1.0 | 图像读写 |
| tqdm | 4.67.3 | 进度条 |
| notebook | 7.5.3 | 运行 .ipynb |
| ipykernel | 7.1.0 | Jupyter 内核 |
| NVIDIA 驱动 | 580.65.06 | 支持 CUDA 13.0，向下兼容 cu128 |

> 无 GPU 时程序会自动回退到 CPU，但训练速度会非常慢，不建议。

---

## 2. 创建并激活 conda 环境

```bash
conda create -n unetpp python=3.11 -y
conda activate unetpp
```

---

## 3. 安装依赖

### 方式一：pip 安装（推荐，与当前验证环境一致）

```bash
# 1) 安装 GPU 版 PyTorch（CUDA 12.8）
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 2) 安装其余依赖
pip install numpy==2.3.5 matplotlib==3.10.8 pillow==12.1.0 tqdm==4.67.3
pip install notebook ipykernel jupyter
```

如果本机 CUDA 版本不同，请替换 `--index-url`：

| 本机 CUDA | index-url |
|---|---|
| CUDA 12.8 | `https://download.pytorch.org/whl/cu128` |
| CUDA 12.1 | `https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| 仅 CPU | `https://download.pytorch.org/whl/cpu` |

### 方式二：conda 安装 PyTorch

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy matplotlib pillow tqdm notebook ipykernel
```

### 可选：使用 environment.yml 一键创建

```yaml
name: unetpp
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy=2.3.5
  - matplotlib=3.10.8
  - pillow=12.1.0
  - tqdm=4.67.3
  - jupyter
  - ipykernel
  - pip:
      - torch==2.10.0
      - torchvision==0.25.0
```

使用方式：

```bash
conda env create -f environment.yml
conda activate unetpp
```

---

## 4. 验证环境是否可用

```bash
python -c "import torch, torchvision, numpy, PIL, matplotlib; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available())"
nvidia-smi
```

期望输出中 `cuda_avail` 为 `True`。

---

## 5. 数据集准备（Cityscapes）

把 Cityscapes 数据集放到项目内的 `data/cityscapes`，目录结构如下：

```
Unet++语义分割/
└── data/
    └── cityscapes/
        ├── leftImg8bit/
        │   ├── train/<城市>/*_leftImg8bit.png      # 原始 RGB 图像
        │   └── val/<城市>/*_leftImg8bit.png
        └── gtFine/
            ├── train/<城市>/*_gtFine_labelTrainIds.png   # 语义标签（像素值即类别 0~18，255 忽略）
            └── val/<城市>/*_gtFine_labelTrainIds.png
```

- 训练集 2975 张、验证集 500 张。
- 若本机已有 Cityscapes 数据的其它副本，可用硬链接直接引用（不额外占磁盘）：

```bash
cd Unet++语义分割
mkdir -p data/cityscapes/leftImg8bit data/cityscapes/gtFine
cp -al /path/to/cityscapes/leftImg8bit/train /path/to/cityscapes/leftImg8bit/val data/cityscapes/leftImg8bit/
cp -al /path/to/cityscapes/gtFine/train    /path/to/cityscapes/gtFine/val    data/cityscapes/gtFine/
```

> 说明：压缩包内**未包含 Cityscapes 数据集**（体积过大），请按上面步骤自行放置。

---

## 6. 训练 / 推理

```bash
# 训练（脚本内使用相对路径 ../data/cityscapes，必须在 training 目录下运行）
cd Unet++语义分割/training
python model.py

# 推理
cd ../predict
python predict.py
```

- 训练权重保存在 `logs/weights/unetpp_ep{N}.pth`（每 10 轮保存一次）。
- `predict.py` 中默认权重路径为 `../logs/weights/unetpp_ep150.pth`，请按实际生成的文件名修改。
- 训练日志写入 `logs/train_log/train_<时间戳>.log`，格式：`Epoch | TrainLoss | ValLoss | mIoU`。

---

## 7. Jupyter 运行（可选）

```bash
conda activate unetpp
python -m ipykernel install --user --name unetpp --display-name "Python (unetpp)"
jupyter notebook
```

打开 `training/model.ipynb` 或 `predict/predict.ipynb`，内核选择 `Python (unetpp)`。

---

## 8. 常见问题

1. **`torch.cuda.is_available()` 为 False**
   检查显卡驱动与 CUDA 版本是否匹配，或改用对应 index-url 安装 torch。
2. **显存不足（CUDA out of memory）**
   在 `training/model.py` 的 `main()` 中调小 `batch_size`（如 8 / 4）。
3. **找不到数据集**
   确认在 `training/` 目录下运行（脚本内为相对路径），且 `data/cityscapes` 已就位。
4. **类别数不对**
   Cityscapes 共 19 类，忽略像素 255，由 `NUM_CLASSES` / `IGNORE_INDEX` 控制，无需手动修改。
5. **权重加载报 key 不匹配**
   预测用的 `UNetPlusPlus(num_classes=...)` 必须与训练时一致（均为 19 类）。
