项目目录结构

Unet++语义分割/
├── data/                     # 数据集根目录
│   └── cityscapes/           # Cityscapes 原始数据集（本项目数据源）
│       ├── leftImg8bit/      # 原始 RGB 图像
│       │   ├── train/        # 训练集图像（按城市分文件夹）
│       │   └── val/          # 验证集图像（按城市分文件夹）
│       └── gtFine/           # 精细标注
│           ├── train/        # 训练集标注（使用 *_gtFine_labelTrainIds.png）
│           └── val/          # 验证集标注（使用 *_gtFine_labelTrainIds.png）
├── logs/                     # 训练日志与权重保存根目录
│   ├── train_log/            # 训练损失、指标日志
│   └── weights/              # 训练过程保存的模型权重文件
├── predict/                  # 推理预测相关文件夹
│   ├── input/                # 待推理的原始测试图片存放目录
│   ├── output/               # 推理输出的结果图
│   ├── predict.ipynb         # Jupyter交互式推理调试脚本
│   └── predict.py            # 图片推理程序
├── tools/                    # 通用工具函数库
│   ├── __init__.py
│   └── animator.py           # 可视化/结果动画生成工具
├── training/                 # 模型定义、训练核心代码
│   ├── model.ipynb           # Jupyter交互式模型搭建、训练调试
│   └── model.py              # UNet++ 网络结构 + 训练循环完整代码
└── readme.txt                # 项目说明文档

数据集说明
- 数据集：Cityscapes 语义分割，共 19 类（trainIds 0~18），标注中的 255 为忽略像素。
- 图像路径：data/cityscapes/leftImg8bit/<split>/<城市>/*_leftImg8bit.png
- 标注路径：data/cityscapes/gtFine/<split>/<城市>/*_gtFine_labelTrainIds.png
- 训练集 2975 张、验证集 500 张。

训练
cd training && python model.py

推理
cd predict && python predict.py
