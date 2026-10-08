
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path



# ---------- 复制模型定义（与训练时完全一致）----------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    def forward(self, x):
        return self.up(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=12, deep_supervision=True, dropout=0.1):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 512]

        self.x00 = ConvBlock(in_channels, filters[0], dropout)
        self.x10 = ConvBlock(filters[0], filters[1], dropout)
        self.x20 = ConvBlock(filters[1], filters[2], dropout)
        self.x30 = ConvBlock(filters[2], filters[3], dropout)
        self.x40 = ConvBlock(filters[3], filters[4], dropout)
        self.pool = nn.MaxPool2d(2)

        self.up10 = UpBlock(filters[1], filters[0])
        self.x01 = ConvBlock(filters[0]*2, filters[0], dropout)
        self.up20 = UpBlock(filters[2], filters[1])
        self.x11 = ConvBlock(filters[1]*2, filters[1], dropout)
        self.up30 = UpBlock(filters[3], filters[2])
        self.x21 = ConvBlock(filters[2]*2, filters[2], dropout)
        self.up40 = UpBlock(filters[4], filters[3])
        self.x31 = ConvBlock(filters[3]*2, filters[3], dropout)

        self.up11 = UpBlock(filters[1], filters[0])
        self.x02 = ConvBlock(filters[0]*3, filters[0], dropout)
        self.up21 = UpBlock(filters[2], filters[1])
        self.x12 = ConvBlock(filters[1]*3, filters[1], dropout)
        self.up31 = UpBlock(filters[3], filters[2])
        self.x22 = ConvBlock(filters[2]*3, filters[2], dropout)

        self.up12 = UpBlock(filters[1], filters[0])
        self.x03 = ConvBlock(filters[0]*4, filters[0], dropout)
        self.up22 = UpBlock(filters[2], filters[1])
        self.x13 = ConvBlock(filters[1]*4, filters[1], dropout)

        self.up13 = UpBlock(filters[1], filters[0])
        self.x04 = ConvBlock(filters[0]*5, filters[0], dropout)

        if self.deep_supervision:
            self.out1 = nn.Conv2d(filters[0], num_classes, 1)
            self.out2 = nn.Conv2d(filters[0], num_classes, 1)
            self.out3 = nn.Conv2d(filters[0], num_classes, 1)
            self.out4 = nn.Conv2d(filters[0], num_classes, 1)
        else:
            self.out = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x31 = self.x31(torch.cat([x30, self.up40(x40)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up30(x31)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up20(x21)], dim=1))
        x01 = self.x01(torch.cat([x00, self.up10(x10)], dim=1))

        x22 = self.x22(torch.cat([x20, x21, self.up31(x31)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up21(x21)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up11(x11)], dim=1))

        x13 = self.x13(torch.cat([x10, x11, x12, self.up22(x22)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up12(x12)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up13(x13)], dim=1))

        if self.deep_supervision:
            return [self.out1(x01), self.out2(x02), self.out3(x03), self.out4(x04)]
        return self.out(x04)

# ---------- 颜色映射（将类别索引转为彩色可视化）----------
# Cityscapes 19 类官方调色板（标签 -> RGB）
LABEL_TO_COLOR = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}
NUM_CLASSES = len(LABEL_TO_COLOR)   # 19
IGNORE_INDEX = 255                  # 忽略像素，可视化时保持黑色

def label_to_rgb(label_np):
    """将 H×W 的类别索引数组转为 H×W×3 的 RGB 图像"""
    h, w = label_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL_TO_COLOR.items():
        rgb[label_np == label] = color
    return rgb

# ---------- 预测主函数 ----------
def predict_image(model, image_path, device, output_path=None, input_size=(256, 256)):
    """
    :param model: 已加载权重的模型
    :param image_path: 输入图片路径
    :param device: 运行设备
    :param output_path: 输出保存路径（若为 None 则自动生成）
    :param input_size: 模型期望的输入尺寸 (H, W)
    :return: 分割结果的 PIL Image（彩色可视化）
    """
    # 1. 读取并预处理图片
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (W, H)

    # 与训练时相同的预处理（Resize + Normalize）
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # 2. 推理
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)  # 如果 deep_supervision=True，返回列表
        if isinstance(outputs, list):
            output = outputs[-1]      # 取最后一个（最精细的输出）
        else:
            output = outputs
        pred = torch.argmax(output, dim=1).squeeze(0)  # [H, W]

    # 3. 后处理：将预测结果 resize 回原始尺寸（最近邻插值）
    pred_np = pred.cpu().numpy().astype(np.uint8)      # 256×256
    pred_resized = Image.fromarray(pred_np, mode='L')
    pred_resized = pred_resized.resize(orig_size, Image.NEAREST)
    pred_np_resized = np.array(pred_resized)

    # 4. 转为彩色 RGB 图像
    rgb_result = label_to_rgb(pred_np_resized)

    # 5. 保存
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(image_path))
        output_path = f"{base}_seg.png"
    Image.fromarray(rgb_result).save(output_path)
    print(f"预测结果已保存至: {output_path}")

    return Image.fromarray(rgb_result)

# ---------- 批量预测（遍历文件夹）----------
def predict_folder(model, input_dir, output_dir, device, input_size=(256, 256), ext_filter=('.png', '.jpg', '.jpeg')):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in ext_filter:
            out_path = output_dir / f"{img_path.stem}_seg.png"
            predict_image(model, str(img_path), device, str(out_path), input_size)



# ---------- 主程序入口 ----------
if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    weights_path = "../logs/weights/unetpp_ep150.pth"   # 训练好的权重
    input_folder = "./input"              # 待预测图片文件夹
    output_folder = "./output"            # 预测结果保存文件夹

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型并加载权重
    model = UNetPlusPlus(in_channels=3, num_classes=NUM_CLASSES, deep_supervision=True).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"模型加载成功: {weights_path}")

    # 2. 执行批量预测
    predict_folder(model, input_folder, output_folder, device)
    
    