import fvcore.common.config
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch.mask_cond_unet import Unet

# 加载图片
image_path = "test_image.jpg"
mask_path = "mask_image.jpg"
image = Image.open(image_path).convert("RGB")  # 确保图片是 RGB 格式
mask = Image.open(mask_path).convert("RGB")
# 图片预处理
transform = transforms.Compose([
    transforms.Resize((320, 320)),  # 调整图片大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

transform1 = transforms.Compose([
    transforms.Resize((80, 80)),  # 调整图片大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 将图片转换为 PyTorch 张量
image_tensor = transform1(image).unsqueeze(0)  # 添加 batch 维度
print("Image tensor shape:", image_tensor.shape)  #
mask_tensor = transform(mask).unsqueeze(0)  # 添加 batch 维度
print("Mask tensor shape:", mask_tensor.shape)
# 初始化模型
model = Unet(
    dim=128,
    dim_mults=(1, 2, 4, 4),
    cond_dim=128,
    cond_dim_mults=(2, 4, ),
    channels=3,  # 修改为 3，以匹配输入图片的通道数
    window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
    window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
    cfg=fvcore.common.config.CfgNode({
        'cond_pe': False,
        'input_size': [80, 80],
        'cond_feature_size': (32, 128),
        'cond_net': 'swin',
        'num_pos_feats': 96
    })
)

# 将模型设置为评估模式
model.eval()

# 模拟时间和条件输入
time = torch.tensor([0.5])  # 时间参数

# 前向传播
with torch.no_grad():  # 禁用梯度计算
    output1, output2 = model(image_tensor, time, mask_tensor)

print("Output 1 shape:", output1.shape)  # 输出形状: [1, 3, 320, 320]
print("Output 2 shape:", output2.shape)  # 输出形状: [1, 3, 320, 320]

# 将输出张量转换为图片
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)  # 去掉 batch 维度
    tensor = tensor.permute(1, 2, 0)  # 从 [C, H, W] 转换为 [H, W, C]
    tensor = (tensor * 0.5) + 0.5  # 反归一化
    tensor = tensor.clamp(0, 1)  # 限制像素值在 [0, 1] 范围内
    return tensor.cpu().numpy()

# 可视化输出
output1_image = tensor_to_image(output1)
output2_image = tensor_to_image(output2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Output 1")
plt.imshow(output1_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Output 2")
plt.imshow(output2_image)
plt.axis("off")

plt.show()

# 保存输出图片
output1_image = Image.fromarray((output1_image * 255).astype("uint8"))
output1_image.save("output1.jpg")

output2_image = Image.fromarray((output2_image * 255).astype("uint8"))
output2_image.save("output2.jpg")