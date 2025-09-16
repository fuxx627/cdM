import torch
import torch.nn as nn

class LowLightEnhancer(nn.Module):
    """Low-light Enhancement Module"""
    def __init__(self, c1, c2, reduction=16):  # 只接受 3 个参数
        """
        c1: 输入通道数 (自动从前一层的输出通道数传递)
        c2: 输出通道数 (从配置文件中获取)
        reduction: 注意力机制的压缩率 (从配置文件中获取)
        """
        super().__init__()
        # 确保输入输出通道数正确
        assert c1 == 3, f"输入通道数应为3 (RGB图像), 但得到 {c1}"
        assert c2 == 3, f"输出通道数应为3 (RGB图像), 但得到 {c2}"
        
        self.conv1 = nn.Conv2d(c1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        
        # 轻量级卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.conv2 = nn.Conv2d(32, c2, 3, padding=1)
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习增强系数
        
        # 通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(1, c2 // reduction), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, c2 // reduction), c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv_block(x)
        base = self.conv2(x)
        attn = self.attention(base)
        enhanced = x + self.alpha * attn * base
        return torch.clamp(enhanced, 0, 1)