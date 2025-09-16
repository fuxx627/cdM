import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLightAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(AdaptiveLightAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 光照感知分支
        self.light_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # 原始特征注意力分支
        self.feature_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # 权重融合
        self.weight_fc = nn.Sequential(
            nn.Linear(2 * channel, 2, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # 计算光照感知注意力
        light_att = self.light_fc(y)
        
        # 计算特征注意力
        feat_att = self.feature_fc(y)
        
        # 计算融合权重
        combined = torch.cat([light_att, feat_att], dim=1)
        weights = self.weight_fc(combined)  # [b, 2]
        
        # 融合两种注意力
        final_att = weights[:, 0:1] * light_att + weights[:, 1:2] * feat_att
        return x * final_att.view(b, c, 1, 1)

class LightAwareSKAttention(nn.Module):
    def __init__(self, channel=512, out_channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(LightAwareSKAttention, self).__init__()
        
        # 光照感知模块
        self.light_aware = AdaptiveLightAttention(channel, reduction)
        
        # 原始SKAttention的卷积分支
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k//2, groups=group),
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
            )
        
        # 注意力机制
        self.fc = nn.Linear(channel, max(L, channel//reduction))
        self.fcs = nn.ModuleList([
            nn.Linear(max(L, channel//reduction), channel) for _ in range(len(kernels))
        ])
        self.softmax = nn.Softmax(dim=0)
        
        # 输出层
        self.out_conv = nn.Conv2d(channel, out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用光照感知
        light_enhanced = self.light_aware(x)
        
        # 多尺度特征提取
        feats = [conv(light_enhanced) for conv in self.convs]
        
        # 特征融合
        U = sum(feats)
        
        # 全局池化
        S = U.mean(-1).mean(-1)
        
        # 注意力计算
        Z = self.fc(S)
        attention_weights = torch.stack([fc(Z).view(x.size(0), -1, 1, 1) for fc in self.fcs], dim=0)
        attention_weights = self.softmax(attention_weights)
        
        # 应用注意力
        V = (attention_weights * torch.stack(feats, dim=0)).sum(dim=0)
        
        # 最终输出
        out = self.out_conv(V)
        out = self.bn(out)
        out = self.relu(out)
        
        return out