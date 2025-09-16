import torch
import torch.nn as nn


class SKAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=16, kernel_sizes=[1, 3, 5], gamma_range=(0.5, 2.0)):
        super(SKAttention, self).__init__()
        
        # Default output channels if not provided
        out_channels = out_channels or in_channels
        mid_channels = in_channels // reduction

        # Multi-scale convolutions (parallel)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, k, padding=k//2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])

        # Attention weight fusion (after multi-scale convolutions)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * len(kernel_sizes), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # Light adaptation module (learnable parameters to adapt low-light regions)
        self.illumination_adapt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(mid_channels, in_channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(mid_channels, in_channels // reduction), 1),
            nn.Sigmoid()
        )

        # Noise suppression module (small kernel convolution)
        self.noise_suppression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Adaptive contrast enhancement module
        self.contrast_alpha = nn.Parameter(torch.tensor(1.0))
        self.contrast_beta = nn.Parameter(torch.tensor(0.0))

        # Gamma correction (learnable)
        self.gamma_param = nn.Parameter(torch.tensor(1.0))
        self.gamma_min, self.gamma_max = gamma_range

    def apply_gamma_correction(self, x):
        """Apply learnable gamma correction"""
        gamma = torch.clamp(self.gamma_param, self.gamma_min, self.gamma_max)
        eps = 1e-6
        return torch.pow(x + eps, gamma)

    def adaptive_contrast_enhancement(self, x):
        """Adaptive contrast enhancement"""
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True)
        normalized = (x - mean) / (std + 1e-6)
        enhanced = self.contrast_alpha * normalized + self.contrast_beta
        return torch.sigmoid(enhanced) * x

    def forward(self, x):
        # Apply light adaptation
        illumination_factor = self.illumination_adapt(x)  # [B, 1]
        illumination_factor = illumination_factor.view(x.size(0), 1, 1, 1)
        x_illum = x * (1.0 + illumination_factor)  # Enhance low-light regions
        
        # Apply noise suppression
        x_noisy = self.noise_suppression(x_illum)
        
        # Apply gamma correction
        x_gamma = self.apply_gamma_correction(x_noisy)
        
        # Apply adaptive contrast enhancement
        x_contrast = self.adaptive_contrast_enhancement(x_gamma)

        # Apply multi-scale convolutions
        out = [conv(x_contrast) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [B, C×num_kernels, H, W]
        
        # Apply attention weight fusion
        weight = self.fuse(out)  # Attention map [B, C, H, W]
        
        # Apply attention to the input and return the output
        return x * weight  # Weighted output 