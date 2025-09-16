import torch.nn as nn
import torch.nn.init as init

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LayerNorm([self.channel // self.reduction]),
            nn.Linear(self.channel // self.reduction, self.channel),
            nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channel)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.he_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        x = self.residual(x)
        return x * y

# Example usage:
# se_att = SEAttention(channel=512, reduction=16)
# output = se_att(input_tensor)