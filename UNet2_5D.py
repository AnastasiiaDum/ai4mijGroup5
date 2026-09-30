import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Identity() if cin == cout
                     else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                        nn.BatchNorm2d(cout)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class Down(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResBlock(cin, cout, dropout)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.block = ResBlock(cout * 2, cout, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # guard against odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))


class UNet25D(nn.Module):
    """
    2.5D residual U-Net.
    Input : (B, in_channels, H, W)  -> in_channels = 2*n_neighbors + 1 adjacent slices
    Output: (B, num_classes, H, W)  -> logits for the CENTER slice
    """

    def __init__(self, in_channels=5, num_classes=5, base=32, dropout=0.1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ResBlock(in_channels, c[0])
        self.d1 = Down(c[0], c[1])
        self.d2 = Down(c[1], c[2])
        self.d3 = Down(c[2], c[3], dropout)
        self.d4 = Down(c[3], c[4], dropout)
        self.u1 = Up(c[4], c[3], dropout)
        self.u2 = Up(c[3], c[2])
        self.u3 = Up(c[2], c[1])
        self.u4 = Up(c[1], c[0])
        self.outc = nn.Conv2d(c[0], num_classes, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        y = self.u1(x5, x4)
        y = self.u2(y, x3)
        y = self.u3(y, x2)
        y = self.u4(y, x1)
        return self.outc(y)