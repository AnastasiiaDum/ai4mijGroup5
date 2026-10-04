import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.drop = (nn.Dropout3d(dropout) if dropout > 0 else nn.Identity())

        if cin == cout:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, kernel_size=1, bias=False),
                                      nn.BatchNorm3d(cout))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.skip(x)

        return F.relu(out, inplace=True)


class Down3D(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block = ResBlock3D(cin, cout, dropout)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)


class Up3D(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()

        self.up = nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2)

        self.block = ResBlock3D(cout * 2, cout, dropout)

    def forward(self, x, skip):

        x = self.up(x)

        # Handle odd dimensions
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)

        return self.block(x)


class ResUNet3D(nn.Module):

    def __init__(self, in_channels=1, num_classes=5, base=16,dropout=0.1):
        super().__init__()

        c = [base, base * 2, base * 4, base * 8,base * 16]

        # Encoder
        self.inc = ResBlock3D(in_channels, c[0])

        self.d1 = Down3D(c[0], c[1])
        self.d2 = Down3D(c[1], c[2])
        self.d3 = Down3D(c[2], c[3], dropout)
        self.d4 = Down3D(c[3], c[4], dropout)

        # Decoder
        self.u1 = Up3D(c[4], c[3], dropout)
        self.u2 = Up3D(c[3], c[2])
        self.u3 = Up3D( c[2], c[1])
        self.u4 = Up3D(c[1], c[0])

        # Output
        self.outc = nn.Conv3d(c[0], num_classes, kernel_size=1)

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm3d):
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