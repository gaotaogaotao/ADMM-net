import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.features = features
        
        self.inc = DoubleConv(in_channels, features[0])
        
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))
        
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[i], features[i - 1]))
        
        self.outc = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        
        skip_connections = [x1]
        
        for down in self.downs:
            x1 = down(x1)
            skip_connections.append(x1)
        
        skip_connections = skip_connections[:-1]
        skip_connections = skip_connections[::-1]
        
        for idx, up in enumerate(self.ups):
            x1 = up(x1, skip_connections[idx])
        
        out = self.outc(x1)
        return out


class DeblurNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DeblurNet, self).__init__()
        self.unet = UNet(in_channels, out_channels)
    
    def forward(self, x):
        residual = self.unet(x)
        out = x + residual
        return torch.clamp(out, 0, 1)


class ADMMNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_iterations=5):
        super(ADMMNet, self).__init__()
        self.num_iterations = num_iterations
        
        self.prior_net = UNet(in_channels, out_channels)
        
        self.data_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
        self.rho_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, blur):
        x = blur.clone()
        
        for _ in range(self.num_iterations):
            x_prior = self.prior_net(x)
            
            x_data = self.data_net(blur)
            
            rho = self.rho_conv(torch.cat([x, blur], dim=1)) * 10 + 0.1
            
            x = (x_prior + rho * x_data) / (1 + rho)
            x = torch.clamp(x, 0, 1)
        
        return x


def build_model(model_type='unet', in_channels=3, out_channels=3, **kwargs):
    if model_type == 'unet':
        return DeblurNet(in_channels, out_channels)
    elif model_type == 'admm':
        return ADMMNet(in_channels, out_channels, **kwargs.get('num_iterations', 5))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
