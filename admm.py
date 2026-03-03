import torch
import torch.nn as nn
import torch.nn.functional as F


class ADMMSolver(nn.Module):
    def __init__(self, rho=1.0, max_iter=10, tol=1e-4):
        super(ADMMSolver, self).__init__()
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
    
    def forward(self, blur, prior_net, data_fidelity_fn):
        x = blur.clone()
        z = torch.zeros_like(blur)
        u = torch.zeros_like(blur)
        
        for _ in range(self.max_iter):
            x_old = x.clone()
            x = self._update_x(blur, z, u, data_fidelity_fn)
            z = self._update_z(x, u, prior_net)
            u = u + x - z
            
            if torch.norm(x - x_old) < self.tol:
                break
        
        return x
    
    def _update_x(self, blur, z, u, data_fidelity_fn):
        x = data_fidelity_fn(blur)
        x = (x + self.rho * (z - u)) / (1 + self.rho)
        return torch.clamp(x, 0, 1)
    
    def _update_z(self, x, u, prior_net):
        z_input = x + u
        z = prior_net(z_input)
        return z


class DeepADMM(nn.Module):
    def __init__(self, num_stages=5, channels=64):
        super(DeepADMM, self).__init__()
        self.num_stages = num_stages
        
        self.prior_nets = nn.ModuleList()
        self.data_nets = nn.ModuleList()
        self.rho_nets = nn.ModuleList()
        
        for _ in range(num_stages):
            self.prior_nets.append(self._build_prior_net(channels))
            self.data_nets.append(self._build_data_net(channels))
            self.rho_nets.append(self._build_rho_net())
    
    def _build_prior_net(self, channels):
        return nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1)
        )
    
    def _build_data_net(self, channels):
        return nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1)
        )
    
    def _build_rho_net(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(6, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, blur):
        x = blur.clone()
        z = torch.zeros_like(blur)
        u = torch.zeros_like(blur)
        
        for i in range(self.num_stages):
            x_data = self.data_nets[i](blur)
            z_prior = self.prior_nets[i](x + u)
            rho = self.rho_nets[i](torch.cat([x, blur], dim=1)) * 10 + 0.01
            
            x = (x_data + rho * (z_prior - u)) / (1 + rho)
            x = torch.clamp(x, 0, 1)
            
            z = self.prior_nets[i](x + u)
            z = torch.clamp(z, 0, 1)
            
            u = u + x - z
        
        return x


class ADMMDeblur(nn.Module):
    def __init__(self, num_stages=5, use_learnable_rho=True):
        super(ADMMDeblur, self).__init__()
        self.num_stages = num_stages
        self.use_learnable_rho = use_learnable_rho
        
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            self.stages.append(ADMMStage(64, use_learnable_rho))
        
        self.final_conv = nn.Conv2d(3, 3, 3, padding=1)
    
    def forward(self, blur):
        x = blur.clone()
        z = torch.zeros_like(blur)
        u = torch.zeros_like(blur)
        
        for stage in self.stages:
            x, z, u = stage(blur, x, z, u)
        
        out = self.final_conv(x)
        return torch.clamp(out, 0, 1)


class ADMMStage(nn.Module):
    def __init__(self, channels, use_learnable_rho=True):
        super(ADMMStage, self).__init__()
        self.use_learnable_rho = use_learnable_rho
        
        self.prior_net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1)
        )
        
        self.data_net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1)
        )
        
        if use_learnable_rho:
            self.rho_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(6, 32, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Softplus()
            )
        else:
            self.register_buffer('rho', torch.tensor(1.0))
    
    def forward(self, blur, x, z, u):
        z_input = x + u
        z_new = self.prior_net(z_input)
        z_new = torch.clamp(z_new, 0, 1)
        
        x_data = self.data_net(blur)
        
        if self.use_learnable_rho:
            rho = self.rho_net(torch.cat([x, blur], dim=1))
        else:
            rho = self.rho
        
        x_new = (x_data + rho * (z_new - u)) / (1 + rho)
        x_new = torch.clamp(x_new, 0, 1)
        
        u_new = u + x_new - z_new
        
        return x_new, z_new, u_new
