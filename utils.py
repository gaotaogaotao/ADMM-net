import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        self.layer_weights = layer_weights or {'conv3_3': 1.0}
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.criterion(pred, target)


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.char_loss = CharbonnierLoss()
    
    def forward(self, pred, target):
        loss = self.l1_weight * self.l1_loss(pred, target)
        loss += self.char_loss(pred, target)
        return loss


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def save_image(tensor, path):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    image = tensor.cpu().clone()
    image = transforms.ToPILImage()(image)
    image.save(path)


def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    image = tensor.cpu().clone()
    image = transforms.ToPILImage()(image)
    return image


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=50, decay_rate=0.5):
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
