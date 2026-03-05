import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import build_model
from admm import ADMMDeblur, DeepADMM
from utils import (
    CombinedLoss, 
    CharbonnierLoss,
    calculate_psnr, 
    calculate_ssim, 
    save_checkpoint, 
    load_checkpoint,
    AverageMeter, 
    get_lr, 
    adjust_learning_rate, 
    save_image
)


def parse_args():
    parser = argparse.ArgumentParser(description='ADMM Deep Deblurring Training')
    parser.add_argument('--root_dir', type=str, default='.', help='Dataset root directory')
    parser.add_argument('--model_type', type=str, default='admm', 
                        choices=['unet', 'admm', 'deep_admm'], help='Model type')
    parser.add_argument('--num_stages', type=int, default=5, help='Number of ADMM stages')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patch_size', type=int, default=256, help='Training patch size')
    parser.add_argument('--decay_epochs', type=int, default=50, help='Learning rate decay epochs')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    return parser.parse_args()


def build_model_from_type(args):
    if args.model_type == 'unet':
        model = build_model('unet')
    elif args.model_type == 'admm':
        model = ADMMDeblur(num_stages=args.num_stages)
    elif args.model_type == 'deep_admm':
        model = DeepADMM(num_stages=args.num_stages)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args, writer):
    model.train()
    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter() # 增加 ssim 指标
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for idx, batch in enumerate(pbar):
        blur = batch['blur'].to(device)
        sharp = batch['sharp'].to(device)
        
        optimizer.zero_grad()
        
        output = model(blur)
        
        loss = criterion(output, sharp)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            psnr = calculate_psnr(output, sharp)
            ssim = calculate_ssim(output, sharp) # 计算 ssim
        
        losses.update(loss.item(), blur.size(0))
        psnrs.update(psnr, blur.size(0))
        ssims.update(ssim, blur.size(0)) # 更新 ssim 指标
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'psnr': f'{psnrs.avg:.2f}dB',
            'ssim': f'{ssims.avg:.4f}', # 增加 ssim 指标
            'lr': f'{get_lr(optimizer):.6f}'
        })
        
        global_step = epoch * len(dataloader) + idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/psnr', psnr, global_step)
        writer.add_scalar('train/ssim', ssim, global_step) # 增加 ssim 指标
    
    return losses.avg, psnrs.avg, ssims.avg # 返回 ssim 指标


def validate(model, dataloader, criterion, device, epoch, args, writer):
    model.eval()
    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter() # 增加 ssim 指标
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            blur = batch['blur'].to(device)
            sharp = batch['sharp'].to(device)
            
            output = model(blur)
            
            loss = criterion(output, sharp)
            psnr = calculate_psnr(output, sharp)
            ssim = calculate_ssim(output, sharp) # 计算 ssim
            
            losses.update(loss.item(), blur.size(0))
            psnrs.update(psnr, blur.size(0))
            ssims.update(ssim, blur.size(0)) # 更新 ssim 指标
    
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/psnr', psnrs.avg, epoch)
    writer.add_scalar('val/ssim', ssims.avg, epoch) # 增加 ssim 指标
    
    print(f'Validation - Loss: {losses.avg:.4f}, PSNR: {psnrs.avg:.2f}dB, SSIM: {ssims.avg:.4f}')
    
    return losses.avg, psnrs.avg, ssims.avg # 返回 ssim 指标


def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading training data...')
    train_loader = get_dataloader(
        root_dir=args.root_dir,
        mode='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size
    )
    
    print('Loading validation data...')
    val_loader = get_dataloader(
        root_dir=args.root_dir,
        mode='test',
        batch_size=1,
        num_workers=args.num_workers,
        patch_size=0
    )
    
    print(f'Building model: {args.model_type}')
    model = build_model_from_type(args)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')
    
    criterion = CombinedLoss(l1_weight=1.0)
    criterion = criterion.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
        print(f'Resumed from epoch {start_epoch}')
    
    writer = SummaryWriter(args.log_dir)
    
    best_psnr = 0
    
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(
            optimizer, epoch, args.lr,
            args.decay_epochs, args.decay_rate
        )
        
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args, writer
        )
        
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB, Train SSIM: {train_ssim:.4f}')
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, criterion,
                device, epoch, args, writer
            )
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.save_dir, 'best_model.pth')
                )
                print(f'Saved best model with PSNR: {best_psnr:.2f}dB, SSIM: {val_ssim:.4f}')
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    save_checkpoint(
        model, optimizer, args.epochs, train_loss,
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()
