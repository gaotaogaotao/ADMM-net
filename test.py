import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from model import build_model
from admm import ADMMDeblur, DeepADMM
from utils import calculate_psnr, calculate_ssim, load_checkpoint, tensor_to_image


def parse_args():
    parser = argparse.ArgumentParser(description='ADMM Deep Deblurring Testing')
    parser.add_argument('--root_dir', type=str, default='.', help='Dataset root directory')
    parser.add_argument('--model_type', type=str, default='admm',
                        choices=['unet', 'admm', 'deep_admm'], help='Model type')
    parser.add_argument('--num_stages', type=int, default=5, help='Number of ADMM stages')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')
    parser.add_argument('--save_images', action='store_true', help='Save output images')
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


def load_image(path):
    img = Image.open(path).convert('RGB')
    to_tensor = transforms.ToTensor()
    return to_tensor(img)


def get_test_pairs(root_dir):
    test_dir = os.path.join(root_dir, 'test')
    pairs = []
    
    scene_folders = [f for f in os.listdir(test_dir)
                     if os.path.isdir(os.path.join(test_dir, f))]
    
    for scene in scene_folders:
        scene_path = os.path.join(test_dir, scene)
        blur_dir = os.path.join(scene_path, 'blur')
        sharp_dir = os.path.join(scene_path, 'sharp')
        
        if os.path.exists(blur_dir):
            blur_images = sorted([f for f in os.listdir(blur_dir) if f.endswith('.png')])
            
            for img_name in blur_images:
                blur_path = os.path.join(blur_dir, img_name)
                sharp_path = os.path.join(sharp_dir, img_name) if os.path.exists(sharp_dir) else None
                pairs.append((blur_path, sharp_path, scene, img_name))
    
    return pairs


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'Building model: {args.model_type}')
    model = build_model_from_type(args)
    model = model.to(device)
    
    print(f'Loading checkpoint: {args.checkpoint}')
    load_checkpoint(model, None, args.checkpoint)
    model.eval()
    
    print('Getting test image pairs...')
    test_pairs = get_test_pairs(args.root_dir)
    print(f'Found {len(test_pairs)} test images')
    
    psnrs = []
    ssims = []
    
    print('Testing...')
    with torch.no_grad():
        for blur_path, sharp_path, scene, img_name in tqdm(test_pairs):
            blur = load_image(blur_path).unsqueeze(0).to(device)
            
            output = model(blur)
            
            output_img = tensor_to_image(output.squeeze(0))
            
            if sharp_path and os.path.exists(sharp_path):
                sharp = load_image(sharp_path).unsqueeze(0).to(device)
                psnr = calculate_psnr(output, sharp)
                ssim = calculate_ssim(output, sharp)
                psnrs.append(psnr)
                ssims.append(ssim)
            
            if args.save_images:
                scene_output_dir = os.path.join(args.output_dir, scene)
                os.makedirs(scene_output_dir, exist_ok=True)
                output_path = os.path.join(scene_output_dir, img_name)
                output_img.save(output_path)
    
    if psnrs:
        print('\n' + '=' * 50)
        print('Test Results:')
        print(f'Average PSNR: {np.mean(psnrs):.2f} dB')
        print(f'Average SSIM: {np.mean(ssims):.4f}')
        print(f'PSNR Std: {np.std(psnrs):.2f}')
        print(f'SSIM Std: {np.std(ssims):.4f}')
        print('=' * 50)
    
    print(f'\nResults saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
