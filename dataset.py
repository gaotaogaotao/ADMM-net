import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class DeblurDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, patch_size=256):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        self.image_pairs = []
        
        if mode == 'train':
            data_dir = os.path.join(root_dir, 'train')
        else:
            data_dir = os.path.join(root_dir, 'test')
        
        scene_folders = [f for f in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, f))]
        
        for scene in scene_folders:
            scene_path = os.path.join(data_dir, scene)
            blur_dir = os.path.join(scene_path, 'blur')
            sharp_dir = os.path.join(scene_path, 'sharp')
            
            if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
                blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
                
                for blur_path in blur_images:
                    filename = os.path.basename(blur_path)
                    sharp_path = os.path.join(sharp_dir, filename)
                    
                    if os.path.exists(sharp_path):
                        self.image_pairs.append((blur_path, sharp_path))
        
        print(f"Loaded {len(self.image_pairs)} image pairs for {mode} mode")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        blur_path, sharp_path = self.image_pairs[idx]
        
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        else:
            to_tensor = transforms.ToTensor()
            blur_img = to_tensor(blur_img)
            sharp_img = to_tensor(sharp_img)
        
        if self.mode == 'train' and self.patch_size > 0:
            blur_img, sharp_img = self._random_crop(blur_img, sharp_img)
        
        return {'blur': blur_img, 'sharp': sharp_img, 'blur_path': blur_path}
    
    def _random_crop(self, blur, sharp):
        _, h, w = blur.shape
        patch_h = min(self.patch_size, h)
        patch_w = min(self.patch_size, w)
        
        top = random.randint(0, h - patch_h)
        left = random.randint(0, w - patch_w)
        
        blur_crop = blur[:, top:top+patch_h, left:left+patch_w]
        sharp_crop = sharp[:, top:top+patch_h, left:left+patch_w]
        
        return blur_crop, sharp_crop


def get_dataloader(root_dir, mode='train', batch_size=8, num_workers=4, 
                   patch_size=256, image_size=None):
    if image_size:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()
    
    dataset = DeblurDataset(
        root_dir=root_dir,
        mode=mode,
        transform=transform,
        patch_size=patch_size if mode == 'train' else 0
    )
    
    shuffle = (mode == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
