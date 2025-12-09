import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PetDataset(Dataset):
    def __init__(self, root_dir, img_size=(128, 128), split='train'):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations", "trimaps")
        
        self.img_paths, self.mask_paths = self._get_file_paths()

    def _get_file_paths(self):
        img_paths = []
        mask_paths = []
        
        all_masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        
        # Simple split strategy: 80% train, 20% val
        split_idx = int(len(all_masks) * 0.8)
        if self.split == 'train':
            selected_masks = all_masks[:split_idx]
        else:
            selected_masks = all_masks[split_idx:]

        for mask_name in selected_masks:
            img_name = mask_name.replace('.png', '.jpg')
            img_full = os.path.join(self.img_dir, img_name)
            mask_full = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(img_full):
                img_paths.append(img_full)
                mask_paths.append(mask_full)
                
        return img_paths, mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize 0-1
        
        # Transpose HWC -> CHW (PyTorch Requirement)
        img = np.transpose(img, (2, 0, 1))

        # 2. Load Mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path) # Auto loads as mode 'L' (grayscale) or 'P'
        mask = mask.resize(self.img_size, Image.NEAREST) # Nearest for labels
        mask = np.array(mask, dtype=np.float32)
        
        # 3. Preprocess Mask
        mask = mask - 1  # 1,2,3 -> 0,1,2
        mask = np.where(mask == 0, 1.0, 0.0) # Binary: 0(Pet) -> 1, Others -> 0
        
        # Add channel dim: (H, W) -> (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)