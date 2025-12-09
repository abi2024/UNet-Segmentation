import yaml
import torch
import torch.nn as nn
import random
import numpy as np

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    """Sets seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class DiceLoss(nn.Module):
    """Custom Dice Loss for PyTorch."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply Sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1. - dice

def get_loss_fn(loss_name, device, pos_weight=None):
    """Factory for Loss Functions."""
    if loss_name == 'bce':
        weight = torch.tensor([pos_weight]).to(device) if pos_weight else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    elif loss_name == 'dice':
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")