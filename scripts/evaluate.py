import torch
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet
from dataset import PetDataset
from utils import load_config, get_loss_fn
from trainer import UNetTrainer
from torch.utils.data import DataLoader

def evaluate_model(config_path, weights_path):
    # 1. Load Configuration
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device} using {config['experiment_name']}")

    # 2. Prepare Data
    val_ds = PetDataset(config['data']['path'], split='val')
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)

    # 3. Build Model
    model = UNet(
        downsample_mode=config['model']['downsample_mode'],
        upsample_mode=config['model']['upsample_mode']
    ).to(device)

    # 4. Load Weights
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded weights from {weights_path}")
    else:
        print(f"❌ Weights file not found: {weights_path}")
        return

    # 5. Setup Trainer (Just for eval function)
    loss_fn = get_loss_fn(config['training']['loss'], device)
    trainer = UNetTrainer(model, None, loss_fn, device)

    # 6. Run Evaluation
    loss, acc, dice = trainer.evaluate(val_loader)
    
    print("\n" + "="*30)
    print(f"RESULTS FOR {config['experiment_name']}")
    print("="*30)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Dice Score: {dice:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    # Example usage: python evaluate.py --config ../configs/exp1_mp_tr_bce.yaml --weights ../results/Exp1_MP_Tr_BCE_best.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to yaml config")
    parser.add_argument('--weights', type=str, required=True, help="Path to .pth model weights")
    args = parser.parse_args()
    
    evaluate_model(args.config, args.weights)