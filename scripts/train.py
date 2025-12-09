import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sys
import os

# --- PATH SETUP ---
# Add 'src' to the system path so we can import dataset and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import PetDataset
from models import UNet

# --- CONFIGURATION ---
# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
# Path to your dataset root (ensure this matches your folder name exactly)
DATA_PATH = "../data/raw/Oxford-IIT-PetDataset/" 

# --- CUSTOM LOSS FUNCTION ---
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, logits, targets):
        """
        logits: Raw output from model (No Sigmoid yet)
        targets: Binary ground truth (0 or 1)
        """
        # 1. Apply Sigmoid to turn Logits -> Probabilities (0 to 1)
        probs = torch.sigmoid(logits)
        
        # 2. Flatten tensors
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # 3. Calculate Dice
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1. - dice

# --- TRAINING LOOP ---
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    total_acc = 0.0
    
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Mixed Precision Context
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = loss_fn(logits, targets)

        # Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate Accuracy for monitoring
        # We must apply sigmoid manually to get predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=acc.item())
        
    return total_loss / len(loader), total_acc / len(loader)

# --- VALIDATION LOOP ---
def eval_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    dice_score = 0.0
    
    # No gradient needed for validation
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            with torch.amp.autocast('cuda'):
                logits = model(data)
                loss = loss_fn(logits, targets)
            
            # Metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item()
            total_acc += (preds == targets).float().mean().item()
            
            # Calculate Dice Score (Metric, not Loss)
            # Dice = 2*Intersection / Union
            intersection = (probs * targets).sum()
            union = probs.sum() + targets.sum()
            dice_score += (2. * intersection + 1.0) / (union + 1.0)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_dice = dice_score.item() / len(loader)
    
    return avg_loss, avg_acc, avg_dice

# --- MAIN EXPERIMENT RUNNER ---
def run_experiments():
    print(f"✅ Training on Device: {DEVICE}")
    
    # 1. Setup Data
    train_ds = PetDataset(DATA_PATH, split='train')
    val_ds = PetDataset(DATA_PATH, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Define Experiments
    experiments = [
        {"name": "Exp1_MP_Tr_BCE",      "down": "mp",       "up": "tr",  "loss": "bce"},
        {"name": "Exp2_MP_Tr_Dice",     "down": "mp",       "up": "tr",  "loss": "dice"},
        {"name": "Exp3_StrConv_Tr_BCE", "down": "str_conv", "up": "tr",  "loss": "bce"},
        {"name": "Exp4_StrConv_Ups_Dice","down": "str_conv", "up": "ups", "loss": "dice"},
    ]
    
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # 3. Run Loop
    for exp in experiments:
        print(f"\n{'='*40}")
        print(f"🚀 Starting: {exp['name']}")
        print(f"Config: Down={exp['down']}, Up={exp['up']}, Loss={exp['loss']}")
        print(f"{'='*40}")
        
        # Initialize Model
        model = UNet(downsample_mode=exp['down'], upsample_mode=exp['up']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler('cuda') # Modern PyTorch Syntax
        
        # Select Loss Function
        if exp['loss'] == "bce":
            # Add positive weight to force model to care about the Pet (Class 1)
            # Since Background is ~73% and Pet is ~27%, we weigh Pet x3 harder.
            pos_weight = torch.tensor([3.0]).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = DiceLoss()
                    
        # Storage for graphs
        history = {
            'loss': [], 'val_loss': [], 
            'acc': [], 'val_acc': [],
            'val_dice_coef': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            
            # Train
            train_loss, train_acc = train_fn(train_loader, model, optimizer, criterion, scaler)
            
            # Validate
            val_loss, val_acc, val_dice = eval_fn(val_loader, model, criterion)
            
            # Store Logs
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_dice_coef'].append(val_dice)
            
            print(f" -> Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"../results/{exp['name']}_best.pth")

        # Save History
        with open(f"../results/{exp['name']}_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        print(f"✅ {exp['name']} Completed.\n")

if __name__ == "__main__":
    run_experiments()