import torch
from tqdm import tqdm

class UNetTrainer:
    def __init__(self, model, optimizer, loss_fn, device, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler  # For Mixed Precision

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        loop = tqdm(loader, desc="Training", leave=False)
        
        for data, targets in loop:
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward
            with torch.amp.autocast('cuda'):
                logits = self.model(data)
                loss = self.loss_fn(logits, targets)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == targets).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            loop.set_postfix(loss=loss.item(), acc=acc.item())
            
        return total_loss / len(loader), total_acc / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    logits = self.model(data)
                    loss = self.loss_fn(logits, targets)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                total_loss += loss.item()
                total_acc += (preds == targets).float().mean().item()
                
                # Dice Score Metric
                intersection = (probs * targets).sum()
                dice = (2. * intersection + 1.0) / (probs.sum() + targets.sum() + 1.0)
                total_dice += dice.item()
                
        return total_loss / len(loader), total_acc / len(loader), total_dice / len(loader)