import unittest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import DiceLoss

class TestDiceLoss(unittest.TestCase):
    def test_perfect_match(self):
        """If pred == target, loss should be roughly 0"""
        loss_fn = DiceLoss()
        # Logits: High positive value = 1 after sigmoid
        pred_logits = torch.ones((1, 1, 128, 128)) * 10 
        target = torch.ones((1, 1, 128, 128))
        
        loss = loss_fn(pred_logits, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_complete_mismatch(self):
        """If pred is opposite of target, loss should be roughly 1"""
        loss_fn = DiceLoss()
        # Logits: High negative value = 0 after sigmoid
        pred_logits = torch.ones((1, 1, 128, 128)) * -10 
        target = torch.ones((1, 1, 128, 128))
        
        loss = loss_fn(pred_logits, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

if __name__ == '__main__':
    unittest.main()