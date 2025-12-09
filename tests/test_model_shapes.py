import unittest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import UNet

class TestUNetShapes(unittest.TestCase):
    def test_mp_tr_shape(self):
        """Test MaxPool + Transpose Conv configuration"""
        model = UNet(downsample_mode='mp', upsample_mode='tr')
        # Batch=2, Channels=3, H=128, W=128
        dummy_input = torch.randn(2, 3, 128, 128)
        output = model(dummy_input)
        
        # Expected: Batch=2, Channels=1, H=128, W=128
        self.assertEqual(output.shape, (2, 1, 128, 128))

    def test_str_ups_shape(self):
        """Test Strided Conv + Upsample configuration"""
        model = UNet(downsample_mode='str_conv', upsample_mode='ups')
        dummy_input = torch.randn(1, 3, 128, 128)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 1, 128, 128))

if __name__ == '__main__':
    unittest.main()