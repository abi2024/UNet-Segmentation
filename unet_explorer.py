"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      U-NET FEATURE EXPLORER                                   ║
║                                                                              ║
║  A comprehensive tool to understand your U-Net implementation:               ║
║  1. Analyze every trainable parameter (what it is, what it does)            ║
║  2. Visualize feature maps at each layer                                     ║
║  3. See the complete transformation from pixels → semantics → mask          ║
║                                                                              ║
║  Usage:                                                                       ║
║    python unet_feature_explorer.py                           # Synthetic     ║
║    python unet_feature_explorer.py --image path/to/pet.jpg   # Real image   ║
║    python unet_feature_explorer.py --weights model.pth       # Trained model║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from collections import OrderedDict
from PIL import Image
import argparse
import json

# ============================================================================
# PART 1: U-NET MODEL DEFINITION (Copy of your models.py)
# ============================================================================

class DoubleConv(nn.Module):
    """
    The fundamental building block of U-Net: Two convolutions with BatchNorm and ReLU.
    
    WHAT IT LEARNS:
    - First conv: Transforms input channels → output channels
    - Second conv: Refines the features at the same channel depth
    - BatchNorm: Learns scale (γ) and shift (β) to normalize activations
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    The U-Net Architecture with configurable down/upsampling.
    
    ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────────┐
    │  INPUT (3ch, 128x128)                                           │
    │      ↓                                                          │
    │  [inc] DoubleConv → 64 channels, 128x128                       │
    │      ↓ (pool)                                                   │
    │  [down1] DoubleConv → 128 channels, 64x64                      │
    │      ↓ (pool)                                                   │
    │  [down2] DoubleConv → 256 channels, 32x32                      │
    │      ↓ (pool)                                                   │
    │  [down3] DoubleConv → 512 channels, 16x16                      │
    │      ↓ (pool)                                                   │
    │  [bridge] DoubleConv → 1024 channels, 8x8  ← BOTTLENECK        │
    │      ↓ (upsample)                                               │
    │  [up1] DoubleConv → 512 channels, 16x16 (+ skip from down3)    │
    │      ↓ (upsample)                                               │
    │  [up2] DoubleConv → 256 channels, 32x32 (+ skip from down2)    │
    │      ↓ (upsample)                                               │
    │  [up3] DoubleConv → 128 channels, 64x64 (+ skip from down1)    │
    │      ↓ (upsample)                                               │
    │  [up4] DoubleConv → 64 channels, 128x128 (+ skip from inc)     │
    │      ↓                                                          │
    │  [outc] 1x1 Conv → 1 channel, 128x128 (segmentation logits)    │
    └─────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, downsample_mode='mp', upsample_mode='tr'):
        super().__init__()
        self.down_mode = downsample_mode
        self.up_mode = upsample_mode

        # --- ENCODER (Contracting Path) ---
        self.inc = DoubleConv(3, 64)
        
        self.down1_pool = self._make_down_layer(64)
        self.down1_conv = DoubleConv(64, 128)
        
        self.down2_pool = self._make_down_layer(128)
        self.down2_conv = DoubleConv(128, 256)
        
        self.down3_pool = self._make_down_layer(256)
        self.down3_conv = DoubleConv(256, 512)
        
        # --- BRIDGE (Bottleneck) ---
        self.bridge_pool = self._make_down_layer(512)
        self.bridge_conv = DoubleConv(512, 1024)
        
        # --- DECODER (Expanding Path) ---
        self.up1 = self._make_up_layer(1024, 512)
        self.up1_conv = DoubleConv(1024, 512)  # 1024 = 512 (upsampled) + 512 (skip)
        
        self.up2 = self._make_up_layer(512, 256)
        self.up2_conv = DoubleConv(512, 256)   # 512 = 256 + 256
        
        self.up3 = self._make_up_layer(256, 128)
        self.up3_conv = DoubleConv(256, 128)   # 256 = 128 + 128
        
        self.up4 = self._make_up_layer(128, 64)
        self.up4_conv = DoubleConv(128, 64)    # 128 = 64 + 64
        
        # --- OUTPUT ---
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def _make_down_layer(self, in_c):
        if self.down_mode == 'mp':
            return nn.MaxPool2d(2)
        elif self.down_mode == 'str_conv':
            return nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1)

    def _make_up_layer(self, in_c, out_c):
        if self.up_mode == 'tr':
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        elif self.up_mode == 'ups':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        
        x2_in = self.down1_pool(x1)
        x2 = self.down1_conv(x2_in)

        x3_in = self.down2_pool(x2)
        x3 = self.down2_conv(x3_in)

        x4_in = self.down3_pool(x3)
        x4 = self.down3_conv(x4_in)
        
        # Bridge
        bot_in = self.bridge_pool(x4)
        x5 = self.bridge_conv(bot_in)

        # Decoder with skip connections
        x = self.up1(x5)
        if x.shape != x4.shape:
            x = nn.functional.interpolate(x, size=x4.shape[2:])
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)
        
        x = self.up2(x)
        if x.shape != x3.shape:
            x = nn.functional.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        x = self.up3(x)
        if x.shape != x2.shape:
            x = nn.functional.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        x = self.up4(x)
        if x.shape != x1.shape:
            x = nn.functional.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)
        
        return self.outc(x)


# ============================================================================
# PART 2: PARAMETER ANALYZER - Understand Every Trainable Weight
# ============================================================================

class ParameterAnalyzer:
    """
    Analyzes and explains every trainable parameter in your U-Net.
    
    KEY CONCEPT: A "trainable parameter" is a number that changes during training.
    These are the "knobs" that gradient descent adjusts to minimize loss.
    
    In your U-Net, trainable parameters are found in:
    1. Conv2d layers: weights (filters) and biases
    2. BatchNorm2d layers: gamma (scale) and beta (shift)
    3. ConvTranspose2d layers: weights and biases (for upsampling)
    """
    
    def __init__(self, model):
        self.model = model
        
    def count_parameters(self):
        """Count total trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total
    
    def analyze_conv2d(self, name, module):
        """
        Explain what a Conv2d layer learns.
        
        Conv2d Weight Shape: [out_channels, in_channels, kernel_H, kernel_W]
        
        INTUITION:
        - out_channels = number of different patterns to detect
        - in_channels = how many input feature maps to look at
        - kernel_H × kernel_W = size of the "sliding window" (typically 3×3)
        
        Each filter is a small pattern detector. Early filters learn edges,
        later filters learn more complex patterns like textures and shapes.
        """
        out_c, in_c, kh, kw = module.weight.shape
        weight_params = module.weight.numel()
        bias_params = module.bias.numel() if module.bias is not None else 0
        
        return {
            'name': name,
            'type': 'Conv2d',
            'total_params': weight_params + bias_params,
            'weight_shape': f'[{out_c}, {in_c}, {kh}, {kw}]',
            'weight_params': weight_params,
            'bias_params': bias_params,
            'explanation': f"""
    🔍 CONV2D: {name}
    ├── Output Channels: {out_c} (learns {out_c} different filters/patterns)
    ├── Input Channels: {in_c} (looks at {in_c} feature maps)
    ├── Kernel Size: {kh}×{kw} (each filter is a {kh}×{kw} pixel window)
    ├── Weight Parameters: {out_c} × {in_c} × {kh} × {kw} = {weight_params:,}
    └── Bias Parameters: {bias_params:,} (one bias per output channel)
    
    WHAT IT LEARNS:
    Each of the {out_c} filters learns to detect a specific pattern.
    The filter "slides" across the input and produces high values where
    the pattern matches. Think of it as {out_c} specialized "pattern detectors".
"""
        }
    
    def analyze_batchnorm(self, name, module):
        """
        Explain BatchNorm parameters.
        
        BatchNorm learns TWO things per channel:
        - gamma (weight): How much to scale the normalized values
        - beta (bias): How much to shift the normalized values
        
        WHY: After normalizing to mean=0, std=1, the network might need
        different scales for different channels. gamma and beta let it
        "undo" normalization selectively.
        """
        num_features = module.num_features
        
        return {
            'name': name,
            'type': 'BatchNorm2d',
            'total_params': 2 * num_features,
            'gamma_params': num_features,
            'beta_params': num_features,
            'explanation': f"""
    📊 BATCHNORM: {name}
    ├── Channels: {num_features}
    ├── Gamma (scale): {num_features} params - learned multiplier per channel
    └── Beta (shift): {num_features} params - learned offset per channel
    
    WHAT IT LEARNS:
    After normalizing activations to mean=0, std=1, BatchNorm learns
    the optimal scale and offset for each channel. This helps training
    by keeping activations in a "healthy" range.
"""
        }
    
    def analyze_convtranspose2d(self, name, module):
        """
        Explain ConvTranspose2d (Transposed/Deconvolution).
        
        This is "learnable upsampling" - it learns how to expand spatial size
        while producing smooth, meaningful features.
        
        Weight Shape: [in_channels, out_channels, kernel_H, kernel_W]
        (Note: shape is transposed compared to Conv2d!)
        """
        in_c, out_c, kh, kw = module.weight.shape
        weight_params = module.weight.numel()
        bias_params = module.bias.numel() if module.bias is not None else 0
        
        return {
            'name': name,
            'type': 'ConvTranspose2d',
            'total_params': weight_params + bias_params,
            'weight_shape': f'[{in_c}, {out_c}, {kh}, {kw}]',
            'explanation': f"""
    🔼 CONV TRANSPOSE: {name}
    ├── Input Channels: {in_c}
    ├── Output Channels: {out_c}
    ├── Kernel Size: {kh}×{kw}
    ├── Weight Parameters: {weight_params:,}
    └── Bias Parameters: {bias_params:,}
    
    WHAT IT LEARNS:
    This is "learnable upsampling". Instead of simple interpolation,
    it learns filters that expand the spatial resolution while
    producing smooth, artifact-free features. Essential for the
    decoder to reconstruct fine details.
"""
        }
    
    def full_analysis(self):
        """Generate complete parameter breakdown."""
        print("\n" + "=" * 80)
        print("🧠 U-NET PARAMETER ANALYSIS")
        print("=" * 80)
        
        total_params = self.count_parameters()
        print(f"\n📊 TOTAL TRAINABLE PARAMETERS: {total_params:,}")
        print(f"   (That's {total_params:,} numbers the network learns from data!)\n")
        
        # Categorize parameters
        encoder_params = 0
        decoder_params = 0
        bridge_params = 0
        
        layer_details = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 'outc' not in name:
                info = self.analyze_conv2d(name, module)
                layer_details.append(info)
                
                if 'inc' in name or 'down' in name:
                    encoder_params += info['total_params']
                elif 'bridge' in name:
                    bridge_params += info['total_params']
                else:
                    decoder_params += info['total_params']
                    
            elif isinstance(module, nn.BatchNorm2d):
                info = self.analyze_batchnorm(name, module)
                layer_details.append(info)
                
                if 'inc' in name or 'down' in name:
                    encoder_params += info['total_params']
                elif 'bridge' in name:
                    bridge_params += info['total_params']
                else:
                    decoder_params += info['total_params']
                    
            elif isinstance(module, nn.ConvTranspose2d):
                info = self.analyze_convtranspose2d(name, module)
                layer_details.append(info)
                decoder_params += info['total_params']
            
            elif isinstance(module, nn.Conv2d) and 'outc' in name:
                info = self.analyze_conv2d(name, module)
                layer_details.append(info)
                decoder_params += info['total_params']
        
        # Print summary
        print("📈 PARAMETER DISTRIBUTION:")
        print(f"   ├── Encoder (contracting path): {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
        print(f"   ├── Bridge (bottleneck):        {bridge_params:,} ({100*bridge_params/total_params:.1f}%)")
        print(f"   └── Decoder (expanding path):   {decoder_params:,} ({100*decoder_params/total_params:.1f}%)")
        
        print("\n" + "-" * 80)
        print("📋 LAYER-BY-LAYER BREAKDOWN")
        print("-" * 80)
        
        for info in layer_details:
            print(info['explanation'])
        
        return {
            'total_params': total_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'bridge_params': bridge_params,
            'layers': layer_details
        }
    
    def visualize_parameter_distribution(self, save_path='parameter_distribution.png'):
        """Create a visual diagram of parameter distribution."""
        # Collect data by layer type
        conv_params = []
        bn_params = []
        layer_names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                params = sum(p.numel() for p in module.parameters())
                conv_params.append(params)
                bn_params.append(0)
                # Simplify name
                simple_name = name.replace('.conv.', '_').replace('.', '_')
                layer_names.append(simple_name[:15])
            elif isinstance(module, nn.BatchNorm2d):
                params = sum(p.numel() for p in module.parameters())
                conv_params.append(0)
                bn_params.append(params)
                simple_name = name.replace('.conv.', '_').replace('.', '_')
                layer_names.append(simple_name[:15])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(layer_names))
        width = 0.8
        
        bars1 = ax.bar(x, conv_params, width, label='Conv/TransposeConv Weights', color='steelblue')
        bars2 = ax.bar(x, bn_params, width, bottom=conv_params, label='BatchNorm Params', color='coral')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Number of Parameters', fontsize=12)
        ax.set_title('U-Net Parameter Distribution by Layer\n'
                    'Notice: Most parameters are in Conv layers. BatchNorm adds minimal overhead.',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        
        # Add total annotation
        total = self.count_parameters()
        ax.annotate(f'Total: {total:,} parameters', xy=(0.98, 0.98), xycoords='axes fraction',
                   ha='right', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved parameter distribution to: {save_path}")
        return save_path


# ============================================================================
# PART 3: FEATURE MAP EXTRACTOR - Hook into Intermediate Layers
# ============================================================================

class FeatureMapExtractor:
    """
    Extracts intermediate feature maps using PyTorch hooks.
    
    HOW IT WORKS:
    PyTorch lets you register "hooks" - functions that are called every time
    a layer produces output. We use this to capture intermediate activations
    without modifying the model's forward() method.
    
    This is the key to "opening the black box"!
    """
    
    def __init__(self, model):
        self.model = model
        self.activations = OrderedDict()
        self.hooks = []
        
    def _get_activation(self, name):
        """Factory function that creates a hook for a specific layer."""
        def hook(module, input, output):
            # Detach from computation graph and store
            self.activations[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self):
        """Register hooks on all important layers."""
        self.remove_hooks()
        self.activations.clear()
        
        # Define which layers to capture
        # These are the key points in the U-Net architecture
        layers_to_capture = {
            # Encoder
            '1_enc_inc': self.model.inc,           # First conv block (64 ch)
            '2_enc_down1': self.model.down1_conv,  # After 1st pool (128 ch)
            '3_enc_down2': self.model.down2_conv,  # After 2nd pool (256 ch)
            '4_enc_down3': self.model.down3_conv,  # After 3rd pool (512 ch)
            # Bridge
            '5_bridge': self.model.bridge_conv,     # Bottleneck (1024 ch)
            # Decoder
            '6_dec_up1': self.model.up1_conv,      # After 1st upsample (512 ch)
            '7_dec_up2': self.model.up2_conv,      # After 2nd upsample (256 ch)
            '8_dec_up3': self.model.up3_conv,      # After 3rd upsample (128 ch)
            '9_dec_up4': self.model.up4_conv,      # After 4th upsample (64 ch)
            # Output
            '10_output': self.model.outc,           # Final output (1 ch)
        }
        
        for name, layer in layers_to_capture.items():
            hook = layer.register_forward_hook(self._get_activation(name))
            self.hooks.append(hook)
            
        print(f"✅ Registered {len(self.hooks)} hooks on key U-Net layers")
            
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def extract_features(self, image_tensor):
        """
        Run forward pass and capture all intermediate feature maps.
        
        Args:
            image_tensor: Shape [1, 3, H, W] (single image with batch dim)
            
        Returns:
            activations: OrderedDict of feature maps
            output: Final model output
        """
        self.model.eval()
        self.activations.clear()
        
        with torch.no_grad():
            output = self.model(image_tensor)
            
        return self.activations.copy(), output


# ============================================================================
# PART 4: VISUALIZATION ENGINE - Make Feature Maps Beautiful
# ============================================================================

class FeatureVisualizer:
    """
    Creates beautiful, educational visualizations of feature maps.
    
    WHAT YOU'LL LEARN BY LOOKING AT THESE:
    
    1. EARLY LAYERS (inc, down1):
       - Look like edge detectors (horizontal, vertical, diagonal)
       - Similar to classical Sobel/Gabor filters
       - The network "rediscovers" classical computer vision!
    
    2. MIDDLE LAYERS (down2, down3):
       - Detect parts of objects (ears, eyes, nose)
       - Activations become spatially localized
       - Specific channels "light up" for specific features
    
    3. BRIDGE/BOTTLENECK:
       - Most compressed (8×8 spatial, 1024 channels)
       - Contains abstract "semantic concepts"
       - Individual channels might represent "is_cat", "has_fur", etc.
    
    4. DECODER LAYERS (up1 → up4):
       - Gradually reconstruct spatial detail
       - Skip connections bring back fine-grained information
       - Watch resolution increase while meaning is preserved
    """
    
    def __init__(self, save_dir='feature_explorer_output'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Output directory: {save_dir}/")
        
    def visualize_single_layer(self, feature_map, layer_name, num_channels=16):
        """
        Visualize individual channels from a feature map.
        
        Each channel in a feature map is like a "heat map" showing where
        a specific pattern was detected in the image.
        """
        features = feature_map.squeeze(0).numpy()  # Remove batch dim: [C, H, W]
        
        num_channels = min(num_channels, features.shape[0])
        num_total_channels = features.shape[0]
        
        cols = 4
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        fig.suptitle(f'Feature Maps: {layer_name}\n'
                    f'Showing {num_channels} of {num_total_channels} channels | '
                    f'Spatial size: {features.shape[1]}×{features.shape[2]}',
                    fontsize=14, fontweight='bold')
        
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        for i in range(num_channels):
            ax = axes[i]
            channel = features[i]
            
            # Use viridis colormap (yellow = high activation, purple = low)
            im = ax.imshow(channel, cmap='viridis')
            ax.set_title(f'Channel {i}\nmax={channel.max():.2f}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{layer_name}_channels.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_layer_summary(self, feature_map, layer_name):
        """
        Create summary visualizations: mean, max, and std across channels.
        
        - MEAN: Average activation - shows "consensus" of what all filters see
        - MAX: Maximum activation - shows strongest response anywhere
        - STD: Standard deviation - shows where filters disagree (interesting regions!)
        """
        features = feature_map.squeeze(0).numpy()  # [C, H, W]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Layer Summary: {layer_name} ({features.shape[0]} channels, {features.shape[1]}×{features.shape[2]} spatial)',
                    fontsize=14, fontweight='bold')
        
        # Mean across channels
        mean_map = features.mean(axis=0)
        im0 = axes[0].imshow(mean_map, cmap='hot')
        axes[0].set_title('Mean Activation\n(Consensus view)', fontsize=11)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Max across channels
        max_map = features.max(axis=0)
        im1 = axes[1].imshow(max_map, cmap='hot')
        axes[1].set_title('Max Activation\n(Strongest response)', fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Std across channels
        std_map = features.std(axis=0)
        im2 = axes[2].imshow(std_map, cmap='hot')
        axes[2].set_title('Std Deviation\n(Where filters disagree)', fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # Channel activity histogram
        channel_means = features.mean(axis=(1, 2))
        axes[3].bar(range(len(channel_means)), channel_means, color='steelblue', alpha=0.7)
        axes[3].set_title('Channel Activity\n(Mean per channel)', fontsize=11)
        axes[3].set_xlabel('Channel Index')
        axes[3].set_ylabel('Mean Activation')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{layer_name}_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_unet_journey(self, activations, original_image, final_output):
        """
        Create the MAIN visualization: complete journey through the U-Net.
        
        This shows how an image transforms from pixels → abstract semantics → mask.
        """
        fig = plt.figure(figsize=(22, 14))
        
        # Title
        fig.suptitle('🔬 THE U-NET JOURNEY: From Pixels to Segmentation Mask\n'
                    'Watch how the image is compressed into abstract concepts, '
                    'then reconstructed with spatial precision',
                    fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.3)
        
        # Row 0: Original image + Encoder
        # Original input
        ax_orig = fig.add_subplot(gs[0, 0])
        img = original_image.squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax_orig.imshow(img)
        ax_orig.set_title('INPUT IMAGE\n3 channels, 128×128', fontweight='bold', fontsize=10)
        ax_orig.axis('off')
        
        # Encoder layers
        encoder_layers = [
            ('1_enc_inc', '64ch, 128×128\n"Edge Detectors"'),
            ('2_enc_down1', '128ch, 64×64\n"Textures"'),
            ('3_enc_down2', '256ch, 32×32\n"Parts"'),
            ('4_enc_down3', '512ch, 16×16\n"Objects"'),
        ]
        
        for i, (layer_name, description) in enumerate(encoder_layers):
            if layer_name in activations:
                ax = fig.add_subplot(gs[0, i + 1])
                feat = activations[layer_name].squeeze(0)
                mean_activation = feat.mean(dim=0).numpy()
                ax.imshow(mean_activation, cmap='hot')
                ax.set_title(f'ENCODER\n{description}', fontsize=9, fontweight='bold')
                ax.axis('off')
        
        # Row 1: Bridge (center)
        if '5_bridge' in activations:
            ax_bridge = fig.add_subplot(gs[1, 2:4])
            feat = activations['5_bridge'].squeeze(0)
            mean_activation = feat.mean(dim=0).numpy()
            ax_bridge.imshow(mean_activation, cmap='hot')
            ax_bridge.set_title('🌉 BRIDGE / BOTTLENECK\n1024 channels, 8×8\n"Abstract Semantic Concepts"',
                              fontsize=12, fontweight='bold')
            ax_bridge.axis('off')
            
            # Add annotation
            ax_bridge.annotate('Most compressed\nrepresentation!', xy=(0.5, -0.15),
                             xycoords='axes fraction', ha='center', fontsize=10,
                             style='italic', color='darkred')
        
        # Row 2: Decoder + Output
        decoder_layers = [
            ('6_dec_up1', '512ch, 16×16\n"Refine Objects"'),
            ('7_dec_up2', '256ch, 32×32\n"Refine Parts"'),
            ('8_dec_up3', '128ch, 64×64\n"Refine Textures"'),
            ('9_dec_up4', '64ch, 128×128\n"Refine Edges"'),
        ]
        
        for i, (layer_name, description) in enumerate(decoder_layers):
            if layer_name in activations:
                ax = fig.add_subplot(gs[2, i])
                feat = activations[layer_name].squeeze(0)
                mean_activation = feat.mean(dim=0).numpy()
                ax.imshow(mean_activation, cmap='hot')
                ax.set_title(f'DECODER\n{description}', fontsize=9, fontweight='bold')
                ax.axis('off')
        
        # Final output
        ax_out = fig.add_subplot(gs[2, 4])
        output_sigmoid = torch.sigmoid(final_output).squeeze().numpy()
        ax_out.imshow(output_sigmoid, cmap='gray')
        ax_out.set_title('OUTPUT\n1ch, 128×128\n"Segmentation Mask"', fontsize=10, fontweight='bold')
        ax_out.axis('off')
        
        # Thresholded output
        ax_thresh = fig.add_subplot(gs[2, 5])
        binary_mask = (output_sigmoid > 0.5).astype(float)
        ax_thresh.imshow(binary_mask, cmap='gray')
        ax_thresh.set_title('PREDICTION\n(threshold=0.5)', fontsize=10, fontweight='bold')
        ax_thresh.axis('off')
        
        # Add arrows showing the flow
        # (These are approximate - matplotlib arrows across subplots are tricky)
        
        save_path = os.path.join(self.save_dir, 'unet_complete_journey.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved U-Net journey to: {save_path}")
        return save_path
    
    def visualize_learned_filters(self, model, save_path=None):
        """
        Visualize the actual learned convolutional filters.
        
        Early layer filters often look like classical edge detectors:
        - Horizontal edges
        - Vertical edges
        - Diagonal edges
        - Blob detectors
        """
        # Get the first conv layer weights
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                break
        
        if first_conv is None:
            print("⚠️ No Conv2d layer found")
            return None
        
        weights = first_conv.weight.detach().cpu().numpy()
        # Shape: [out_channels, in_channels, H, W]
        
        num_filters = min(32, weights.shape[0])
        
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(f'Learned Filters: {first_conv_name}\n'
                    f'Each 3×3 grid is a pattern the network learned to detect\n'
                    f'(Averaged across RGB input channels for visualization)',
                    fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for i in range(num_filters):
            ax = axes[i]
            # Average across input channels for visualization
            filter_viz = weights[i].mean(axis=0)
            
            # Use diverging colormap centered at 0
            vmax = max(abs(filter_viz.min()), abs(filter_viz.max()))
            ax.imshow(filter_viz, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Filter {i}', fontsize=8)
            ax.axis('off')
            
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_path = save_path or os.path.join(self.save_dir, 'learned_filters.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved learned filters to: {save_path}")
        return save_path
    
    def compare_random_vs_trained(self, random_activations, trained_activations, 
                                  original_image, random_output, trained_output):
        """
        Side-by-side comparison of random vs trained model.
        
        This is THE most educational visualization - it shows exactly
        what the network LEARNED from data!
        """
        fig, axes = plt.subplots(3, 6, figsize=(20, 12))
        
        fig.suptitle('🎓 WHAT THE NETWORK LEARNED\n'
                    'Top: Random initialization (meaningless noise)\n'
                    'Middle: Trained model (structured, meaningful features)\n'
                    'Bottom: The difference (what training added)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        layers_to_compare = ['1_enc_inc', '3_enc_down2', '5_bridge', '7_dec_up2', '9_dec_up4']
        
        # Row 0: Random model
        axes[0, 0].imshow(original_image.squeeze(0).permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Input', fontweight='bold')
        axes[0, 0].axis('off')
        
        for i, layer in enumerate(layers_to_compare):
            if layer in random_activations:
                feat = random_activations[layer].squeeze(0).mean(dim=0).numpy()
                axes[0, i + 1].imshow(feat, cmap='hot')
                axes[0, i + 1].set_title(f'Random\n{layer}', fontsize=10)
                axes[0, i + 1].axis('off')
        
        # Row 1: Trained model
        axes[1, 0].imshow(original_image.squeeze(0).permute(1, 2, 0).numpy())
        axes[1, 0].set_title('Input', fontweight='bold')
        axes[1, 0].axis('off')
        
        for i, layer in enumerate(layers_to_compare):
            if layer in trained_activations:
                feat = trained_activations[layer].squeeze(0).mean(dim=0).numpy()
                axes[1, i + 1].imshow(feat, cmap='hot')
                axes[1, i + 1].set_title(f'Trained\n{layer}', fontsize=10)
                axes[1, i + 1].axis('off')
        
        # Row 2: Outputs comparison
        axes[2, 0].set_title('Output Comparison', fontweight='bold')
        axes[2, 0].axis('off')
        
        # Random output
        random_pred = torch.sigmoid(random_output).squeeze().numpy()
        axes[2, 1].imshow(random_pred, cmap='gray')
        axes[2, 1].set_title('Random Output\n(Meaningless)', fontsize=10)
        axes[2, 1].axis('off')
        
        # Trained output
        trained_pred = torch.sigmoid(trained_output).squeeze().numpy()
        axes[2, 2].imshow(trained_pred, cmap='gray')
        axes[2, 2].set_title('Trained Output\n(Segmentation!)', fontsize=10)
        axes[2, 2].axis('off')
        
        # Binary predictions
        axes[2, 3].imshow((random_pred > 0.5).astype(float), cmap='gray')
        axes[2, 3].set_title('Random Binary', fontsize=10)
        axes[2, 3].axis('off')
        
        axes[2, 4].imshow((trained_pred > 0.5).astype(float), cmap='gray')
        axes[2, 4].set_title('Trained Binary', fontsize=10)
        axes[2, 4].axis('off')
        
        # Overlay
        img = original_image.squeeze(0).permute(1, 2, 0).numpy()
        overlay = img.copy()
        mask = trained_pred > 0.5
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green overlay
        axes[2, 5].imshow(np.clip(overlay, 0, 1))
        axes[2, 5].set_title('Prediction Overlay', fontsize=10)
        axes[2, 5].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'random_vs_trained.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved random vs trained comparison to: {save_path}")
        return save_path


# ============================================================================
# PART 5: MAIN EXPLORER CLASS
# ============================================================================

class UNetFeatureExplorer:
    """
    Main class that orchestrates the entire exploration.
    
    USAGE:
        explorer = UNetFeatureExplorer()
        results = explorer.explore(image_path='my_pet.jpg', weights_path='model.pth')
    """
    
    def __init__(self, downsample_mode='mp', upsample_mode='tr', 
                 output_dir='feature_explorer_output'):
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("🔬 U-NET FEATURE EXPLORER")
        print("=" * 70)
        print(f"   Mode: downsample={downsample_mode}, upsample={upsample_mode}")
        print(f"   Output: {output_dir}/")
        print("=" * 70)
        
        # Initialize model
        self.model = UNet(downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        
        # Initialize components
        self.analyzer = ParameterAnalyzer(self.model)
        self.extractor = FeatureMapExtractor(self.model)
        self.visualizer = FeatureVisualizer(save_dir=output_dir)
        
    def load_weights(self, weights_path):
        """Load trained weights from a .pth file."""
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded trained weights from: {weights_path}")
            return True
        else:
            print(f"⚠️ Weights file not found: {weights_path}")
            return False
    
    def load_image(self, image_path, size=(128, 128)):
        """Load and preprocess a real image."""
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            img = img.resize(size, Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            print(f"✅ Loaded image: {image_path}")
            print(f"   Shape: {img_tensor.shape} (batch, channels, height, width)")
            return img_tensor
        else:
            print(f"⚠️ Image not found: {image_path}")
            return None
    
    def create_synthetic_image(self):
        """
        Create a synthetic test image (circle on gradient background).
        Useful for testing without needing a real pet image.
        """
        print("🎨 Creating synthetic test image...")
        
        size = 128
        img = np.zeros((size, size, 3), dtype=np.float32)
        
        # Create gradient background
        for i in range(size):
            img[i, :, 0] = i / size * 0.5  # Red gradient top-bottom
            img[:, i, 2] = i / size * 0.5  # Blue gradient left-right
        
        # Add green base
        img[:, :, 1] = 0.2
        
        # Draw a filled circle (simulating a pet-like blob)
        center = (size // 2, size // 2)
        radius = size // 3
        
        y, x = np.ogrid[:size, :size]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Make the circle a brownish "pet" color
        img[mask] = [0.7, 0.5, 0.3]
        
        # Add some texture noise
        noise = np.random.rand(size, size, 3) * 0.1
        img = np.clip(img + noise, 0, 1)
        
        img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        print(f"   Shape: {img_tensor.shape}")
        
        return img_tensor
    
    def explore(self, image_source='synthetic', weights_path=None):
        """
        Run the complete exploration!
        
        Args:
            image_source: Path to an image file, or 'synthetic' for test image
            weights_path: Path to .pth file with trained weights (optional)
        """
        generated_files = []
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Parameter Analysis
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("📊 STEP 1: Analyzing Model Parameters")
        print("─" * 70)
        
        param_report = self.analyzer.full_analysis()
        
        # Save parameter distribution chart
        param_chart_path = self.analyzer.visualize_parameter_distribution(
            os.path.join(self.output_dir, 'parameter_distribution.png')
        )
        generated_files.append(('Parameter Distribution', param_chart_path))
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Prepare Test Image
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("🖼️ STEP 2: Preparing Test Image")
        print("─" * 70)
        
        if image_source == 'synthetic':
            image_tensor = self.create_synthetic_image()
        else:
            image_tensor = self.load_image(image_source)
            if image_tensor is None:
                print("   Falling back to synthetic image...")
                image_tensor = self.create_synthetic_image()
        
        # Save input image
        plt.figure(figsize=(6, 6))
        plt.imshow(image_tensor.squeeze(0).permute(1, 2, 0).numpy())
        plt.title('Input Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        input_path = os.path.join(self.output_dir, 'input_image.png')
        plt.savefig(input_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(('Input Image', input_path))
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Extract Features from RANDOM Model
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("🎲 STEP 3: Extracting Features from RANDOM Model")
        print("─" * 70)
        
        random_model = UNet(downsample_mode=self.downsample_mode, 
                           upsample_mode=self.upsample_mode)
        random_extractor = FeatureMapExtractor(random_model)
        random_extractor.register_hooks()
        random_activations, random_output = random_extractor.extract_features(image_tensor)
        random_extractor.remove_hooks()
        
        print(f"   Captured {len(random_activations)} layer activations")
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Load Trained Weights (if provided)
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("🏋️ STEP 4: Loading Trained Model")
        print("─" * 70)
        
        has_trained_weights = False
        if weights_path:
            has_trained_weights = self.load_weights(weights_path)
        
        if has_trained_weights:
            self.extractor.register_hooks()
            trained_activations, trained_output = self.extractor.extract_features(image_tensor)
            self.extractor.remove_hooks()
        else:
            print("   ⚠️ No trained weights - using random model for all visualizations")
            print("   💡 To see the full power, provide a trained .pth file!")
            trained_activations = random_activations
            trained_output = random_output
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 5: Generate Visualizations
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("🎨 STEP 5: Generating Visualizations")
        print("─" * 70)
        
        # 5a: Complete U-Net Journey
        print("   → Creating U-Net journey visualization...")
        path = self.visualizer.visualize_unet_journey(
            trained_activations, image_tensor, trained_output
        )
        generated_files.append(('U-Net Journey', path))
        
        # 5b: Individual layer feature maps
        layers_to_visualize = ['1_enc_inc', '3_enc_down2', '5_bridge', '7_dec_up2', '9_dec_up4']
        
        for layer_name in layers_to_visualize:
            if layer_name in trained_activations:
                print(f"   → Visualizing {layer_name}...")
                
                # Individual channels
                path = self.visualizer.visualize_single_layer(
                    trained_activations[layer_name], 
                    layer_name,
                    num_channels=16
                )
                generated_files.append((f'{layer_name} Channels', path))
                
                # Summary statistics
                path = self.visualizer.visualize_layer_summary(
                    trained_activations[layer_name],
                    layer_name
                )
                generated_files.append((f'{layer_name} Summary', path))
        
        # 5c: Learned filters
        print("   → Visualizing learned convolutional filters...")
        path = self.visualizer.visualize_learned_filters(self.model)
        if path:
            generated_files.append(('Learned Filters', path))
        
        # 5d: Random vs Trained comparison
        if has_trained_weights:
            print("   → Creating Random vs Trained comparison...")
            path = self.visualizer.compare_random_vs_trained(
                random_activations, trained_activations,
                image_tensor, random_output, trained_output
            )
            generated_files.append(('Random vs Trained', path))
        
        # ─────────────────────────────────────────────────────────────────
        # SUMMARY
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("✅ EXPLORATION COMPLETE!")
        print("=" * 70)
        
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        print("\n📋 Generated Files:")
        for name, path in generated_files:
            print(f"   • {name}: {os.path.basename(path)}")
        
        # Print learning guide
        print_learning_guide()
        
        return {
            'parameter_report': param_report,
            'random_activations': random_activations,
            'trained_activations': trained_activations,
            'generated_files': generated_files
        }


# ============================================================================
# PART 6: EDUCATIONAL GUIDE
# ============================================================================

def print_learning_guide():
    """Print a guide explaining what to look for in the visualizations."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📚 HOW TO READ YOUR VISUALIZATIONS                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔹 EARLY ENCODER LAYERS (inc, down1):                                      ║
║     • Look like EDGE DETECTORS: horizontal, vertical, diagonal lines        ║
║     • These are similar to classical Sobel/Gabor filters                    ║
║     • The network "rediscovers" computer vision basics!                     ║
║                                                                              ║
║  🔹 MIDDLE LAYERS (down2, down3):                                           ║
║     • Detect PARTS of objects: ears, eyes, nose, fur patterns              ║
║     • Activations become spatially localized                                ║
║     • Certain channels "light up" for specific features                     ║
║                                                                              ║
║  🔹 BRIDGE/BOTTLENECK (bridge):                                             ║
║     • Most compressed: 8×8 spatial, 1024 channels                          ║
║     • Contains ABSTRACT SEMANTIC CONCEPTS                                   ║
║     • Individual channels might represent "is_cat", "has_fur", etc.        ║
║     • This is where the network "understands" what it's looking at         ║
║                                                                              ║
║  🔹 DECODER LAYERS (up1 → up4):                                             ║
║     • Gradually RECONSTRUCT spatial details                                 ║
║     • Skip connections bring back fine-grained info from encoder           ║
║     • Watch resolution increase while semantic meaning is preserved         ║
║                                                                              ║
║  🔹 CHANNEL STATISTICS:                                                      ║
║     • High std dev = "active" channel (detecting something present)        ║
║     • Low std dev = "quiet" channel (feature not in this image)            ║
║     • This shows the network has learned SPECIALIZED detectors             ║
║                                                                              ║
║  🔹 RANDOM VS TRAINED:                                                       ║
║     • Random: Pure noise, no structure, random predictions                  ║
║     • Trained: Organized features, meaningful segmentation                  ║
║     • THE DIFFERENCE IS WHAT THE NETWORK LEARNED FROM YOUR DATA!           ║
║                                                                              ║
║  🔹 LEARNED FILTERS:                                                         ║
║     • Red/Blue colormap shows positive/negative weights                     ║
║     • Edge detectors have clear patterns (light on one side, dark on other)║
║     • Blob detectors have center different from surroundings               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    # =========================================================================
    # DEFAULT PATHS - Update these to match your project structure!
    # =========================================================================
    DEFAULT_WEIGHTS = r"C:\Users\ANT-PC\ERA_V4\Session_15_UNet\results\Exp2_MP_Tr_Dice_best.pth"
    DEFAULT_IMAGE = r"C:\Users\ANT-PC\ERA_V4\Session_15_UNet\data\raw\Oxford-IIT-PetDataset\images\yorkshire_terrier_178.jpg"
    # =========================================================================
    
    parser = argparse.ArgumentParser(
        description='U-Net Feature Explorer - Visualize what your network learns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default exploration (uses your trained model and Yorkshire Terrier image):
  python unet_feature_explorer.py

  # With a different pet image:
  python unet_feature_explorer.py --image path/to/other_pet.jpg

  # With different trained weights:
  python unet_feature_explorer.py --weights results/Exp1_MP_Tr_BCE_best.pth

  # Use synthetic test image instead:
  python unet_feature_explorer.py --image synthetic

  # Different architecture (for Exp3/Exp4):
  python unet_feature_explorer.py --down str_conv --up ups --weights results/Exp4_StrConv_Ups_Dice_best.pth
        """
    )
    
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE,
                       help=f'Path to input image, or "synthetic" for test image (default: {DEFAULT_IMAGE})')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                       help=f'Path to trained weights (.pth file) (default: {DEFAULT_WEIGHTS})')
    parser.add_argument('--output', type=str, default='feature_explorer_output',
                       help='Output directory for visualizations')
    parser.add_argument('--down', type=str, default='mp', choices=['mp', 'str_conv'],
                       help='Downsampling mode: mp (MaxPool) or str_conv (Strided Conv)')
    parser.add_argument('--up', type=str, default='tr', choices=['tr', 'ups'],
                       help='Upsampling mode: tr (TransposeConv) or ups (Upsample+Conv)')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = UNetFeatureExplorer(
        downsample_mode=args.down,
        upsample_mode=args.up,
        output_dir=args.output
    )
    
    # Run exploration
    results = explorer.explore(
        image_source=args.image,
        weights_path=args.weights
    )
    
    print("\n🎉 Done! Open the output folder to explore your U-Net!")
    print(f"   → {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()