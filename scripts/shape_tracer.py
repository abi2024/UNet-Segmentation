"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      U-NET TENSOR SHAPE TRACER                               ║
║                                                                              ║
║  "Stop reading code; start drawing shapes."                                  ║
║                                                                              ║
║  This script traces a tensor through every layer of your U-Net,             ║
║  showing exactly where information is:                                       ║
║    📉 COMPRESSED (Pooling/Strided Conv) - spatial size decreases            ║
║    📈 EXPANDED (TransposeConv/Upsample) - spatial size increases            ║
║    🔗 CONCATENATED (Skip Connections) - channels double                     ║
║                                                                              ║
║  Usage:                                                                       ║
║    python unet_shape_tracer.py                    # Default 128x128         ║
║    python unet_shape_tracer.py --size 256         # Custom size             ║
║    python unet_shape_tracer.py --size 512 --down str_conv                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import argparse

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # For systems that don't support ANSI codes
    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.UNDERLINE = cls.END = ''


def format_shape(tensor):
    """Format tensor shape as a readable string."""
    shape = list(tensor.shape)
    return f"[{shape[0]}, {shape[1]:4d}, {shape[2]:3d}, {shape[3]:3d}]"


def format_shape_change(before, after, operation):
    """Show what changed between two tensor shapes."""
    b_shape = list(before.shape)
    a_shape = list(after.shape)
    
    changes = []
    
    # Check channels
    if b_shape[1] != a_shape[1]:
        changes.append(f"ch: {b_shape[1]}→{a_shape[1]}")
    
    # Check spatial
    if b_shape[2] != a_shape[2]:
        if a_shape[2] < b_shape[2]:
            changes.append(f"↓ spatial: {b_shape[2]}→{a_shape[2]}")
        else:
            changes.append(f"↑ spatial: {b_shape[2]}→{a_shape[2]}")
    
    return ", ".join(changes) if changes else "shape unchanged"


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def trace_conv_block(x, conv_block, block_name, indent=""):
    """Trace through a DoubleConv block showing each sub-layer."""
    print(f"\n{indent}{Colors.BOLD}┌─ {block_name}{Colors.END}")
    
    current = x
    for i, layer in enumerate(conv_block.conv):
        layer_name = layer.__class__.__name__
        
        if isinstance(layer, nn.Conv2d):
            before = current
            current = layer(current)
            params = count_parameters(layer)
            print(f"{indent}│  Conv2d({layer.in_channels}→{layer.out_channels}, k={layer.kernel_size[0]})")
            print(f"{indent}│  {Colors.CYAN}{format_shape(before)} → {format_shape(current)}{Colors.END}")
            print(f"{indent}│  params: {params:,}")
            
        elif isinstance(layer, nn.BatchNorm2d):
            current = layer(current)
            params = count_parameters(layer)
            print(f"{indent}│  BatchNorm2d({layer.num_features}) - params: {params}")
            
        elif isinstance(layer, nn.ReLU):
            current = layer(current)
            print(f"{indent}│  ReLU")
    
    print(f"{indent}└─ Output: {Colors.GREEN}{format_shape(current)}{Colors.END}")
    return current


def trace_downsample(x, pool_layer, layer_name, mode):
    """Trace through a downsampling layer."""
    before = x
    after = pool_layer(x)
    
    if mode == 'mp':
        print(f"\n{Colors.RED}📉 DOWNSAMPLE: MaxPool2d(2){Colors.END}")
    else:
        params = count_parameters(pool_layer)
        print(f"\n{Colors.RED}📉 DOWNSAMPLE: StridedConv(stride=2) - params: {params:,}{Colors.END}")
    
    print(f"   {Colors.CYAN}{format_shape(before)} → {format_shape(after)}{Colors.END}")
    print(f"   {Colors.YELLOW}Spatial reduced by 2x! Information compressed.{Colors.END}")
    
    return after


def trace_upsample(x, up_layer, layer_name, mode):
    """Trace through an upsampling layer."""
    before = x
    after = up_layer(x)
    
    if mode == 'tr':
        params = count_parameters(up_layer)
        print(f"\n{Colors.GREEN}📈 UPSAMPLE: ConvTranspose2d(stride=2) - params: {params:,}{Colors.END}")
    else:
        params = count_parameters(up_layer)
        print(f"\n{Colors.GREEN}📈 UPSAMPLE: Upsample + Conv - params: {params:,}{Colors.END}")
    
    print(f"   {Colors.CYAN}{format_shape(before)} → {format_shape(after)}{Colors.END}")
    print(f"   {Colors.YELLOW}Spatial increased by 2x! Reconstructing details.{Colors.END}")
    
    return after


def trace_skip_connection(upsampled, skip, name):
    """Trace a skip connection concatenation."""
    # Handle size mismatch
    if upsampled.shape[2:] != skip.shape[2:]:
        upsampled = nn.functional.interpolate(upsampled, size=skip.shape[2:])
        print(f"\n{Colors.YELLOW}⚠️  Size mismatch! Interpolated to match skip connection.{Colors.END}")
    
    concatenated = torch.cat([upsampled, skip], dim=1)
    
    print(f"\n{Colors.BLUE}🔗 SKIP CONNECTION: Concatenate{Colors.END}")
    print(f"   Upsampled: {format_shape(upsampled)}")
    print(f"   Skip:      {format_shape(skip)}")
    print(f"   {Colors.CYAN}Concatenated: {format_shape(concatenated)}{Colors.END}")
    print(f"   {Colors.YELLOW}Channels doubled! Fine details from encoder restored.{Colors.END}")
    
    return concatenated


def print_section_header(title, emoji=""):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{emoji} {Colors.BOLD}{title}{Colors.END}")
    print('='*70)


def print_architecture_diagram(input_size):
    """Print ASCII art of U-Net architecture with dimensions."""
    s = input_size
    diagram = f"""
    THE U-NET ARCHITECTURE (Input: {s}×{s})
    
    ENCODER (Contracting)              DECODER (Expanding)
    ─────────────────────              ───────────────────
    
    [3, {s}, {s}] Input
         │
         ▼
    ┌─────────────────┐                ┌─────────────────┐
    │  inc (64 ch)    │───────────────►│  up4_conv       │
    │  {s}×{s}          │    skip 1      │  64 ch, {s}×{s}   │
    └────────┬────────┘                └────────▲────────┘
             │ pool                             │ up
             ▼                                  │
    ┌─────────────────┐                ┌─────────────────┐
    │  down1 (128 ch) │───────────────►│  up3_conv       │
    │  {s//2}×{s//2}          │    skip 2      │  128 ch, {s//2}×{s//2}  │
    └────────┬────────┘                └────────▲────────┘
             │ pool                             │ up
             ▼                                  │
    ┌─────────────────┐                ┌─────────────────┐
    │  down2 (256 ch) │───────────────►│  up2_conv       │
    │  {s//4}×{s//4}          │    skip 3      │  256 ch, {s//4}×{s//4}  │
    └────────┬────────┘                └────────▲────────┘
             │ pool                             │ up
             ▼                                  │
    ┌─────────────────┐                ┌─────────────────┐
    │  down3 (512 ch) │───────────────►│  up1_conv       │
    │  {s//8}×{s//8}          │    skip 4      │  512 ch, {s//8}×{s//8}  │
    └────────┬────────┘                └────────▲────────┘
             │ pool                             │ up
             ▼                                  │
         ┌───────────────────────────────────┐
         │        BRIDGE (Bottleneck)         │
         │        1024 channels, {s//16}×{s//16}         │
         │   Most compressed representation   │
         └───────────────────────────────────┘

    Output: [1, 1, {s}, {s}] (Segmentation Mask)
    """
    print(diagram)


def trace_unet(input_size=128, downsample_mode='mp', upsample_mode='tr', batch_size=1):
    """
    Trace tensor shapes through the entire U-Net architecture.
    
    This is the main function that walks through every layer and shows
    exactly how the tensor shape changes at each step.
    """
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + f"{'U-NET TENSOR SHAPE TRACER':^68}" + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print(f"\n📋 Configuration:")
    print(f"   • Input Size: {input_size}×{input_size}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Downsample Mode: {downsample_mode} ({'MaxPool2d' if downsample_mode == 'mp' else 'Strided Conv'})")
    print(f"   • Upsample Mode: {upsample_mode} ({'ConvTranspose2d' if upsample_mode == 'tr' else 'Upsample+Conv'})")
    
    # Print architecture diagram first
    print_architecture_diagram(input_size)
    
    # =========================================================================
    # BUILD THE MODEL COMPONENTS
    # =========================================================================
    
    class DoubleConv(nn.Module):
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
    
    def make_down(in_c):
        if downsample_mode == 'mp':
            return nn.MaxPool2d(2)
        else:
            return nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1)
    
    def make_up(in_c, out_c):
        if upsample_mode == 'tr':
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            )
    
    # Build all layers
    inc = DoubleConv(3, 64)
    down1_pool = make_down(64)
    down1_conv = DoubleConv(64, 128)
    down2_pool = make_down(128)
    down2_conv = DoubleConv(128, 256)
    down3_pool = make_down(256)
    down3_conv = DoubleConv(256, 512)
    bridge_pool = make_down(512)
    bridge_conv = DoubleConv(512, 1024)
    
    up1 = make_up(1024, 512)
    up1_conv = DoubleConv(1024, 512)
    up2 = make_up(512, 256)
    up2_conv = DoubleConv(512, 256)
    up3 = make_up(256, 128)
    up3_conv = DoubleConv(256, 128)
    up4 = make_up(128, 64)
    up4_conv = DoubleConv(128, 64)
    
    outc = nn.Conv2d(64, 1, kernel_size=1)
    
    # =========================================================================
    # TRACE THROUGH THE NETWORK
    # =========================================================================
    
    # Create input tensor
    x = torch.randn(batch_size, 3, input_size, input_size)
    
    print_section_header("INPUT", "📥")
    print(f"\nInput tensor shape: {Colors.GREEN}{format_shape(x)}{Colors.END}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Channels: 3 (RGB)")
    print(f"   • Height: {input_size}")
    print(f"   • Width: {input_size}")
    print(f"   • Total elements: {x.numel():,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENCODER
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("ENCODER (Contracting Path)", "⬇️")
    print("\nThe encoder COMPRESSES spatial information into more channels.")
    print("Think of it as: 'What is in this image?' (losing WHERE details)")
    
    # Initial convolution
    x1 = trace_conv_block(x, inc, "inc: Initial Convolution (3 → 64 channels)")
    
    # Down 1
    x = trace_downsample(x1, down1_pool, "down1_pool", downsample_mode)
    x2 = trace_conv_block(x, down1_conv, "down1_conv: (64 → 128 channels)")
    
    # Down 2
    x = trace_downsample(x2, down2_pool, "down2_pool", downsample_mode)
    x3 = trace_conv_block(x, down2_conv, "down2_conv: (128 → 256 channels)")
    
    # Down 3
    x = trace_downsample(x3, down3_pool, "down3_pool", downsample_mode)
    x4 = trace_conv_block(x, down3_conv, "down3_conv: (256 → 512 channels)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # BRIDGE
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("BRIDGE (Bottleneck)", "🌉")
    print("\nThe bridge is the MOST COMPRESSED representation.")
    print("Maximum semantic understanding, minimum spatial detail.")
    
    x = trace_downsample(x4, bridge_pool, "bridge_pool", downsample_mode)
    x5 = trace_conv_block(x, bridge_conv, "bridge_conv: (512 → 1024 channels)")
    
    bridge_spatial = x5.shape[2]
    print(f"\n{Colors.YELLOW}{'─'*50}")
    print(f"🎯 BOTTLENECK REACHED!")
    print(f"   Original: {input_size}×{input_size} pixels, 3 channels")
    print(f"   Now:      {bridge_spatial}×{bridge_spatial} pixels, 1024 channels")
    print(f"   Compression ratio: {(input_size/bridge_spatial)**2:.0f}x spatial, {1024/3:.0f}x channels")
    print(f"{'─'*50}{Colors.END}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # DECODER
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("DECODER (Expanding Path)", "⬆️")
    print("\nThe decoder EXPANDS back to original resolution.")
    print("Think of it as: 'WHERE exactly is each thing?' (using skip connections)")
    
    # Up 1
    x = trace_upsample(x5, up1, "up1", upsample_mode)
    x = trace_skip_connection(x, x4, "skip4")
    x = trace_conv_block(x, up1_conv, "up1_conv: (1024 → 512 channels)")
    
    # Up 2
    x = trace_upsample(x, up2, "up2", upsample_mode)
    x = trace_skip_connection(x, x3, "skip3")
    x = trace_conv_block(x, up2_conv, "up2_conv: (512 → 256 channels)")
    
    # Up 3
    x = trace_upsample(x, up3, "up3", upsample_mode)
    x = trace_skip_connection(x, x2, "skip2")
    x = trace_conv_block(x, up3_conv, "up3_conv: (256 → 128 channels)")
    
    # Up 4
    x = trace_upsample(x, up4, "up4", upsample_mode)
    x = trace_skip_connection(x, x1, "skip1")
    x = trace_conv_block(x, up4_conv, "up4_conv: (128 → 64 channels)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("OUTPUT", "📤")
    
    before = x
    x = outc(x)
    
    print(f"\n{Colors.BOLD}Final 1×1 Convolution (64 → 1 channel){Colors.END}")
    print(f"   {Colors.CYAN}{format_shape(before)} → {format_shape(x)}{Colors.END}")
    print(f"   params: {count_parameters(outc):,}")
    
    print(f"\n{Colors.GREEN}{'─'*50}")
    print(f"✅ OUTPUT TENSOR: {format_shape(x)}")
    print(f"   • Batch: {x.shape[0]}")
    print(f"   • Channels: {x.shape[1]} (1 = binary segmentation)")
    print(f"   • Height: {x.shape[2]} (same as input!)")
    print(f"   • Width: {x.shape[3]} (same as input!)")
    print(f"{'─'*50}{Colors.END}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("SHAPE JOURNEY SUMMARY", "📊")
    
    print(f"""
    INPUT  → ENCODER → BRIDGE → DECODER → OUTPUT
    
    Spatial Size Journey:
    {input_size}×{input_size} → {input_size//2}×{input_size//2} → {input_size//4}×{input_size//4} → {input_size//8}×{input_size//8} → {input_size//16}×{input_size//16} → {input_size//8}×{input_size//8} → {input_size//4}×{input_size//4} → {input_size//2}×{input_size//2} → {input_size}×{input_size}
    
    Channel Journey:
    3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1
    
    Skip Connection Summary:
    ┌────────────────┬─────────────────┬─────────────────────────────────┐
    │ Encoder Layer  │ Shape           │ Connected To                    │
    ├────────────────┼─────────────────┼─────────────────────────────────┤
    │ x1 (inc)       │ [64, {input_size:3d}, {input_size:3d}]  │ up4 (restores fine edges)       │
    │ x2 (down1)     │ [128, {input_size//2:3d}, {input_size//2:3d}] │ up3 (restores textures)         │
    │ x3 (down2)     │ [256, {input_size//4:3d}, {input_size//4:3d}]  │ up2 (restores parts)            │
    │ x4 (down3)     │ [512, {input_size//8:3d}, {input_size//8:3d}]  │ up1 (restores object shapes)    │
    └────────────────┴─────────────────┴─────────────────────────────────┘
    """)
    
    # Count total parameters
    total_params = (
        count_parameters(inc) + count_parameters(down1_conv) + 
        count_parameters(down2_conv) + count_parameters(down3_conv) +
        count_parameters(bridge_conv) +
        count_parameters(up1) + count_parameters(up1_conv) +
        count_parameters(up2) + count_parameters(up2_conv) +
        count_parameters(up3) + count_parameters(up3_conv) +
        count_parameters(up4) + count_parameters(up4_conv) +
        count_parameters(outc)
    )
    
    # Add pooling params if strided conv
    if downsample_mode == 'str_conv':
        total_params += (
            count_parameters(down1_pool) + count_parameters(down2_pool) +
            count_parameters(down3_pool) + count_parameters(bridge_pool)
        )
    
    print(f"\n{Colors.BOLD}Total Trainable Parameters: {total_params:,}{Colors.END}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # KEY INSIGHTS
    # ─────────────────────────────────────────────────────────────────────────
    
    print_section_header("KEY INSIGHTS", "💡")
    
    print("""
    1️⃣  SPATIAL vs CHANNELS Trade-off:
        • As spatial size ↓ (pooling), channels ↑ (more features)
        • As spatial size ↑ (upsampling), channels ↓ (fewer features)
        • Total information is roughly preserved!
    
    2️⃣  Why Skip Connections?
        • Encoder loses spatial precision (WHERE things are)
        • Skip connections restore this lost spatial information
        • Without them: blurry, imprecise segmentation boundaries
    
    3️⃣  The Concatenation Trick:
        • Skip connections CONCATENATE, not add
        • This doubles channels temporarily (e.g., 512 + 512 = 1024)
        • The following conv block reduces channels back (1024 → 512)
    
    4️⃣  Bottleneck = Semantic Understanding:
        • Smallest spatial size = most abstract representation
        • 1024 channels encode "what" is in the image
        • Decoder uses this to guide "where" to segment
    
    5️⃣  Output is Same Size as Input:
        • U-Net is fully convolutional - works on any input size!
        • (As long as it's divisible by 16 for 4 pooling layers)
    """)
    
    # Test different input sizes
    print_section_header("TRY DIFFERENT INPUT SIZES", "🧪")
    
    print("\n    Run with different sizes to see how shapes change:")
    print(f"    python {__file__} --size 64")
    print(f"    python {__file__} --size 128")
    print(f"    python {__file__} --size 256")
    print(f"    python {__file__} --size 512")
    print("\n    Minimum size: 16 (for 4 pooling layers)")
    print("    Size must be divisible by 16!")


def main():
    parser = argparse.ArgumentParser(
        description='U-Net Tensor Shape Tracer - Visualize how tensor shapes change through the network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unet_shape_tracer.py                     # Default 128x128 input
  python unet_shape_tracer.py --size 256          # 256x256 input
  python unet_shape_tracer.py --size 512          # 512x512 input  
  python unet_shape_tracer.py --down str_conv     # Use strided convolutions
  python unet_shape_tracer.py --up ups            # Use Upsample instead of TransposeConv
  python unet_shape_tracer.py --batch 4           # Batch size of 4
        """
    )
    
    parser.add_argument('--size', type=int, default=128,
                       help='Input image size (default: 128). Must be divisible by 16.')
    parser.add_argument('--batch', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--down', type=str, default='mp', choices=['mp', 'str_conv'],
                       help='Downsampling: mp (MaxPool) or str_conv (Strided Conv)')
    parser.add_argument('--up', type=str, default='tr', choices=['tr', 'ups'],
                       help='Upsampling: tr (TransposeConv) or ups (Upsample+Conv)')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Validate input size
    if args.size % 16 != 0:
        print(f"⚠️  Warning: Input size {args.size} is not divisible by 16.")
        print(f"   This may cause shape mismatches. Recommended: 64, 128, 256, 512...")
    
    if args.size < 16:
        print(f"❌ Error: Input size must be at least 16 (got {args.size})")
        return
    
    if args.no_color:
        Colors.disable()
    
    # Run the tracer
    trace_unet(
        input_size=args.size,
        downsample_mode=args.down,
        upsample_mode=args.up,
        batch_size=args.batch
    )
    
    print("\n" + "█"*70)
    print("█" + f"{'🎓 Now you understand tensor shapes in U-Net!':^68}" + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()