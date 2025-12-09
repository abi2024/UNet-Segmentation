import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), # Added BatchNorm for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), # Added BatchNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, downsample_mode='mp', upsample_mode='tr'):
        super().__init__()
        self.down_mode = downsample_mode
        self.up_mode = upsample_mode

        # --- ENCODER LAYERS ---
        self.inc = DoubleConv(3, 64)
        
        # Down 1
        self.down1_pool = self._make_down_layer(64)
        self.down1_conv = DoubleConv(64, 128)
        
        # Down 2
        self.down2_pool = self._make_down_layer(128)
        self.down2_conv = DoubleConv(128, 256)
        
        # Down 3
        self.down3_pool = self._make_down_layer(256)
        self.down3_conv = DoubleConv(256, 512)
        
        # Bridge (Down 4)
        self.bridge_pool = self._make_down_layer(512)
        self.bridge_conv = DoubleConv(512, 1024)
        
        # --- DECODER LAYERS ---
        self.up1 = self._make_up_layer(1024, 512)
        self.up1_conv = DoubleConv(1024, 512)
        
        self.up2 = self._make_up_layer(512, 256)
        self.up2_conv = DoubleConv(512, 256)
        
        self.up3 = self._make_up_layer(256, 128)
        self.up3_conv = DoubleConv(256, 128)
        
        self.up4 = self._make_up_layer(128, 64)
        self.up4_conv = DoubleConv(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def _make_down_layer(self, in_c):
        """Helper to create the correct downsampling layer"""
        if self.down_mode == 'mp':
            return nn.MaxPool2d(2)
        elif self.down_mode == 'str_conv':
            # Strided Conv to halve the size
            return nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1)

    def _make_up_layer(self, in_c, out_c):
        """Helper to create the correct upsampling layer"""
        if self.up_mode == 'tr':
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        elif self.up_mode == 'ups':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            )

    def forward(self, x):
        # --- ENCODER ---
        x1 = self.inc(x) # 64 channels
        
        x2_in = self.down1_pool(x1)
        x2 = self.down1_conv(x2_in) # 128 channels

        x3_in = self.down2_pool(x2)
        x3 = self.down2_conv(x3_in) # 256 channels

        x4_in = self.down3_pool(x3)
        x4 = self.down3_conv(x4_in) # 512 channels
        
        # --- BRIDGE ---
        bot_in = self.bridge_pool(x4)
        x5 = self.bridge_conv(bot_in) # 1024 channels

        # --- DECODER ---
        # Up 1
        x = self.up1(x5) # 1024 -> 512
        # If shapes don't perfectly match (due to rounding), resize x to match x4
        if x.shape != x4.shape:
            x = nn.functional.interpolate(x, size=x4.shape[2:])
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)
        
        # Up 2
        x = self.up2(x) # 512 -> 256
        if x.shape != x3.shape: x = nn.functional.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        # Up 3
        x = self.up3(x) # 256 -> 128
        if x.shape != x2.shape: x = nn.functional.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        # Up 4
        x = self.up4(x) # 128 -> 64
        if x.shape != x1.shape: x = nn.functional.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)
        
        # Logits output (No Sigmoid)
        return self.outc(x)