import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image

# --- PATH SETUP ---
# We need to find your src folder to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet

# --- CONFIGURATION ---
# Point this to one of your saved models
MODEL_PATH = "../results/Exp2_MP_Tr_Dice_best.pth" 
# Point this to a specific image in your dataset
IMAGE_PATH = "../data/raw/Oxford-IIT-PetDataset/images/american_bulldog_2.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the architecture and weights."""
    # Ensure these parameters match your saved experiment (Exp 2 used 'mp' and 'tr')
    model = UNet(downsample_mode='mp', upsample_mode='tr')
    
    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Weights file not found. Using random initialization (Patterns will look like noise).")
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Replicates the preprocessing in your dataset.py manually."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    
    # HWC -> CHW -> BCHW (Batch, Channel, Height, Width)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(DEVICE), img

# --- THE MAGIC: FORWARD HOOKS ---
# This dictionary will store the output feature maps
activation = {}

def get_activation(name):
    """
    Creates a 'hook' function that saves the output of a specific layer
    into the activation dictionary.
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def visualize_features():
    model = load_model()
    input_tensor, original_img = preprocess_image(IMAGE_PATH)

    # 1. Register Hooks
    # We want to see what the model sees at different depths
    
    # Early Encoder (Edges/Textures)
    model.inc.register_forward_hook(get_activation('1_Input_Layer'))
    model.down1_conv.register_forward_hook(get_activation('2_Encoder_Layer1'))
    
    # Bottom of U-Net (Abstract Concepts / "The Blob")
    model.bridge_conv.register_forward_hook(get_activation('3_Bottleneck'))
    
    # Late Decoder (Reconstructing Shape)
    model.up3_conv.register_forward_hook(get_activation('4_Decoder_Layer3'))
    
    # Final Output
    model.outc.register_forward_hook(get_activation('5_Final_Logits'))

    # 2. Pass the image through the model
    print("🤖 Processing image...")
    with torch.no_grad():
        output = model(input_tensor)

    # 3. Visualization
    layers_to_plot = ['1_Input_Layer', '2_Encoder_Layer1', '3_Bottleneck', '4_Decoder_Layer3', '5_Final_Logits']
    
    plt.figure(figsize=(15, 12))
    
    # Show Original Image
    plt.subplot(len(layers_to_plot) + 1, 5, 3)
    plt.title("Original Input")
    plt.imshow(original_img)
    plt.axis('off')

    for i, layer_name in enumerate(layers_to_plot):
        feats = activation[layer_name].cpu().squeeze(0) # Remove batch dim: (C, H, W)
        
        # We will plot the first 5 filters (channels) of this layer
        # If it's the final layer, it only has 1 channel
        num_filters = 5 if feats.shape[0] > 5 else feats.shape[0]
        
        for j in range(num_filters):
            # Calculate subplot index
            # Row = i+1 (offset by original image row), Col = j+1
            ax = plt.subplot(len(layers_to_plot) + 1, 5, (i + 1) * 5 + (j + 1))
            
            # Grab the j-th feature map
            fmap = feats[j, :, :]
            
            ax.imshow(fmap, cmap='viridis') # 'viridis' or 'magma' makes 'hot' activations pop
            ax.axis('off')
            
            if j == 2: # Set title on middle column
                ax.set_title(f"{layer_name}\n({feats.shape[0]} filters, {feats.shape[1]}x{feats.shape[2]})")

    plt.tight_layout()
    plt.savefig("feature_maps_visualization.png")
    print("✅ Visualization saved to 'feature_maps_visualization.png'")
    plt.show()

if __name__ == "__main__":
    visualize_features()