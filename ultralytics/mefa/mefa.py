import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Prevent runtime error in Google Colab
import matplotlib.pyplot as plt
import numpy as np



# Inception Block A
class InceptionBlockA(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(InceptionBlockA, self).__init__()

        # Ensure `out_channels` is divisible across branches
        assert out_channels % 4 == 0, "out_channels should be divisible by 4 for balance"

        branch_out = out_channels // 4  # Each branch contributes 1/4th of `out_channels`

        # 1x1 Convolution Branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 3x3 Convolution Branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_out, branch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 5x5 Convolution Branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out // 2, kernel_size=1),  # Reduce channels before 5x5
            nn.BatchNorm2d(branch_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_out // 2, branch_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 3x3 MaxPooling Branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        
        return torch.cat([b1, b2, b3, b4], dim=1)  # Concatenate along the channel dimension

class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(LocalAttention, self).__init__()

        # Apply 3 convolution layers to generate attention map
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Compute local attention weights
        attn = self.conv2(F.relu(self.bn2(self.conv1(x))))

        return attn

class GlobalAttention(nn.Module):
    def __init__(self, in_channels, num_partitions=(5, 10)):  # 5 rows, 10 columns
        super(GlobalAttention, self).__init__()
        self.num_partitions = num_partitions  # Defines partitioning of the feature map

        # Global attention feature extractor
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_part, w_part = self.num_partitions

        # 1. Partition-wise Average Pooling (downsampling to partition size)
        x_pooled = F.adaptive_avg_pool2d(x, (h_part, w_part))  # Shape: (B, C, 5, 10)

        # 2. Generate global attention scores
        attn = self.conv2(F.relu(self.bn1(self.conv1(x_pooled))))  # Compute attention scores

        # 3. Return the pooled attention map (before softmax)
        return attn

def load_npy_image(npy_path):
    npy_image = np.load(npy_path)  # Shape: (H, W, 4)
    if npy_image.shape[-1] != 4:
        raise ValueError(f"Expected 4-channel image (RGB + IR), but got shape {npy_image.shape}")

    # Extract RGB and IR
    rgb = npy_image[..., :3]  
    ir = npy_image[..., 3:]  

    # Resize to 640×640
    from torchvision.transforms.functional import resize
    rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    ir_tensor = torch.tensor(ir, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)

    rgb_tensor = F.interpolate(rgb_tensor, size=(640, 640), mode="bilinear", align_corners=False)
    ir_tensor = F.interpolate(ir_tensor, size=(640, 640), mode="bilinear", align_corners=False)

    return rgb_tensor, ir_tensor
# Load and Process Image
input_image_path = "test.npy"

input_rgb, input_ir = load_npy_image(input_image_path) 

print("Loaded RGB Shape:", input_rgb.shape)  # Expected: (1, 3, 640, 640)
print("Loaded IR Shape:", input_ir.shape)    # Expected: (1, 1, 640, 640)

#  1️⃣ Inception Block Section
inception_block = InceptionBlockA(in_channels=3)
inception_block_1c = InceptionBlockA(in_channels=1)

# Forward Pass through Inception Blocks
in_rgb = inception_block(input_rgb)
in_ir = inception_block_1c(input_ir)


# 2️⃣ Local Attention Section
local_attention = LocalAttention(in_channels=128)

# Apply Local Attention
la_rgb = local_attention(in_rgb)
la_ir = local_attention(in_ir)

# Apply Softmax Across Modalities
attn = torch.softmax(torch.stack([la_rgb, la_ir], dim=1), dim=1)  # Normalize
la_attn_rgb, la_attn_ir = attn.unbind(dim=1)  # Separate maps

# 3️⃣ Element-wise multiplication Section 1
mul_la_in_rgb = in_rgb * la_attn_rgb
mul_la_in_ir = in_ir * la_attn_ir


# 4️⃣ Concatenate the multiplied values of RGB and IR
fused_features_1 = torch.cat([mul_la_in_rgb, mul_la_in_ir], dim=1)  # Concatenate along channels


# 5️⃣ Global Attention Section
global_attention = GlobalAttention(in_channels=128)

# Apply Global Attention to Each Modality Before Local Attention
ga_rgb = global_attention(in_rgb)
ga_ir = global_attention(in_ir)

# Upscale Attention Maps to Original Feature Map Size
g_attn_rgb = F.interpolate(ga_rgb, size=(640, 640), mode="bilinear", align_corners=False)
g_attn_ir = F.interpolate(ga_ir, size=(640, 640), mode="bilinear", align_corners=False)

ga_test = ga_rgb + ga_ir
# # Apply Softmax across the modality dimension (so sum = 1 for each region)
# attn = torch.softmax(torch.stack([ga_rgb, ga_ir], dim=1), dim=1)  # Normalize across modalities
# attn_rgb, attn_ir = attn.unbind(dim=1)  # Separate the attention maps

# # Upscale Attention Maps to Original Feature Map Size
# g_attn_rgb = F.interpolate(attn_rgb, size=(640, 640), mode="bilinear", align_corners=False)
# g_attn_ir = F.interpolate(attn_ir, size=(640, 640), mode="bilinear", align_corners=False)


# 6️⃣ Last Inception Block Section
inception_block_2 = InceptionBlockA(in_channels=256)

# Pass the fused feature map through the Inception Block
in_fused = inception_block_2(fused_features_1)


# 7️⃣ Element-wise multiplication Section 2
mul_ga_in_fused_rgb = g_attn_rgb * in_fused
mul_ga_in_fused_ir = g_attn_ir * in_fused


# 8️⃣ Concatenate the multiplied values of RGB and IR
fused_features_2 = torch.cat([mul_ga_in_fused_rgb, mul_ga_in_fused_ir], dim=1)

# Final Convolution to Reduce Channels to 3
final_conv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)  # Reduce channels to 3

# Apply the final layer
final_output = final_conv(fused_features_2)

# Print new shape to verify
print("Final Output Shape:", final_output.shape)  # Expected: [1, 3, 640, 640]


# Convert Processed Image for Visualization
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).mean(dim=0)  # Convert (1, C, H, W) → (H, W)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize
    return tensor.cpu().detach().numpy()

# Function to visualize feature maps
def visualize_feature_map(tensor, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor_to_image(tensor), cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.show()
    # Save the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {filename}")  # Print confirmation message
    
    plt.close()  # Close the figure to prevent memory issues
    

# Step 1: Print Input Shape
print("Step 1 - Input Shape (RGB):", input_rgb.shape)  # Expected: (1, 3, 640, 640)
print("Step 1 - Input Shape (IR):", input_ir.shape)  # Expected: (1, 3, 640, 640)

# Step 2: After Inception Block A
print("Step 2 - Inception Block Output Shape (RGB):", in_rgb.shape)  # Expected: (1, 128, 640, 640)
print("Step 2 - Inception Block Output Shape (IR):", in_ir.shape)  # Expected: (1, 128, 640, 640)
visualize_feature_map(in_rgb, "Inception Output - RGB", "inception_output_rgb.png")
visualize_feature_map(in_ir, "Inception Output - IR", "inception_output_ir.png")

# Step 3: After Local Attention
print("Step 3 - Local Attention Output Shape (RGB):", la_attn_rgb.shape)  # Expected: (1, 128, 640, 640)
print("Step 3 - Local Attention Output Shape (IR):", la_attn_ir.shape)  # Expected: (1, 128, 640, 640)
visualize_feature_map(la_attn_rgb, "Local Attention - RGB", "local_attention_output_rgb.png")
visualize_feature_map(la_attn_ir, "Local Attention - IR", "local_attention_output_ir.png")

# Step 4: After Element-Wise Multiplication
print("Step 4 - Refined Feature Shape (RGB):", mul_la_in_rgb.shape)  # Expected: (1, 128, 640, 640)
print("Step 4 - Refined Feature Shape (IR):", mul_la_in_ir.shape)  # Expected: (1, 128, 640, 640)
visualize_feature_map(mul_la_in_rgb, "Refined Features - RGB", "refined_features_rgb.png")
visualize_feature_map(mul_la_in_ir, "Refined Features - IR", "refined_features_ir.png")

# Step 5: After Concatenation of RGB and IR features
print("Step 5 - Fused Feature Shape:", fused_features_1.shape)  # Expected: (1, 256, 640, 640)
visualize_feature_map(fused_features_1, "Fused Features (RGB + IR)", "fused_features_1.png")

# Step 6: After Global Attention
print("Step 6 - Global Attention Output Shape (RGB):", g_attn_rgb.shape)  # Expected: (1, 128, 640, 640)
print("Step 6 - Global Attention Output Shape (IR):", g_attn_ir.shape)  # Expected: (1, 128, 640, 640)
visualize_feature_map(g_attn_rgb, "Global Attention - RGB", "global_attention_output_rgb.png")
visualize_feature_map(g_attn_ir, "Global Attention - IR", "global_attention_output_ir.png")

# Step 7: After Passing Fused Features through Inception Block A
print("Step 7 - Inception Output After Fusion Shape:", in_fused.shape)  # Expected: (1, 128, 640, 640)
visualize_feature_map(in_fused, "Inception After Fusion", "inception_after_fusion.png")

# Step 8: After Multiplication with Global Attention
print("Step 8 - Global Attention * Inception After Fusion (RGB):", mul_ga_in_fused_rgb.shape)
print("Step 8 - Global Attention * Inception After Fusion (IR):", mul_ga_in_fused_ir.shape)
visualize_feature_map(mul_ga_in_fused_rgb, "Global Attention * Inception After Fusion - RGB", "global_attention_rgb_inception_after_fusion_multiplication.png")
visualize_feature_map(mul_ga_in_fused_ir, "Global Attention * Inception After Fusion - IR", "global_attention_ir_inception_after_fusion_multiplication.png")

# Step 9: fused features Output After Concatenation
print("Step 9 - Final Output Shape:", fused_features_2.shape)  # Expected: (1, 256, 640, 640)
visualize_feature_map(fused_features_2, "Final Output Feature Map", "fused_features_2.png")

# Step 10: Final Output After Convolution
print("Step 10 - Final Output  Shape:", final_output.shape)  # Expected: (1, 3, 640, 640)
visualize_feature_map(final_output, "Final Output ", "final_output.png")

print("Test Ga")
visualize_feature_map(ga_test, "Test Output ", "test.png")
