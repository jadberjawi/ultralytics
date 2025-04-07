import torch
import torch.nn as nn
import torch.nn.functional as F

# Redundant Spectrum Removal (RSR) Module
class RSR(nn.Module):
    def __init__(self, topk=768):
        super(RSR, self).__init__()
        self.topk = topk
        self.rgb_embed = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),  # now 3 in → 3 out
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.ir_embed = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # 1 in → 1 out
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
    def forward(self, x):
        if isinstance(x, dict):
            x = x["img"]  # Extract image batch from YOLO's dataset dictionary

        # Ensure batch dimension is present
        assert x.ndim == 4, f"Expected 4D input (B, C, H, W), got {x.shape}"

        # Debugging: Print input batch shape
        print(f"🔍 MEFA Input Shape: {x.shape}")
        
        # Split RGB and IR channels
        rgb, ir = x[:, :3, :, :], x[:, 3:, :, :]

        rgb_freq = torch.fft.fft2(rgb, norm='ortho')
        ir_freq = torch.fft.fft2(ir, norm='ortho')

        rgb_feat = self.rgb_embed(rgb)
        ir_feat = self.ir_embed(ir)

        B, C, H, W = rgb_feat.shape
        D = C * H * W
        topk = min(self.topk, D)

        rgb_feat_flat = rgb_feat.view(B, -1)
        ir_feat_flat = ir_feat.view(B, -1)

        _, rgb_indices = torch.topk(rgb_feat_flat, topk, dim=1)
        _, ir_indices = torch.topk(ir_feat_flat, topk, dim=1)

        def create_mask(indices, total_size):
            mask = torch.zeros((indices.size(0), total_size), device=indices.device)
            for b in range(indices.size(0)):
                mask[b, indices[b]] = 1.0
            return mask.view(B, C, H, W)

        rgb_mask = create_mask(rgb_indices, D)
        ir_mask = create_mask(ir_indices, D)

        # Upsample to match frequency size
        _, _, Hf, Wf = rgb.shape
        rgb_mask_up = F.interpolate(rgb_mask, size=(Hf, Wf), mode='bilinear', align_corners=False)
        ir_mask_up = F.interpolate(ir_mask, size=(Hf, Wf), mode='bilinear', align_corners=False)

        rgb_freq_filtered = rgb_freq * rgb_mask_up
        ir_freq_filtered = ir_freq * ir_mask_up

        rgb_filtered = torch.fft.ifft2(rgb_freq_filtered, norm='ortho').real
        ir_filtered = torch.fft.ifft2(ir_freq_filtered, norm='ortho').real

        return torch.cat([rgb_filtered, ir_filtered], dim=1)

