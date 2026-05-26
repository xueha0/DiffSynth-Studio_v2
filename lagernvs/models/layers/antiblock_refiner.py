import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiBlockRefiner(nn.Module):
    """Local residual image refiner used to soften patch seams."""

    def __init__(
        self,
        hidden_size,
        patch_size,
        token_dim=96,
        hidden_dim=64,
        residual_scale=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.token_proj = nn.Conv2d(hidden_size, token_dim, kernel_size=1, bias=True)
        mid_dim = max(32, hidden_dim // 2)
        self.refiner = nn.Sequential(
            nn.Conv2d(token_dim + 3 + 6, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, 3, kernel_size=3, padding=1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        final_conv = self.refiner[-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(self, coarse_rgb, token_grid, target_rays):
        """Refine coarse RGB using local token features and corrected target rays."""
        token_features = self.token_proj(token_grid)
        token_features = F.interpolate(
            token_features,
            size=coarse_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refine_input = torch.cat([coarse_rgb, token_features, target_rays], dim=1)
        delta = self.refiner(refine_input)
        refined_rgb = torch.clamp(
            coarse_rgb + self.residual_scale * torch.tanh(delta),
            min=0.0,
            max=1.0,
        )
        aux_outputs = {
            "refiner_delta_abs_mean": delta.abs().mean().detach(),
        }
        return refined_rgb, aux_outputs
