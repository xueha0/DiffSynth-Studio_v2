import torch
import torch.nn as nn
from einops import rearrange


class CrossViewSourceVideoProjector3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 512,
        out_channels: int = 1536,
        max_source_views: int = 4,
        max_time: int = 64,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.max_source_views = max_source_views
        self.max_time = max_time
        self.projector = nn.Sequential(
            nn.Conv3d(
                in_channels,
                hidden_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.GELU(),
            nn.Conv3d(
                hidden_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
        )
        self.view_embedding = nn.Parameter(torch.zeros(max_source_views, out_channels))
        self.time_embedding = nn.Parameter(torch.zeros(max_time, out_channels))
        nn.init.normal_(self.view_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, source_latents: torch.Tensor) -> torch.Tensor:
        if source_latents.ndim == 5:
            source_latents = source_latents.unsqueeze(0)
        if source_latents.ndim != 6:
            raise ValueError(
                "Expected source_latents with shape (V,C,T,H,W) or (B,V,C,T,H,W)."
            )

        batch_size, num_views, _, _, _, _ = source_latents.shape
        if num_views > self.max_source_views:
            raise ValueError(
                f"num_views={num_views} exceeds max_source_views={self.max_source_views}"
            )

        x = rearrange(source_latents, "b v c t h w -> (b v) c t h w")
        x = self.projector(x)
        _, channels, num_frames, _, _ = x.shape
        if num_frames > self.max_time:
            raise ValueError(
                f"num_frames={num_frames} exceeds max_time={self.max_time}"
            )

        x = rearrange(x, "(b v) c t h w -> b v c t h w", b=batch_size, v=num_views)
        view_emb = self.view_embedding[:num_views].view(1, num_views, channels, 1, 1, 1)
        time_emb = self.time_embedding[:num_frames].transpose(0, 1).view(
            1, 1, channels, num_frames, 1, 1
        )
        x = x + view_emb + time_emb
        x = rearrange(x, "b v c t h w -> b (v t h w) c")
        return x


class CrossViewSourceVideoProjector3DTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 512,
        out_channels: int = 1536,
        max_source_views: int = 4,
        max_time: int = 64,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.max_source_views = max_source_views
        self.max_time = max_time
        self.projector = nn.Sequential(
            nn.Conv3d(
                in_channels,
                hidden_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.GELU(),
            nn.Conv3d(
                hidden_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
        )
        self.view_embedding = nn.Parameter(torch.zeros(max_source_views, out_channels))
        self.time_embedding = nn.Parameter(torch.zeros(max_time, out_channels))
        nn.init.normal_(self.view_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, source_latents: torch.Tensor) -> torch.Tensor:
        if source_latents.ndim == 5:
            source_latents = source_latents.unsqueeze(0)
        if source_latents.ndim != 6:
            raise ValueError(
                "Expected source_latents with shape (V,C,T,H,W) or (B,V,C,T,H,W)."
            )

        batch_size, num_views, _, _, _, _ = source_latents.shape
        if num_views > self.max_source_views:
            raise ValueError(
                f"num_views={num_views} exceeds max_source_views={self.max_source_views}"
            )

        x = rearrange(source_latents, "b v c t h w -> (b v) c t h w")
        x = self.projector(x)
        _, channels, num_frames, _, _ = x.shape
        if num_frames > self.max_time:
            raise ValueError(
                f"num_frames={num_frames} exceeds max_time={self.max_time}"
            )

        x = rearrange(x, "(b v) c t h w -> b v c t h w", b=batch_size, v=num_views)
        view_emb = self.view_embedding[:num_views].view(1, num_views, channels, 1, 1, 1)
        time_emb = self.time_embedding[:num_frames].transpose(0, 1).view(
            1, 1, channels, num_frames, 1, 1
        )
        x = x + view_emb + time_emb
        x = rearrange(x, "b v c t h w -> b t (v h w) c")
        return x
