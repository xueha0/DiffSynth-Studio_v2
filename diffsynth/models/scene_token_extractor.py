import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneTokenExtractor(nn.Module):
    """Frozen LagerNVS Reconstructor，提取 3D-aware scene tokens。

    适配 LagerNVS_v1：EncDec_VitB/8，hidden_size=768，patch_size=8。
    输入图像范围 [0, 1]（若从 DiffSynth 的 [-1, 1] 则会自动换算）。
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
        input_value_range: str = "minus1_1",
    ):
        super().__init__()
        from lagernvs.models.encoder_decoder import Reconstructor

        self.reconstructor = Reconstructor(
            renderer_hidden_size=768,
            target_patch_size=8,
            pretrained_vggt=False,
            freeze_vggt=freeze,
        )
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model" in state:
                state = state["model"]
            rec_state = {
                k.replace("reconstructor.", ""): v
                for k, v in state.items()
                if k.startswith("reconstructor.")
            }
            missing, unexpected = self.reconstructor.load_state_dict(
                rec_state, strict=False
            )
            if len(unexpected) > 0:
                print(f"[SceneTokenExtractor] unexpected keys: {unexpected[:3]}...")
            if len(missing) > 0:
                print(f"[SceneTokenExtractor] missing keys: {missing[:3]}...")

        self.input_value_range = input_value_range
        if freeze:
            self.reconstructor.eval()
            for p in self.reconstructor.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, cam_token: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            images: (B, V=2, 3, H, W) source view 首帧图像。
                    若 input_value_range='minus1_1'，会先转换为 [0,1]。
            cam_token: (B, V, 11) 相机参数（若 None，则全零）
        Returns:
            (B, V*P, 768) scene tokens
        """
        if self.input_value_range == "minus1_1":
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)

        B, V = images.shape[:2]
        if cam_token is None:
            cam_token = torch.zeros(
                B, V, 11, device=images.device, dtype=images.dtype
            )

        tokens = self.reconstructor(images, cam_token)
        B, V, P, C = tokens.shape
        return tokens.reshape(B, V * P, C)


class SceneTokenAdapter(nn.Module):
    """将 scene tokens 从 NVS 维度映射到 DiT 维度，可选空间池化。"""

    def __init__(self, in_dim: int = 768, out_dim: int = 1536, pool_size: int = 512):
        super().__init__()
        self.pool_size = pool_size
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: (B, N_raw, 768)
        Returns:
            (B, N_pooled, out_dim)
        """
        B, N, C = scene_tokens.shape
        if self.pool_size > 0 and N > self.pool_size:
            x = scene_tokens.transpose(1, 2)
            x = F.adaptive_avg_pool1d(x, self.pool_size)
            scene_tokens = x.transpose(1, 2)
        return self.proj(scene_tokens) + self.type_embedding
