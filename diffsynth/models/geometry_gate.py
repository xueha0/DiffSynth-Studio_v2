import torch
import torch.nn as nn


class TimestepAdaptiveGeometryGate(nn.Module):
    """
    根据去噪 timestep 动态调节 scene_tokens 和 source_memory 的权重。

    高噪声阶段（早期去噪）→ 全局 3D 结构重要 → gate_scene 偏高
    低噪声阶段（晚期去噪）→ 局部时序对齐重要 → gate_source 偏高
    """

    def __init__(self, dim: int = 1536, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, 2),
            )
            nn.init.normal_(self.gate_mlp[0].weight, std=0.01)
            nn.init.zeros_(self.gate_mlp[0].bias)
            nn.init.normal_(self.gate_mlp[2].weight, std=0.01)
            nn.init.zeros_(self.gate_mlp[2].bias)

    def forward(self, t_emb: torch.Tensor):
        """
        Args:
            t_emb: (B, dim) - DiT time_embedding 输出
        Returns:
            gate_scene: (B, 1, 1)
            gate_source: (B, 1, 1)
        """
        if self.mode == "learned":
            gates = torch.sigmoid(self.gate_mlp(t_emb))
            return gates[:, 0:1].unsqueeze(-1), gates[:, 1:2].unsqueeze(-1)
        elif self.mode == "constant":
            B = t_emb.shape[0]
            half = torch.full((B, 1, 1), 0.5, device=t_emb.device, dtype=t_emb.dtype)
            return half, half
        else:
            raise ValueError(f"Unknown gate mode: {self.mode}")
