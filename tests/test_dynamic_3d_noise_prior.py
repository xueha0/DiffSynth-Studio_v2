import torch
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from examples.wanvideo.model_training.train import (
        DynamicViewActionScene3DNoisePriorAdapter,
        normalize_noise_like,
    )
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Skipping dynamic 3D prior smoke test because a training dependency is missing: {exc}",
        allow_module_level=True,
    )


def test_dynamic_view_action_prior_shape_and_stats():
    adapter = DynamicViewActionScene3DNoisePriorAdapter(
        scene_dim=32,
        condition_dim=7,
        latent_channels=4,
        hidden_dim=16,
        num_heads=4,
        max_views=4,
    )
    scene_tokens = torch.randn(1, 8, 32)
    condition_sequence = torch.randn(1, 3, 7)

    noise = adapter(
        scene_tokens,
        (1, 4, 3, 5, 6),
        condition_sequence,
        source_view_ids=(0, 1),
        target_view_id=2,
    )

    assert noise.shape == (1, 4, 3, 5, 6)
    assert torch.isfinite(noise).all()
    assert torch.allclose(noise.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(noise.std(), torch.tensor(1.0), atol=1e-5)


def test_normalize_noise_like_keeps_shape():
    noise = torch.randn(2, 4, 3, 5, 6)
    normalized = normalize_noise_like(noise)

    assert normalized.shape == noise.shape
    assert torch.allclose(
        normalized.mean(dim=(1, 2, 3, 4)),
        torch.zeros(2),
        atol=1e-5,
    )
