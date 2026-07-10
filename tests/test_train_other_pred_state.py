import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

try:
    from diffsynth.models.wan_video_dit import WanModel
    from examples.wanvideo.model_training.train_other import (
        WanTrainingModule,
        wan_parser,
    )
    from safetensors.torch import save_file
except ModuleNotFoundError as import_error:
    WanModel = None
    WanTrainingModule = None
    wan_parser = None
    save_file = None
    TRAIN_IMPORT_ERROR = import_error
else:
    TRAIN_IMPORT_ERROR = None


class _SourceProjector(nn.Module):
    def forward(self, source_latents: torch.Tensor) -> torch.Tensor:
        batch_size = source_latents.shape[0]
        return torch.ones(batch_size, 3, 4, 8, dtype=source_latents.dtype)


class _FailingLegacyGate(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("temporal_local must bypass the legacy source gate")


@unittest.skipIf(
    WanTrainingModule is None,
    f"Training dependencies unavailable: {TRAIN_IMPORT_ERROR}",
)
class TrainOtherPredictedStateTest(unittest.TestCase):
    def make_module(self) -> WanTrainingModule:
        module = WanTrainingModule.__new__(WanTrainingModule)
        nn.Module.__init__(module)
        module.pipe = SimpleNamespace(
            device=torch.device("cpu"),
            torch_dtype=torch.float32,
        )
        return module

    def test_parser_accepts_predicted_state_options(self):
        args = wan_parser().parse_args(
            [
                "--dataset_base_path",
                "/tmp/data",
                "--state_loader",
                "predicted",
                "--cached_pred_state_root",
                "/tmp/predicted",
            ]
        )

        self.assertEqual(args.state_loader, "predicted")
        self.assertEqual(args.cached_pred_state_root, "/tmp/predicted")

    def test_cached_predicted_state_overwrites_state_and_action(self):
        module = self.make_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_file = root / "cache" / "train" / "sample_0001.pth"
            pred_dir = root / "predicted" / "train"
            pred_dir.mkdir(parents=True)
            predicted = np.zeros((3, 7), dtype=np.float32)
            predicted[-1] = 2.0
            np.save(pred_dir / "sample_0001.npy", predicted)
            module.cached_pred_state_root = str(root / "predicted")

            original = {
                "__cache_file__": str(cache_file),
                "num_frames": 5,
                "state": torch.full((1, 5, 7), -1.0),
            }
            updated = module.attach_cached_predicted_state(original)

        self.assertIsNot(updated, original)
        self.assertTrue(torch.equal(updated["state"], updated["action"]))
        self.assertEqual(tuple(updated["state"].shape), (1, 5, 7))
        self.assertTrue(torch.equal(updated["state"][:, -1], torch.ones(1, 7)))
        self.assertTrue(torch.equal(original["state"], torch.full((1, 5, 7), -1.0)))

    def test_temporal_local_uses_independent_memory_only(self):
        module = self.make_module()
        module.cross_view_source_injection_mode = "temporal_local"
        module.cross_view_source_views = (0, 1)
        module.cross_view_source_window_radius = 2
        module.pipe.source_video_projector = _SourceProjector()
        module.pipe.source_temporal_gate = _FailingLegacyGate()

        condition = module.build_cross_view_source_condition(
            condition_sequence=torch.randn(1, 3, 7),
            source_latents=torch.randn(2, 16, 3, 4, 4),
        )

        self.assertEqual(
            set(condition),
            {"source_memory_by_time", "source_window_radius"},
        )
        self.assertEqual(condition["source_window_radius"], 2)
        self.assertEqual(tuple(condition["source_memory_by_time"].shape), (1, 3, 4, 8))

    def test_temporal_local_trainables_include_dit_without_legacy_gate(self):
        module = self.make_module()
        module.cross_view_stage = 2
        module.cross_view_source_injection_mode = "temporal_local"
        module.cross_view_source_gate_mode = "state_aware"
        module.cross_view_state_loss_weight = 0.0
        module.geometry_target_camera_mode = "none"
        module.scene_token_checkpoint = None
        module.cross_view_3d_noise_prior_mode = "none"
        module.cross_view_3d_noise_prior_weight = 0.0

        trainables = set(
            module.extend_trainable_models(
                "source_temporal_gate,geometry_gates"
            ).split(",")
        )

        self.assertIn("dit", trainables)
        self.assertIn("source_video_projector", trainables)
        self.assertNotIn("source_temporal_gate", trainables)
        self.assertNotIn("geometry_gates", trainables)

    def test_legacy_checkpoint_initializes_independent_source_attention(self):
        model = WanModel(
            dim=32,
            in_dim=4,
            ffn_dim=64,
            out_dim=4,
            text_dim=16,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=4,
            num_layers=2,
            has_image_input=False,
        )
        model.enable_source_memory_attention()
        legacy_q = torch.full_like(model.blocks[0].cross_attn.q.weight, 0.25)
        loader = self.make_module()
        loader.pipe.dit = model
        for name in (
            "action_encoder",
            "source_video_projector",
            "source_temporal_gate",
            "target_state_head",
            "target_camera_encoder",
            "scene_token_adapter",
            "geometry_gates",
            "scene_3d_noise_prior_adapter",
            "action_noise_modulator",
        ):
            setattr(loader.pipe, name, None)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.safetensors"
            save_file(
                {"pipe.dit.blocks.0.cross_attn.q.weight": legacy_q},
                str(checkpoint_path),
            )
            loader.load_checkpoint_weights(str(checkpoint_path))

        self.assertTrue(
            torch.equal(model.blocks[0].source_cross_attn.q.weight, legacy_q)
        )
        self.assertEqual(
            torch.count_nonzero(model.blocks[0].source_cross_attn.o.weight).item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
