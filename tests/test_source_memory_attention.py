import copy
import tempfile
import unittest
from types import SimpleNamespace

import torch

import diffsynth.models.wan_video_dit as wan_video_dit
from diffsynth.models.wan_video_dit import DiTBlock, WanModel

try:
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    from examples.wanvideo.model_training.train import WanTrainingModule
    from safetensors.torch import save_file
except ModuleNotFoundError as import_error:
    model_fn_wan_video = None
    WanTrainingModule = None
    save_file = None
    PIPELINE_IMPORT_ERROR = import_error
else:
    PIPELINE_IMPORT_ERROR = None


def setUpModule():
    global ATTENTION_BACKEND_FLAGS
    ATTENTION_BACKEND_FLAGS = (
        wan_video_dit.FLASH_ATTN_3_AVAILABLE,
        wan_video_dit.FLASH_ATTN_2_AVAILABLE,
        wan_video_dit.SAGE_ATTN_AVAILABLE,
    )
    wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
    wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
    wan_video_dit.SAGE_ATTN_AVAILABLE = False


def tearDownModule():
    (
        wan_video_dit.FLASH_ATTN_3_AVAILABLE,
        wan_video_dit.FLASH_ATTN_2_AVAILABLE,
        wan_video_dit.SAGE_ATTN_AVAILABLE,
    ) = ATTENTION_BACKEND_FLAGS


def make_block() -> DiTBlock:
    torch.manual_seed(0)
    block = DiTBlock(
        has_image_input=False,
        dim=32,
        num_heads=4,
        ffn_dim=64,
    )
    return block


def make_block_inputs(batch_size: int = 2):
    return {
        "x": torch.randn(batch_size, 6, 32),
        "context": torch.randn(batch_size, 5, 32),
        "t_mod": torch.randn(batch_size, 6, 32),
        "freqs": torch.ones(6, 1, 4, dtype=torch.complex64),
        "source_memory_by_time": torch.randn(batch_size, 3, 4, 32),
        "timestep_emb": torch.randn(batch_size, 32),
    }


def make_model() -> WanModel:
    torch.manual_seed(0)
    return WanModel(
        dim=32,
        in_dim=4,
        ffn_dim=64,
        out_dim=4,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=3,
        has_image_input=False,
    )


def make_checkpoint_loader(model: WanModel):
    loader = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(loader)
    loader.pipe = SimpleNamespace(
        dit=model,
        torch_dtype=torch.float32,
        action_encoder=None,
        source_video_projector=None,
        source_temporal_gate=None,
        target_state_head=None,
        target_camera_encoder=None,
        scene_token_adapter=None,
        geometry_gates=None,
        scene_3d_noise_prior_adapter=None,
        action_noise_modulator=None,
    )
    return loader


def save_checkpoint(path: str, state_dict: dict[str, torch.Tensor]):
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in state_dict.items()
        },
        path,
    )


class SourceMemoryAttentionTest(unittest.TestCase):
    def test_source_modules_are_attached_after_base_construction(self):
        model = make_model()
        self.assertFalse(any(".source_" in key for key in model.state_dict()))

        model.enable_source_memory_attention()

        source_keys = {key for key in model.state_dict() if ".source_" in key}
        self.assertTrue(source_keys)
        for block in model.blocks:
            self.assertEqual(block.source_cross_attn.q.in_features, model.dim)
            self.assertEqual(block.source_cross_attn.q.out_features, model.dim)
            self.assertEqual(
                torch.count_nonzero(block.source_cross_attn.o.weight).item(),
                0,
            )
            self.assertTrue(
                torch.equal(
                    block.source_cross_attn.q.weight,
                    block.cross_attn.q.weight,
                )
            )
            gate = block.source_router(torch.randn(2, model.dim))
            self.assertTrue(torch.equal(gate, torch.full_like(gate, 0.5)))

    def test_zero_initialized_source_branch_preserves_output(self):
        block = make_block().eval()
        inputs = make_block_inputs()
        with torch.no_grad():
            baseline = block(
                inputs["x"],
                inputs["context"],
                inputs["t_mod"],
                inputs["freqs"],
            )

        block.enable_source_memory_attention()
        with torch.no_grad():
            no_memory = block(
                inputs["x"],
                inputs["context"],
                inputs["t_mod"],
                inputs["freqs"],
            )
            with_memory = block(
                inputs["x"],
                inputs["context"],
                inputs["t_mod"],
                inputs["freqs"],
                source_memory_by_time=inputs["source_memory_by_time"],
                source_window_radius=1,
                token_grid=(3, 2, 1),
                timestep_emb=inputs["timestep_emb"],
            )

        self.assertTrue(torch.equal(baseline, no_memory))
        self.assertTrue(torch.equal(baseline, with_memory))

    def test_router_and_source_attention_receive_gradients(self):
        block = make_block().train()
        block.enable_source_memory_attention()
        with torch.no_grad():
            block.source_cross_attn.o.weight.copy_(torch.eye(block.dim))

        inputs = make_block_inputs()
        source_memory = inputs["source_memory_by_time"].requires_grad_()
        output = block(
            inputs["x"],
            inputs["context"],
            inputs["t_mod"],
            inputs["freqs"],
            source_memory_by_time=source_memory,
            source_window_radius=1,
            token_grid=(3, 2, 1),
            timestep_emb=inputs["timestep_emb"],
        )
        output.square().mean().backward()

        self.assertIsNotNone(source_memory.grad)
        self.assertGreater(source_memory.grad.abs().sum().item(), 0)
        self.assertGreater(
            block.source_cross_attn.q.weight.grad.abs().sum().item(),
            0,
        )
        self.assertGreater(
            block.source_router.net[-1].weight.grad.abs().sum().item(),
            0,
        )

    def test_router_supports_global_and_per_frame_timesteps(self):
        block = make_block()
        block.enable_source_memory_attention()

        global_gate = block.source_router(torch.randn(2, 32))
        frame_gate = block.source_router(torch.randn(2, 3, 32))

        self.assertEqual(global_gate.shape, (2, 1))
        self.assertEqual(frame_gate.shape, (2, 3, 1))
        with torch.no_grad():
            block.source_router.net[1].weight.zero_()
            block.source_router.net[1].bias.zero_()
            block.source_router.net[1].weight[0, 0] = 1.0
            block.source_router.net[-1].weight.zero_()
            block.source_router.net[-1].bias.zero_()
            block.source_router.net[-1].weight[0, 0] = 1.0
        timestep_pair = torch.zeros(2, 32)
        timestep_pair[0, 0] = 1.0
        timestep_pair[1, 0] = -1.0
        routed_pair = block.source_router(timestep_pair)
        self.assertNotEqual(routed_pair[0].item(), routed_pair[1].item())
        with self.assertRaisesRegex(ValueError, "does not match token-grid length"):
            block.temporal_source_cross_attn(
                torch.randn(2, 6, 32),
                torch.randn(2, 3, 4, 32),
                source_window_radius=1,
                token_grid=(3, 2, 1),
                timestep_emb=torch.randn(2, 2, 32),
            )

    def test_legacy_and_new_checkpoint_initialization(self):
        legacy_model = make_model()
        legacy_state = copy.deepcopy(legacy_model.state_dict())
        legacy_model.enable_source_memory_attention()
        with torch.no_grad():
            legacy_model.blocks[0].cross_attn.q.weight.add_(1.0)
        legacy_state = {
            **legacy_state,
            "blocks.0.cross_attn.q.weight": copy.deepcopy(
                legacy_model.blocks[0].cross_attn.q.weight
            ),
        }

        legacy_model.load_state_dict(legacy_state, strict=False)
        legacy_model.initialize_source_memory_attention_from_base()
        self.assertTrue(
            torch.equal(
                legacy_model.blocks[0].source_cross_attn.q.weight,
                legacy_model.blocks[0].cross_attn.q.weight,
            )
        )
        self.assertEqual(
            torch.count_nonzero(
                legacy_model.blocks[0].source_cross_attn.o.weight
            ).item(),
            0,
        )

        with torch.no_grad():
            legacy_model.blocks[0].source_cross_attn.o.weight.fill_(0.25)
            legacy_model.blocks[0].source_router.net[-1].bias.fill_(1.0)
        new_state = copy.deepcopy(legacy_model.state_dict())
        restored_model = make_model()
        restored_model.enable_source_memory_attention()
        load_result = restored_model.load_state_dict(new_state, strict=False)

        self.assertFalse(load_result.missing_keys)
        self.assertFalse(load_result.unexpected_keys)
        self.assertTrue(
            torch.equal(
                restored_model.blocks[0].source_cross_attn.o.weight,
                legacy_model.blocks[0].source_cross_attn.o.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                restored_model.blocks[0].source_router.net[-1].bias,
                legacy_model.blocks[0].source_router.net[-1].bias,
            )
        )

    @unittest.skipIf(
        model_fn_wan_video is None,
        f"Pipeline dependencies unavailable: {PIPELINE_IMPORT_ERROR}",
    )
    def test_pipeline_checkpoint_path_routes_timestep_and_source_memory(self):
        model = make_model().train()
        model.enable_source_memory_attention()
        with torch.no_grad():
            for block in model.blocks:
                block.source_cross_attn.o.weight.copy_(torch.eye(model.dim))

        latents = torch.randn(2, 4, 3, 4, 2)
        source_memory = torch.randn(2, 3, 4, 32, requires_grad=True)
        output = model_fn_wan_video(
            model,
            latents=latents,
            timestep=torch.tensor([900.0, 100.0]),
            context=torch.randn(2, 5, 16),
            source_memory_by_time=source_memory,
            source_window_radius=1,
            use_gradient_checkpointing=True,
        )
        output.square().mean().backward()

        self.assertEqual(output.shape, latents.shape)
        self.assertIsNotNone(source_memory.grad)
        self.assertGreater(source_memory.grad.abs().sum().item(), 0)

    @unittest.skipIf(
        WanTrainingModule is None,
        f"Training dependencies unavailable: {PIPELINE_IMPORT_ERROR}",
    )
    def test_checkpoint_loader_migrates_legacy_dit_after_loading(self):
        model = make_model()
        model.enable_source_memory_attention()
        legacy_q = torch.full_like(model.blocks[0].cross_attn.q.weight, 0.25)
        legacy_state = {
            "pipe.dit.blocks.0.cross_attn.q.weight": legacy_q,
        }
        loader = make_checkpoint_loader(model)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/legacy.safetensors"
            save_checkpoint(path, legacy_state)
            loader.load_checkpoint_weights(path)

        self.assertTrue(
            torch.equal(model.blocks[0].source_cross_attn.q.weight, legacy_q)
        )
        self.assertEqual(
            torch.count_nonzero(model.blocks[0].source_cross_attn.o.weight).item(),
            0,
        )

    @unittest.skipIf(
        WanTrainingModule is None,
        f"Training dependencies unavailable: {PIPELINE_IMPORT_ERROR}",
    )
    def test_checkpoint_loader_accepts_complete_new_branch(self):
        source_model = make_model()
        source_model.enable_source_memory_attention()
        with torch.no_grad():
            source_model.blocks[0].source_cross_attn.o.weight.fill_(0.125)
        checkpoint_state = {
            f"pipe.dit.{key}": value
            for key, value in source_model.state_dict().items()
        }
        restored_model = make_model()
        restored_model.enable_source_memory_attention()
        loader = make_checkpoint_loader(restored_model)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/new.safetensors"
            save_checkpoint(path, checkpoint_state)
            loader.load_checkpoint_weights(path)

        self.assertTrue(
            torch.equal(
                restored_model.blocks[0].source_cross_attn.o.weight,
                source_model.blocks[0].source_cross_attn.o.weight,
            )
        )

    @unittest.skipIf(
        WanTrainingModule is None,
        f"Training dependencies unavailable: {PIPELINE_IMPORT_ERROR}",
    )
    def test_checkpoint_loader_rejects_partial_or_disabled_branch(self):
        source_model = make_model()
        source_model.enable_source_memory_attention()
        partial_state = {
            "blocks.0.source_cross_attn.q.weight": (
                source_model.blocks[0].source_cross_attn.q.weight
            ),
        }
        source_loader = make_checkpoint_loader(source_model)

        with tempfile.TemporaryDirectory() as directory:
            partial_path = f"{directory}/partial.safetensors"
            save_checkpoint(partial_path, partial_state)
            with self.assertRaisesRegex(ValueError, "partial source-memory"):
                source_loader.load_checkpoint_weights(partial_path)

            complete_source_state = {
                f"pipe.dit.{key}": value
                for key, value in source_model.state_dict().items()
                if ".source_" in key
            }
            disabled_path = f"{directory}/disabled.safetensors"
            save_checkpoint(disabled_path, complete_source_state)
            disabled_loader = make_checkpoint_loader(make_model())
            with self.assertRaisesRegex(ValueError, "does not enable temporal-local"):
                disabled_loader.load_checkpoint_weights(disabled_path)


if __name__ == "__main__":
    unittest.main()
