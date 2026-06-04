import os
import torch, types
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from typing import Optional, Union
from typing_extensions import Literal

from ..diffusion import FlowMatchScheduler
from ..diffusion.parsers import resolve_wan_action_injection_mode, resolve_wan_text_mode
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_action_encoder import WanVideoActionEncoder
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder


class WanVideoPipeline(BasePipeline):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        modules: Optional[list[str]] = None,
    ):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        if modules is not None:
            modules = [str(item).strip().lower() for item in modules if str(item).strip()]
            self.modules = tuple(modules)
            self.text_mode = resolve_wan_text_mode(modules)
            self.enable_text = self.text_mode != "none"
            self.enable_text_encoder = self.text_mode == "t5"
        else:
            self.modules = None
            self.text_mode = "t5"
            self.enable_text = True
            self.enable_text_encoder = True

        self.action_injection_mode = resolve_wan_action_injection_mode(modules)
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.action_encoder: WanVideoActionEncoder = None
        self.vae: WanVideoVAE = None
        self.source_video_projector = None
        self.source_temporal_gate = None
        self.target_state_head = None
        self.target_camera_encoder = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]
        if self.enable_text:
            self.units.append(WanVideoUnit_PromptEmbedder())
        if self.action_injection_mode != "none":
            self.units.append(WanVideoUnit_ActionEmbedder())


    def model_fn(self, *args, **kwargs):
        return model_fn_wan_video(*args, action_injection_mode=self.action_injection_mode, **kwargs)


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        vram_limit: float = None,
        modules: Optional[list[str]] = None,
    ):
        if modules is not None:
            modules = [str(item).strip().lower() for item in modules if str(item).strip()]

        text_mode = resolve_wan_text_mode(modules)
        enable_text = text_mode != "none"
        enable_text_encoder = text_mode == "t5"
        action_injection_mode = resolve_wan_action_injection_mode(modules)
        action_enabled = action_injection_mode != "none"

        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors")
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype,
            modules=modules,
        )
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder") if enable_text_encoder else None
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.action_encoder = model_pool.fetch_model("wan_video_action_encoder") if action_enabled else None

        if action_enabled and pipe.action_encoder is None:
            action_dim = 14
            dim = getattr(pipe.dit, "dim", 1536) if pipe.dit is not None else 1536
            if action_injection_mode == "adaln":
                pipe.action_encoder = WanVideoActionEncoder(
                    action_dim=action_dim,
                    dim=dim,
                    num_action_per_chunk=81,
                    in_features=None,
                    hidden_features=dim * 4,
                )
            else:
                pipe.action_encoder = WanVideoActionEncoder(action_dim=action_dim, dim=dim)
            pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None and enable_text_encoder:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        prompt_emb: Optional[Union[torch.Tensor, str, os.PathLike]] = None,
        negative_prompt_emb: Optional[Union[torch.Tensor, str, os.PathLike]] = None,
        # Unified video input (i2v uses T=1)
        input_video: Optional[torch.Tensor] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        num_history_frames=1,
        # Action conditioning
        action: Optional[torch.Tensor] = None,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        if not isinstance(input_video, torch.Tensor) or input_video.ndim != 5:
            raise TypeError("`input_video` must be a torch.Tensor with shape (V, C, T, H, W).")

        input_video = input_video.to(dtype=self.torch_dtype, device=self.device)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "prompt_emb": negative_prompt_emb,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_video": input_video, "denoising_strength": denoising_strength,
            "num_views": int(input_video.shape[0]),
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames, "num_history_frames": num_history_frames,
            "action": action,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Historical-frame conditioning path: keep history latents fixed during denoising.
        if int(input_video.shape[2]) > 1 and int(input_video.shape[2]) < int(num_frames):
            self.load_models_to_device(["vae"])
            history_latents_views = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            history_latents_views = history_latents_views.to(dtype=self.torch_dtype, device=self.device)
            history_latents = rearrange(history_latents_views, "v c t h w -> 1 c t (v h) w")
            history_t = ((num_history_frames - 1) // 4) + 1
            inputs_shared["history_latents"] = history_latents[:, :, :history_t]

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "history_latents" in inputs_shared:
                history_t = ((inputs_shared["num_history_frames"] - 1) // 4) + 1
                inputs_shared["latents"][:, :, :history_t] = inputs_shared["history_latents"][:, :, :history_t]
        
        # Decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        num_views = int(inputs_shared.get("num_views", 1))
        if latents.shape[-2] % num_views != 0:
            raise ValueError(f"Latent height {latents.shape[-2]} is not divisible by num_views={num_views}.")

        latents_by_view = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=num_views, h=latents.shape[-2] // num_views)
        video = self.vae.decode(latents_by_view, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "num_history_frames"),
            output_params=("height", "width", "num_frames", "num_history_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, num_history_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames, "num_history_frames": num_history_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit): #infer & train
    """
    作用: 在扩散模型的潜在空间(latent space)中生成初始随机噪声
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, height, width, num_frames, seed, rand_device):
        num_views = int(input_video.shape[0])
        # 计算 VAE 潜在空间的帧数
        length = (num_frames - 1) // 4 + 1
        latent_height = (height * num_views) // pipe.vae.upsampling_factor
        latent_width = width // pipe.vae.upsampling_factor
        shape = (1, pipe.vae.model.z_dim, length, latent_height, latent_width)

        # noise: (B=1, C_vae=16, F_lat=F/4, H/8, W/8)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit): # no infer & train
    """
    作用: 将输入视频(像素空间)通过 VAE 编码到潜在空间
    用途: 用于 Video-to-Video 任务,在训练和推理阶段都会用到
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, num_frames, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        if int(input_video.shape[2]) <= 1:
            # i2v path: single-frame conditioning is handled by CLIP/VAE image embedders.
            return {"latents": noise}
        if (not pipe.scheduler.training) and int(input_video.shape[2]) < int(num_frames):
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Encode each view independently in VAE batch, then concatenate views in latent height.
        input_latents_views = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        input_latents_views = input_latents_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = rearrange(input_latents_views, "v c t h w -> 1 c t (v h) w")

        if pipe.scheduler.training:
            # 训练模式: 返回纯噪声和编码后的 latents (用于计算损失)
            return {"latents": noise, "input_latents": input_latents}
        else:
            # 推理模式: 将噪声添加到编码后的 latents 上,形成扩散过程的起点
            # timesteps[0] 是扩散过程的第一个时间步 (噪声最多的时刻)
            # 这个操作相当于: noisy_latents = input_latents + noise_level * noise
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit): #infer & train
    """
    作用: 将文本提示词(prompt)编码为文本嵌入向量,作为扩散模型的条件
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True, 
            input_params_posi={"prompt": "prompt", "prompt_emb": "prompt_emb", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "prompt_emb": "prompt_emb", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        """
        1. 使用 tokenizer 将文本转换为 token IDs
        2. 使用 text_encoder 将 token IDs 编码为高维嵌入向量
        3. 清除 padding 位置的嵌入,避免影响注意力计算
        """
        if pipe.tokenizer is None or pipe.text_encoder is None:
            raise ValueError("Text encoder or tokenizer is not available. Please provide pre-extracted prompt embeddings or load the text encoder.")
        if prompt is None:
            raise ValueError("Prompt is None and no pre-extracted embedding is provided.")

        # 使用 tokenizer 将文本转换为 token IDs 和 attention mask
        # ids: (B=1, L_word=512) - token IDs,L 是序列长度 (最大512)
        # mask: (B=1, L_word=512) - attention mask,1表示有效token,0表示padding
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)

        # 计算每个样本的有效序列长度 (非padding的token数量)
        # seq_lens: (B,) - 每个样本的实际长度
        seq_lens = mask.gt(0).sum(dim=1).long()

        # prompt_emb: (B=1, L_word=512, D_text=4096)
        prompt_emb = pipe.text_encoder(ids, mask)

        # 将 padding 位置的嵌入向量置零
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt=None, positive=None, prompt_emb=None) -> dict:
        if prompt_emb is None:
            pipe.load_models_to_device(self.onload_model_names)
            prompt_emb = self.encode_prompt(pipe, prompt)
        else:
            if isinstance(prompt_emb, (str, os.PathLike)):
                prompt_emb = torch.load(prompt_emb, map_location="cpu", weights_only=False)
            if not isinstance(prompt_emb, torch.Tensor):
                prompt_emb = torch.as_tensor(prompt_emb)
            prompt_emb = prompt_emb.detach()
            prompt_emb = prompt_emb.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"context": prompt_emb}



class WanVideoUnit_ActionEmbedder(PipelineUnit): # infer & train
    """
    作用: 将动作序列编码为条件嵌入,对齐到 VAE 的时间下采样尺度
    """
    def __init__(self):
        super().__init__(
            input_params=("action", "num_frames"),
            output_params=("action_emb",),
            onload_model_names=("action_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, action, num_frames) -> dict:
        if action is None:
            return {}
        if pipe.action_encoder is None:
            raise ValueError("Action encoder is not available in the pipeline.")

        pipe.load_models_to_device(self.onload_model_names)
        # action[B，F,14]
        action = torch.as_tensor(action, device=pipe.device, dtype=pipe.torch_dtype)

        if pipe.action_injection_mode == "noise":
          length = (num_frames - 1) // 4 + 1
          # action[B，F+3,14]
          action = torch.concat(
              [torch.repeat_interleave(action[:, 0:1], repeats=4, dim=1), action[:, 1:]],
              dim=1,
          )
          # action[B，F/4,14]
          action = action.contiguous().view(action.shape[0], length, 4, action.shape[-1]).mean(dim=2)
          # action_emb[B，F/4,D_model]

        if pipe.action_injection_mode == "adaln":
            # action[B, F*14]
            action = rearrange(action, "b f d -> b (f d)").contiguous()
            # action_emb[B, D_model]
            
        action_emb = pipe.action_encoder(action)
        return {"action_emb": action_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit): #infer & train
    """
    CLIP 图像编码器单元
    作用: 使用 CLIP 模型将输入图像编码为高层语义特征,作为条件信息
    用途: Image-to-Video 任务
    特点: 提取的是图像的语义特征
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_history_frames"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, height, width, num_history_frames):
        if input_video is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding or not pipe.dit.has_image_input:
            return {}

        pipe.load_models_to_device(self.onload_model_names)
        current_frame = input_video[:, :, num_history_frames - 1]
        image = rearrange(current_frame, "v c h w -> 1 c (v h) w")

        # 使用 CLIP 图像编码器提取特征
        # clip_context: (B=1, N_token=1[cls]+256=257, D_clip=1280)
        clip_context = pipe.image_encoder.encode_image([image])

        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit): #infer &train
    """
    VAE 图像编码器单元
    作用: 将输入的开始帧（可选末尾帧）通过 VAE 编码为潜在表示,并附加掩码信息
    用途: Image-to-Video / First-Last-Frame 任务
    特点: 提供像素级的条件信息,并通过掩码明确标记哪些帧是已知的条件帧

    双端锚定模式 (num_tail_frames > 0):
        - input_video[:, :, -num_tail_frames:] 作为末尾已知帧, 与首帧一起 encode
        - mask 通道在末端 num_tail_frames 帧也置 1, 让 DiT 通过 patch embedding
          的 mask 通道感知哪些 latent slot 是已知锚帧
        - 等价于 WAN-Fun-InP 原版的 first+end image 双端 anchored generation
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "num_frames", "num_history_frames",
                          "num_tail_frames", "anchor_frame_indices", "height", "width", "tiled",
                          "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, num_frames,
                num_history_frames, num_tail_frames, anchor_frame_indices, height, width,
                tiled, tile_size, tile_stride):
        if input_video is None or not pipe.dit.require_vae_embedding or not pipe.dit.has_image_input:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        num_history_frames = int(num_history_frames)
        num_tail_frames = int(num_tail_frames or 0)
        if num_tail_frames < 0:
            raise ValueError(f"num_tail_frames must be >= 0, got {num_tail_frames}")
        if num_history_frames + num_tail_frames > int(num_frames):
            raise ValueError(
                f"num_history_frames({num_history_frames}) + "
                f"num_tail_frames({num_tail_frames}) exceeds num_frames({num_frames})."
            )
        num_views = int(input_video.shape[0])
        known_indices = set(range(num_history_frames))
        if num_tail_frames > 0:
            known_indices.update(range(int(num_frames) - num_tail_frames, int(num_frames)))
        if anchor_frame_indices is not None:
            if isinstance(anchor_frame_indices, torch.Tensor):
                extra_indices = [int(item) for item in anchor_frame_indices.detach().cpu().flatten().tolist()]
            elif isinstance(anchor_frame_indices, (list, tuple, set)):
                extra_indices = [int(item) for item in anchor_frame_indices]
            else:
                extra_indices = [int(anchor_frame_indices)]
            for index in extra_indices:
                if index < 0 or index >= int(num_frames):
                    raise ValueError(
                        f"anchor_frame_indices contains out-of-range frame {index} "
                        f"for num_frames={num_frames}."
                    )
                known_indices.add(index)

        # 创建掩码 (mask),标记哪些帧是已知的条件帧
        # msk: (B=1, F=num_frames, H/8, W/8); head/tail/keyframe anchors 置 1
        msk = torch.zeros(1, num_frames, height//8, width//8, device=pipe.device, dtype=pipe.torch_dtype)
        for index in sorted(known_indices):
            msk[:, index:index + 1] = 1

        # 时间下采样到 latent 时间轴 (4x): 第一帧重复 4 次以对齐 VAE patch
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]                     # (4, F/4, H/8, W/8)
        msk = msk.unsqueeze(0).repeat(num_views, 1, 1, 1, 1)
        msk = rearrange(msk, "v c t h w -> c t (v h) w")

        # 像素帧输入: 未知位置为 0，known anchors 保持在原始时间位置。
        # 这保持了 head-only / dual-end 行为，同时允许中间关键帧作为额外 anchor。
        vae_inputs = torch.zeros(
            num_views, input_video.shape[1], int(num_frames), int(height), int(width),
            dtype=input_video.dtype,
            device=input_video.device,
        )
        for index in sorted(known_indices):
            vae_inputs[:, :, index] = input_video[:, :, index]

        # 使用 VAE 编码器将输入编码到潜在空间
        # y_views: (V, C_vae=16, F/4, H/8, W/8)
        y_views = pipe.vae.encode(vae_inputs, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        y_views = y_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        # (V, C, T, H, W) -> (C, T, V*H, W)
        y = rearrange(y_views, "v c t h w -> c t (v h) w")

        # 将掩码和 VAE 编码后的特征拼接在通道维度
        # 拼接后: y: (C=4+16=20, F/4, H/8, W/8)
        # - 前4个通道: 掩码信息 (标记哪些帧是已知的; 双端时 head/tail 都置 1)
        # - 后16个通道: VAE 编码的潜在表示 (整段 encode, slot 语义自洽)
        y = torch.concat([msk, y])
        # 添加 batch 维度: (1, C=20, F/4, H/8, W/8)
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "none",
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    source_tokens: Optional[torch.Tensor] = None,
    source_memory_by_time: Optional[torch.Tensor] = None,
    source_window_radius: int = 1,
    return_hidden_by_time: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    scene_tokens: Optional[torch.Tensor] = None,
    geometry_gates=None,
    target_camera_emb: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    - latents: (B=1, C=16, F/4, H/8, W/8) - 潜在空间的噪声
    - timestep: 当前扩散时间步 (0~1000)
    - context: (B=1, L_word=512, D_text=4096) - 文本s嵌入
    - action_emb: (B=1, F/4, D_model) or (B=1, D_model) - action 嵌入 (可选)
    - clip_feature: (B=1, N_token=257, D_clip=1280) - CLIP 图像特征 (可选)
    - y: (B=1, C=4[mask]+16[vae]=20, F/4, H/8, W/8) - mask+VAE 编码的首帧 (可选)
    - use_gradient_checkpointing: 是否使用梯度检查点 (节省显存)
    - use_gradient_checkpointing_offload: 是否将中间激活值 offload 到 CPU (进一步节省显存)
    """

    # ========== 步骤1: 时间步编码 ==========
    # t: (B=1, D_model=1536)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))

    if action_injection_mode == "adaln" and action_emb is not None:
        t = t + action_emb

    # 将时间嵌入投影为调制参数
    # t_mod: (B=1, 6, D_model=1536)
    # 这6个参数用于 AdaLN (Adaptive Layer Normalization) 调制
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # ========== 步骤2: 文本条件处理 ==========
    # context: (B=1, L_word=512, D_text=4096) -> (B=1, L_word=512, D_model=1536)
    if context is not None:
        context = dit.text_embedding(context)
    if source_tokens is not None:
        source_tokens = source_tokens.to(dtype=latents.dtype, device=latents.device)
        if context is None:
            context = source_tokens
        else:
            context = torch.cat([context, source_tokens], dim=1)

    x = latents

    # ========== 步骤3: 图像条件整合 ==========
    # 3.1 整合 VAE 图像条件 (如果提供)
    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        # x:加噪嵌入 与mask 首帧vae嵌入在通道维度拼接
        # x 加噪嵌入: (B=1, C=16, F/4, H/8, W/8) + y: VAE编码: (B=1, C_y=20, F/4, H/8, W/8) -> (B=1, C+C_y=36, F/4, H/8, W/8)
        # 这样模型可以同时看到噪声 latent 和条件图像的潜在表示
        x = torch.cat([x, y], dim=1)

    # 3.2 整合 CLIP 图像特征 (如果提供)
    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        # context: (B=1, L_token=512, D_model=1536) + clip_emb: (B=1, N_img=257, D_model=1538) -> (B, L+N_img=769, D_model)
        if context is None:
            context = clip_embdding
        else:
            context = torch.cat([clip_embdding, context], dim=1)


    # ========== 步骤4: Patchify - 将3D体积转换为token序列 ==========
    # 将连续的潜在表示切分为不重叠的3D patch
    # x: (B=1, C+C_y=20, F/4, H/8, W/8) -> (B=1, D_model=1536, F/8, H/16, W/16)
    # - F_p=4, H_p=2, W_p=2: 单个patch所占 (时间、高度、宽度方向)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]  # 记录 patch 的网格尺寸
    
    if action_injection_mode == "noise" and action_emb is not None:
        # action_emb: (B, F, D_model) -> (B, D_model, F, 1, 1), broadcast to (H/16, W/16)
        action_emb = rearrange(action_emb, "b f d -> b d f 1 1")
        action_emb = repeat(action_emb, "b d f 1 1 -> b d f h w", h=h, w=w)
        x = x + action_emb

    if target_camera_emb is not None:
        target_camera_emb = target_camera_emb.to(dtype=x.dtype, device=x.device)
        if target_camera_emb.ndim == 2:
            target_camera_emb = target_camera_emb.unsqueeze(0)
        if target_camera_emb.shape[1] != f:
            raise ValueError(
                "Target camera embedding length "
                f"{target_camera_emb.shape[1]} does not match latent token frames {f}."
            )
        target_camera_emb = rearrange(target_camera_emb, "b f d -> b d f 1 1")
        x = x + target_camera_emb

    # 将3D patch grid 展平为1D token 序列
    # x: (B, D_model, F/4, H/16, W/16) -> (B, N总token数, D_model=1536)
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    if source_memory_by_time is not None:
        source_memory_by_time = source_memory_by_time.to(
            dtype=x.dtype,
            device=x.device,
        )
        if source_memory_by_time.shape[1] != f:
            raise ValueError(
                "Temporal source memory length "
                f"{source_memory_by_time.shape[1]} does not match latent token frames {f}."
            )

    # ========== 步骤5: 位置编码 (RoPE - Rotary Position Embedding) ==========
    # 为每个 token 生成3D位置编码 (时间、高度、宽度)
    # freqs: (N总token, 1, D_freq=64)
    # 每个 token 的位置编码由其在 (f, h, w) grid 中的坐标决定
    # RoPE 会在 attention 计算时旋转 query 和 key,从而注入位置信息
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # 时间维度的位置编码
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # 高度维度的位置编码
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)   # 宽度维度的位置编码
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # ========== 步骤6: Transformer Blocks - 去噪主循环 ==========
    def create_custom_forward(module, block_scene_tokens, block_gate_scene, block_gate_source):
        def custom_forward(*inputs):
            return module(
                *inputs,
                source_memory_by_time=source_memory_by_time,
                source_window_radius=source_window_radius,
                token_grid=(f, h, w),
                scene_tokens=block_scene_tokens,
                gate_scene=block_gate_scene,
                gate_source=block_gate_source,
            )
        return custom_forward

    for block_idx, block in enumerate(dit.blocks):
        block_scene_tokens = None
        block_gate_scene = None
        block_gate_source = None
        if scene_tokens is not None and geometry_gates is not None:
            gate_module = geometry_gates[block_idx]
            block_gate_scene, block_gate_source = gate_module(t)
            block_scene_tokens = scene_tokens

        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, block_scene_tokens, block_gate_scene, block_gate_source),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block, block_scene_tokens, block_gate_scene, block_gate_source),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(
                x,
                context,
                t_mod,
                freqs,
                source_memory_by_time=source_memory_by_time,
                source_window_radius=source_window_radius,
                token_grid=(f, h, w),
                scene_tokens=block_scene_tokens,
                gate_scene=block_gate_scene,
                gate_source=block_gate_source,
            )

    hidden_by_time = None
    if return_hidden_by_time:
        hidden_by_time = x.reshape(x.shape[0], f, h, w, x.shape[-1])

    # ========== 步骤7: 输出投影和 Unpatchify ==========
    # 7.1 使用最终的 head 层进行输出投影
    # 这里会再次使用时间步 t 进行调制,并投影到输出通道数
    # x: (B, N总token, D_model) -> (B, N总token, C_out=64)
    x = dit.head(x, t)

    # 7.2 将 token 序列重构回 3D 体积
    # x: (B, N, C_out=64) -> (B, C_vae=16, F/4, H/8, W/8)
    # 这就是模型预测的噪声
    x = dit.unpatchify(x, (f, h, w))

    if return_hidden_by_time:
        return x, hidden_by_time
    return x
