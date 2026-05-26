import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_normalized_uv_grid(height, width, device, dtype):
    u = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)[None, :].expand(
        height, -1
    )
    v = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)[:, None].expand(
        -1, width
    )
    return torch.stack([u, v], dim=0)


def _make_pixel_center_grid(height, width, device, dtype):
    x = torch.linspace(0.5, width - 0.5, width, device=device, dtype=dtype)[
        None, :
    ].expand(height, -1)
    y = torch.linspace(0.5, height - 0.5, height, device=device, dtype=dtype)[
        :, None
    ].expand(-1, width)
    return torch.stack([x, y], dim=-1)


def _skew_symmetric(vec):
    zeros = torch.zeros_like(vec[..., 0])
    x, y, z = vec.unbind(dim=-1)
    return torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(*vec.shape[:-1], 3, 3)


def _rotation_from_rotvec(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    unit_axis = rotvec / theta.clamp(min=1e-8)
    skew = _skew_symmetric(unit_axis)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(
        *rotvec.shape[:-1], 3, 3
    )
    sin_term = torch.sin(theta)[..., None]
    cos_term = torch.cos(theta)[..., None]
    return eye + sin_term * skew + (1.0 - cos_term) * (skew @ skew)


def _apply_camera_local_delta(c2w, rotvec, translation):
    delta_rot = _rotation_from_rotvec(rotvec)
    delta_pose = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand_as(c2w).clone()
    delta_pose[..., :3, :3] = delta_rot
    delta_pose[..., :3, 3] = translation
    return c2w @ delta_pose


def _build_plucker_rays(c2w, intrinsics_fxfycxcy, uv_offsets_px, target_hw):
    batch_size, num_views = c2w.shape[:2]
    height, width = target_hw
    dtype = c2w.dtype
    device = c2w.device

    uv = _make_pixel_center_grid(height, width, device=device, dtype=dtype)
    uv = uv.view(1, 1, height, width, 2).expand(batch_size, num_views, -1, -1, -1)
    uv = uv + uv_offsets_px.permute(0, 1, 3, 4, 2)

    fx = intrinsics_fxfycxcy[..., 0][..., None, None]
    fy = intrinsics_fxfycxcy[..., 1][..., None, None]
    cx = intrinsics_fxfycxcy[..., 2][..., None, None]
    cy = intrinsics_fxfycxcy[..., 3][..., None, None]

    dirs_local = torch.stack(
        [
            (uv[..., 0] - cx) / fx,
            (uv[..., 1] - cy) / fy,
            torch.ones_like(uv[..., 0]),
        ],
        dim=-1,
    )
    dirs_local = dirs_local / torch.linalg.norm(
        dirs_local, dim=-1, keepdim=True
    ).clamp(min=1e-8)
    dirs_global = torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], dirs_local)
    ray_origins = c2w[..., :3, 3][:, :, None, None, :].expand_as(dirs_global)
    moments = torch.cross(ray_origins, dirs_global, dim=-1)
    rays = torch.cat([moments, dirs_global], dim=-1)
    return rays.permute(0, 1, 4, 2, 3)


class RayFieldAdapter(nn.Module):
    """Target-only learned calibration residual applied to Plucker rays."""

    def __init__(
        self,
        cam_token_dim=11,
        view_embed_dim=16,
        uv_lowres_hw=(45, 80),
        max_rot_deg=2.0,
        max_trans_norm=0.05,
        max_uv_offset_px=4.0,
        limit_warmup_steps=10000,
    ):
        super().__init__()
        self.max_rot_deg = max_rot_deg
        self.max_trans_norm = max_trans_norm
        self.max_uv_offset_px = max_uv_offset_px
        self.limit_warmup_steps = limit_warmup_steps
        self.uv_lowres_hw = tuple(uv_lowres_hw)

        self.view_embed = nn.Embedding(4, view_embed_dim)
        pose_in_dim = cam_token_dim + view_embed_dim
        self.pose_head = nn.Sequential(
            nn.Linear(pose_in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
        )
        uv_in_dim = cam_token_dim + view_embed_dim + 2
        self.uv_head = nn.Sequential(
            nn.Conv2d(uv_in_dim, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )

        self.register_buffer(
            "uv_lowres_grid",
            _make_normalized_uv_grid(
                self.uv_lowres_hw[0], self.uv_lowres_hw[1], device="cpu", dtype=torch.float32
            ).unsqueeze(0),
            persistent=False,
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        nn.init.zeros_(self.uv_head[-1].weight)
        nn.init.zeros_(self.uv_head[-1].bias)

    def _limit_scale(self, iter_idx):
        if iter_idx is None or self.limit_warmup_steps <= 0:
            return 1.0
        progress = min(float(iter_idx) / float(self.limit_warmup_steps), 1.0)
        return 0.25 + 0.75 * progress

    def forward(self, target_rays, target_cam_token, target_ray_meta, iter_idx=None):
        if target_ray_meta is None:
            return target_rays, {}

        batch_size, num_views, _, height, width = target_rays.shape
        device = target_rays.device
        float_dtype = target_ray_meta["c2w_norm"].dtype

        c2w = target_ray_meta["c2w_norm"].to(device=device, dtype=float_dtype)
        intrinsics = target_ray_meta["intrinsics_px_post_crop"].to(
            device=device, dtype=float_dtype
        )
        view_ids = target_ray_meta["view_ids"].to(device=device)
        view_ids = torch.where(
            (view_ids >= 0) & (view_ids <= 2),
            view_ids,
            torch.full_like(view_ids, 3),
        )

        view_features = self.view_embed(view_ids)
        pose_input = torch.cat(
            [
                target_cam_token.to(dtype=view_features.dtype),
                view_features,
            ],
            dim=-1,
        )
        pose_delta_raw = self.pose_head(pose_input).to(dtype=float_dtype)

        limit_scale = self._limit_scale(iter_idx)
        rot_limit_rad = math.radians(self.max_rot_deg) * limit_scale
        trans_limit = self.max_trans_norm * limit_scale
        uv_limit = self.max_uv_offset_px * limit_scale

        rot_delta = rot_limit_rad * torch.tanh(pose_delta_raw[..., :3])
        trans_delta = trans_limit * torch.tanh(pose_delta_raw[..., 3:])
        c2w_corrected = _apply_camera_local_delta(c2w, rot_delta, trans_delta)

        lowres_h, lowres_w = self.uv_lowres_hw
        cond_features = torch.cat(
            [
                target_cam_token.to(dtype=view_features.dtype),
                view_features,
            ],
            dim=-1,
        )
        cond_features = cond_features.reshape(batch_size * num_views, -1, 1, 1)
        cond_features = cond_features.expand(-1, -1, lowres_h, lowres_w)
        uv_grid = self.uv_lowres_grid.to(
            device=device, dtype=cond_features.dtype
        ).expand(batch_size * num_views, -1, -1, -1)
        uv_input = torch.cat([cond_features, uv_grid], dim=1)
        uv_delta_lowres = self.uv_head(uv_input)
        uv_delta = F.interpolate(
            uv_delta_lowres,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        uv_delta = uv_limit * torch.tanh(uv_delta)
        uv_delta = uv_delta.view(batch_size, num_views, 2, height, width).to(
            dtype=float_dtype
        )

        corrected_rays = _build_plucker_rays(
            c2w_corrected,
            intrinsics,
            uv_delta,
            target_hw=(height, width),
        ).to(dtype=target_rays.dtype)

        aux_outputs = {
            "uv_reg": uv_delta.abs().mean(),
            "pose_reg": rot_delta.abs().mean() + trans_delta.abs().mean(),
            "mean_abs_uv_offset": uv_delta.abs().mean().detach(),
            "max_abs_uv_offset": uv_delta.abs().amax().detach(),
            "mean_rot_delta_deg": (rot_delta.abs().mean() * 180.0 / math.pi).detach(),
            "mean_trans_delta_norm": torch.linalg.norm(trans_delta, dim=-1).mean().detach(),
            "ray_limit_scale": torch.tensor(limit_scale, device=device, dtype=float_dtype),
        }
        return corrected_rays, aux_outputs
