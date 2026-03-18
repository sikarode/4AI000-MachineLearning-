"""
Inverse Design of Magnetoactive Mechanical Metamaterials
=========================================================
Two-stage conditional diffusion model:
  Stage 1: Stress/Strain curves -> Orientation Map (OM)
  Stage 2: Orientation Map + MagCompression -> Br field

Data: .data files with variable-length timesteps, resampled to fixed grid.
Augmentation: 4x rotation (0/90/180/270 degrees).
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
class Config:
    data_dir = "./data"
    image_size = 128
    n_resample = 64

    s1_condition_rows = 8
    s1_condition_cols = n_resample
    s2_mag_condition_rows = 8
    s2_mag_condition_cols = n_resample

    cross_attention_dim = 256
    condition_seq_len = 8

    batch_size = 4
    lr = 2e-4
    num_epochs_stage1 = 500
    num_epochs_stage2 = 500
    num_diffusion_steps = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "./checkpoints"
    ema_decay = 0.999

cfg = Config()


# ============================================================
# Utility
# ============================================================
def resample_curves(block, n_out):
    """Resample (n_rows, T) to (n_rows, n_out) via linear interpolation."""
    n_rows, t_orig = block.shape
    if t_orig == n_out:
        return block.copy()
    x_orig = np.linspace(0.0, 1.0, t_orig)
    x_new = np.linspace(0.0, 1.0, n_out)
    resampled = np.empty((n_rows, n_out), dtype=block.dtype)
    for i in range(n_rows):
        resampled[i] = np.interp(x_new, x_orig, block[i])
    return resampled


class EMAModel:
    """Exponential Moving Average of model weights for stable generation."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


# ============================================================
# Dataset with 4x rotation augmentation
# ============================================================
class MetamaterialDataset(Dataset):
    """
    Loads .data files, applies 4x augmentation (0/90/180/270 rotation),
    caches everything in RAM.
    """

    def __init__(self, data_dir, image_size=128, n_resample=64, augment=True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.data")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .data files found in {data_dir}")
        self.image_size = image_size
        self.n_resample = n_resample
        self.augment = augment
        self._preload_and_cache()

    def _preload_and_cache(self):
        raw_base, raw_mag = [], []
        raw_om_images, raw_br_images = [], []
        all_om_vals = []

        print(f"  Loading {len(self.files)} files into memory...")
        for fpath in self.files:
            with open(fpath, "rb") as f:
                d = pickle.load(f)
            bc, mc = self._extract_conditions(d, self.n_resample)
            raw_base.append(bc)
            raw_mag.append(mc)
            raw_br_images.append(self._extract_br_raw(d))
            om = d["orientationMap"]
            all_om_vals.append(om[~np.isnan(om)])
            raw_om_images.append(om)

        # Compute normalisation stats before augmentation
        all_base = np.stack(raw_base)
        all_mag = np.stack(raw_mag)
        all_br = np.stack(raw_br_images)
        all_om_v = np.concatenate(all_om_vals)

        self.base_cond_min = all_base.min(axis=(0, 2), keepdims=True)
        self.base_cond_max = all_base.max(axis=(0, 2), keepdims=True)
        self.base_cond_range = self.base_cond_max - self.base_cond_min
        self.base_cond_range[self.base_cond_range < 1e-12] = 1.0

        self.mag_cond_min = all_mag.min(axis=(0, 2), keepdims=True)
        self.mag_cond_max = all_mag.max(axis=(0, 2), keepdims=True)
        self.mag_cond_range = self.mag_cond_max - self.mag_cond_min
        self.mag_cond_range[self.mag_cond_range < 1e-12] = 1.0

        self.br_min = float(all_br.min())
        self.br_max = float(all_br.max())
        self.br_range = self.br_max - self.br_min if (self.br_max - self.br_min) > 1e-12 else 1.0

        self.om_min = float(all_om_v.min())
        self.om_max = float(all_om_v.max())
        self.om_range = self.om_max - self.om_min if (self.om_max - self.om_min) > 1e-12 else 1.0

        # Build augmented cache
        bmin, brng = self.base_cond_min.squeeze(0), self.base_cond_range.squeeze(0)
        mmin, mrng = self.mag_cond_min.squeeze(0), self.mag_cond_range.squeeze(0)

        self.cached_base, self.cached_mag = [], []
        self.cached_om, self.cached_br = [], []

        rotations = [0, 1, 2, 3] if self.augment else [0]

        for i in range(len(self.files)):
            bc_norm = ((raw_base[i] - bmin) / brng).astype(np.float32)
            mc_norm = ((raw_mag[i] - mmin) / mrng).astype(np.float32)

            om = raw_om_images[i].copy()
            mask = (~np.isnan(om)).astype(np.float32)
            om_clean = np.nan_to_num(om, nan=0.0)
            om_clean = ((om_clean - self.om_min) / self.om_range).astype(np.float32) * mask
            om_2ch = np.stack([mask, om_clean], axis=0)
            om_t = F.interpolate(torch.from_numpy(om_2ch).unsqueeze(0),
                                 size=self.image_size, mode="bilinear",
                                 align_corners=False).squeeze(0)

            br = ((raw_br_images[i] - self.br_min) / self.br_range).astype(np.float32)
            br_t = F.interpolate(torch.from_numpy(br).unsqueeze(0),
                                 size=self.image_size, mode="bilinear",
                                 align_corners=False).squeeze(0)

            bc_tensor = torch.from_numpy(bc_norm)
            mc_tensor = torch.from_numpy(mc_norm)

            for k in rotations:
                self.cached_base.append(bc_tensor)
                self.cached_mag.append(mc_tensor)

                if k == 0:
                    self.cached_om.append(om_t)
                    self.cached_br.append(br_t)
                else:
                    om_rot = torch.rot90(om_t, k, [1, 2])
                    br_rot = torch.rot90(br_t, k, [1, 2])
                    if k == 1:
                        br_rot = torch.stack([-br_rot[1], br_rot[0]], dim=0)
                    elif k == 2:
                        br_rot = -br_rot
                    elif k == 3:
                        br_rot = torch.stack([br_rot[1], -br_rot[0]], dim=0)
                    self.cached_om.append(om_rot)
                    self.cached_br.append(br_rot)

        del raw_base, raw_mag, raw_br_images, raw_om_images, all_om_vals
        del all_base, all_mag, all_br, all_om_v

        n_orig = len(self.files)
        n_total = len(self.cached_om)
        aug_str = f" (raw {n_orig} x {len(rotations)} rotations = {n_total})" if self.augment else ""
        print(f"  Preload complete: {n_total} samples{aug_str}")

    @staticmethod
    def _extract_conditions(d, n_resample):
        bc = d["baseCompression"]
        mc = d["MagCompression"]
        base_raw = np.concatenate([bc["F"], bc["Stress"]], axis=0)
        mag_raw = np.concatenate([mc["F"], mc["Stress"]], axis=0)
        return resample_curves(base_raw, n_resample), resample_curves(mag_raw, n_resample)

    @staticmethod
    def _extract_br_raw(d):
        br_flat = d["Br"]
        return np.stack([br_flat[0::2].reshape(256, 256),
                         br_flat[1::2].reshape(256, 256)], axis=0)

    def __len__(self):
        return len(self.cached_om)

    def __getitem__(self, idx):
        return (self.cached_base[idx], self.cached_mag[idx],
                self.cached_om[idx], self.cached_br[idx])


# ============================================================
# Stage 1 Encoder
# ============================================================
class StressStrainEncoder(nn.Module):
    def __init__(self, in_rows=8, in_cols=64,
                 cross_attention_dim=256, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.cross_attention_dim = cross_attention_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_rows, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 8, 512), nn.GELU(),
            nn.Linear(512, seq_len * cross_attention_dim),
        )

    def forward(self, x):
        h = self.conv_layers(x).reshape(x.size(0), -1)
        return self.proj(h).view(x.size(0), self.seq_len, self.cross_attention_dim)


# ============================================================
# Stage 2 Encoder
# ============================================================
class OMAndMagEncoder(nn.Module):
    def __init__(self, om_channels=2, mag_rows=8, mag_cols=64,
                 cross_attention_dim=256, om_seq_len=4, mag_seq_len=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.om_seq_len = om_seq_len
        self.mag_seq_len = mag_seq_len

        self.om_extractor = nn.Sequential(
            nn.Conv2d(om_channels, 32, 4, 4), nn.GELU(),
            nn.Conv2d(32, 64, 4, 4), nn.GELU(),
            nn.Conv2d(64, 128, 4, 4), nn.GELU(),
            nn.Flatten(),
        )
        self.om_proj = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, om_seq_len * cross_attention_dim),
        )
        self.mag_conv = nn.Sequential(
            nn.Conv1d(mag_rows, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.mag_proj = nn.Sequential(
            nn.Linear(128 * 8, 512), nn.GELU(),
            nn.Linear(512, mag_seq_len * cross_attention_dim),
        )

    def forward(self, om, mag_cond):
        B = om.size(0)
        om_tok = self.om_proj(self.om_extractor(om)).view(B, self.om_seq_len, self.cross_attention_dim)
        mag_tok = self.mag_proj(self.mag_conv(mag_cond).reshape(B, -1)).view(B, self.mag_seq_len, self.cross_attention_dim)
        return torch.cat([om_tok, mag_tok], dim=1)


# ============================================================
# U-Net builders
# ============================================================
def build_stage1_unet(cross_attention_dim=256):
    return UNet2DConditionModel(
        sample_size=128, in_channels=2, out_channels=2,
        cross_attention_dim=cross_attention_dim,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,
    )

def build_stage2_unet(cross_attention_dim=256):
    return UNet2DConditionModel(
        sample_size=128, in_channels=2, out_channels=2,
        cross_attention_dim=cross_attention_dim,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,
    )
