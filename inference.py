"""
inference.py - Load trained model and generate from input conditions
=====================================================================
Usage:
    python inference.py --model_dir ./output --input_file sample_1.data
    python inference.py --model_dir ./output --input_dir ./test_data
    python inference.py --model_dir ./output --base_cond base.npy --mag_cond mag.npy
"""

import os
import sys
import json
import glob
import argparse
import pickle
import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from inverse_design_diffusion import (
    StressStrainEncoder, OMAndMagEncoder,
    build_stage1_unet, build_stage2_unet,
    resample_curves,
)


class TrainedPipeline:
    """Wraps the trained two-stage model for easy inference."""

    def __init__(self, s1_encoder, s1_unet, s1_scheduler,
                 s2_encoder, s2_unet, s2_scheduler,
                 norm_stats, image_size, diffusion_steps, n_resample, device):
        self.s1_encoder = s1_encoder.eval().to(device)
        self.s1_unet = s1_unet.eval().to(device)
        self.s1_scheduler = s1_scheduler
        self.s2_encoder = s2_encoder.eval().to(device)
        self.s2_unet = s2_unet.eval().to(device)
        self.s2_scheduler = s2_scheduler
        self.norm_stats = norm_stats
        self.image_size = image_size
        self.diffusion_steps = diffusion_steps
        self.n_resample = n_resample
        self.device = device

    @classmethod
    def load(cls, model_dir, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_dir}...")

        with open(os.path.join(model_dir, "norm_stats.json")) as f:
            norm_stats = json.load(f)
        n_resample = norm_stats.get("n_resample", 64)
        print(f"  Norm stats OK (n_resample={n_resample})")

        s1_path = os.path.join(model_dir, "stage1_final.pt")
        if not os.path.exists(s1_path):
            s1_path = _find_latest_ckpt(os.path.join(model_dir, "checkpoints"), "stage1")
        s1_ckpt = torch.load(s1_path, map_location=device, weights_only=False)
        s1_cfg = s1_ckpt["cfg"]
        s1_enc = StressStrainEncoder(
            in_rows=s1_cfg["in_rows"], in_cols=s1_cfg["in_cols"],
            cross_attention_dim=s1_cfg["cross_attention_dim"],
            seq_len=s1_cfg["seq_len"])
        s1_enc.load_state_dict(s1_ckpt["encoder"])
        s1_unet = build_stage1_unet(s1_cfg["cross_attention_dim"])
        s1_unet.load_state_dict(s1_ckpt["unet"])
        s1_sched = DDPMScheduler(num_train_timesteps=s1_cfg["diffusion_steps"],
                                 beta_schedule="squaredcos_cap_v2")
        print(f"  Stage 1 OK ({s1_path})")

        s2_path = os.path.join(model_dir, "stage2_final.pt")
        if not os.path.exists(s2_path):
            s2_path = _find_latest_ckpt(os.path.join(model_dir, "checkpoints"), "stage2")
        s2_ckpt = torch.load(s2_path, map_location=device, weights_only=False)
        s2_cfg = s2_ckpt["cfg"]
        s2_enc = OMAndMagEncoder(
            om_channels=s2_cfg["om_channels"], mag_rows=s2_cfg["mag_rows"],
            mag_cols=s2_cfg["mag_cols"],
            cross_attention_dim=s2_cfg["cross_attention_dim"],
            om_seq_len=s2_cfg["om_seq_len"], mag_seq_len=s2_cfg["mag_seq_len"])
        s2_enc.load_state_dict(s2_ckpt["encoder"])
        s2_unet = build_stage2_unet(s2_cfg["cross_attention_dim"])
        s2_unet.load_state_dict(s2_ckpt["unet"])
        s2_sched = DDPMScheduler(num_train_timesteps=s2_cfg["diffusion_steps"],
                                 beta_schedule="squaredcos_cap_v2")
        print(f"  Stage 2 OK ({s2_path})")

        image_size = s1_cfg["image_size"]
        diffusion_steps = s1_cfg["diffusion_steps"]
        print(f"  Device: {device} | Image: {image_size} | Steps: {diffusion_steps}")

        return cls(s1_enc, s1_unet, s1_sched,
                   s2_enc, s2_unet, s2_sched,
                   norm_stats, image_size, diffusion_steps, n_resample, device)

    def _norm_base(self, raw):
        mn = np.array(self.norm_stats["base_cond_min"]).squeeze(0)
        rng = np.array(self.norm_stats["base_cond_range"]).squeeze(0)
        return ((raw - mn) / rng).astype(np.float32)

    def _norm_mag(self, raw):
        mn = np.array(self.norm_stats["mag_cond_min"]).squeeze(0)
        rng = np.array(self.norm_stats["mag_cond_range"]).squeeze(0)
        return ((raw - mn) / rng).astype(np.float32)

    def _denorm_br(self, br):
        return br * self.norm_stats["br_range"] + self.norm_stats["br_min"]

    def _denorm_om(self, angle):
        return angle * self.norm_stats["om_range"] + self.norm_stats["om_min"]

    @torch.no_grad()
    def generate(self, base_cond_raw, mag_cond_raw, num_samples=1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        base_rs = resample_curves(base_cond_raw, self.n_resample)
        mag_rs = resample_curves(mag_cond_raw, self.n_resample)

        base_n = torch.from_numpy(self._norm_base(base_rs))
        mag_n = torch.from_numpy(self._norm_mag(mag_rs))
        base_n = base_n.unsqueeze(0).expand(num_samples, -1, -1).to(self.device)
        mag_n = mag_n.unsqueeze(0).expand(num_samples, -1, -1).to(self.device)

        # Stage 1
        cond1 = self.s1_encoder(base_n)
        om = torch.randn(num_samples, 2, self.image_size, self.image_size, device=self.device)
        self.s1_scheduler.set_timesteps(self.diffusion_steps)
        for t in tqdm(self.s1_scheduler.timesteps, desc="Stage 1 denoising", leave=False):
            pred = self.s1_unet(om, t.expand(num_samples).to(self.device),
                                encoder_hidden_states=cond1).sample
            om = self.s1_scheduler.step(pred, t, om).prev_sample
        om[:, 0] = (om[:, 0] > 0.5).float()
        om[:, 1] = om[:, 1].clamp(0, 1) * om[:, 0]

        # Stage 2
        cond2 = self.s2_encoder(om, mag_n)
        br = torch.randn(num_samples, 2, self.image_size, self.image_size, device=self.device)
        self.s2_scheduler.set_timesteps(self.diffusion_steps)
        for t in tqdm(self.s2_scheduler.timesteps, desc="Stage 2 denoising", leave=False):
            pred = self.s2_unet(br, t.expand(num_samples).to(self.device),
                                encoder_hidden_states=cond2).sample
            br = self.s2_scheduler.step(pred, t, br).prev_sample

        om_np = om.cpu().numpy()
        br_np = br.cpu().numpy()
        results = []
        for i in range(num_samples):
            mask = om_np[i, 0]
            angle = self._denorm_om(om_np[i, 1]) * mask
            br_phys = self._denorm_br(br_np[i])
            results.append({
                "om_mask": mask, "om_angle": angle,
                "br_x": br_phys[0], "br_y": br_phys[1],
                "om_raw": om_np[i], "br_raw": br_np[i],
            })
        return results


def _find_latest_ckpt(ckpt_dir, prefix):
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch*.pt")))
    if files:
        return files[-1]
    best = os.path.join(ckpt_dir, f"{prefix}_best.pt")
    if os.path.exists(best):
        return best
    raise FileNotFoundError(f"No {prefix} checkpoint found in {ckpt_dir}")


def load_conditions_from_data(filepath):
    with open(filepath, "rb") as f:
        d = pickle.load(f)
    bc, mc = d["baseCompression"], d["MagCompression"]
    base = np.concatenate([bc["F"], bc["Stress"]], axis=0).astype(np.float64)
    mag = np.concatenate([mc["F"], mc["Stress"]], axis=0).astype(np.float64)
    return base, mag


def save_results(result, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(result["om_mask"], cmap="gray")
        axes[0].set_title("Material Mask"); axes[0].axis("off")
        im1 = axes[1].imshow(result["om_angle"], cmap="hsv")
        axes[1].set_title("Orientation Angle"); axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        im2 = axes[2].imshow(result["br_x"], cmap="RdBu_r")
        axes[2].set_title("Br_x"); axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        im3 = axes[3].imshow(result["br_y"], cmap="RdBu_r")
        axes[3].set_title("Br_y"); axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
        plt.suptitle(name, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_overview.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visual -> {name}_overview.png")
    except ImportError:
        print("  [Tip] pip install matplotlib for visualisation")
    np.savez_compressed(os.path.join(output_dir, f"{name}_results.npz"),
                        **{k: v for k, v in result.items()})
    print(f"  Data -> {name}_results.npz")


def main():
    parser = argparse.ArgumentParser(description="Generate OM and Br from trained model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--base_cond", type=str, default=None)
    parser.add_argument("--mag_cond", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pipe = TrainedPipeline.load(args.model_dir, device=args.device)

    tasks = []
    if args.input_file:
        name = os.path.splitext(os.path.basename(args.input_file))[0]
        tasks.append((name, *load_conditions_from_data(args.input_file)))
    elif args.input_dir:
        for fp in sorted(glob.glob(os.path.join(args.input_dir, "*.data"))):
            name = os.path.splitext(os.path.basename(fp))[0]
            tasks.append((name, *load_conditions_from_data(fp)))
        print(f"Found {len(tasks)} .data files")
    elif args.base_cond and args.mag_cond:
        tasks.append(("custom", np.load(args.base_cond), np.load(args.mag_cond)))
    else:
        print("Provide: --input_file, --input_dir, or --base_cond + --mag_cond")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating {len(tasks)} inputs x {args.num_samples} samples...\n")

    for i, (name, base_c, mag_c) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {name} (base: {base_c.shape[1]} steps, mag: {mag_c.shape[1]} steps)")
        results = pipe.generate(base_c, mag_c, args.num_samples, args.seed)
        for j, r in enumerate(results):
            sname = f"{name}_s{j}" if args.num_samples > 1 else name
            save_results(r, sname, args.output_dir)

    print(f"\nDone! -> {args.output_dir}")


if __name__ == "__main__":
    main()
