"""
train.py - Training launcher
==============================
Usage:
    python train.py --data_dir ./data --epochs1 500 --epochs2 500 --batch_size 4
"""

import os
import json
import csv
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm

from inverse_design_diffusion import (
    Config, MetamaterialDataset, EMAModel,
    StressStrainEncoder, OMAndMagEncoder,
    build_stage1_unet, build_stage2_unet,
)


def save_norm_stats(dataset, save_dir):
    stats = {
        "base_cond_min": dataset.base_cond_min.tolist(),
        "base_cond_max": dataset.base_cond_max.tolist(),
        "base_cond_range": dataset.base_cond_range.tolist(),
        "mag_cond_min": dataset.mag_cond_min.tolist(),
        "mag_cond_max": dataset.mag_cond_max.tolist(),
        "mag_cond_range": dataset.mag_cond_range.tolist(),
        "br_min": float(dataset.br_min),
        "br_max": float(dataset.br_max),
        "br_range": float(dataset.br_range),
        "om_min": float(dataset.om_min),
        "om_max": float(dataset.om_max),
        "om_range": float(dataset.om_range),
        "n_resample": dataset.n_resample,
    }
    path = os.path.join(save_dir, "norm_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalisation stats saved -> {path}")


def save_final_checkpoint(encoder, unet, ema_enc, ema_unet, scheduler, cfg_dict, save_path):
    torch.save({
        "encoder": ema_enc.state_dict(),
        "unet": ema_unet.state_dict(),
        "encoder_train": encoder.state_dict(),
        "unet_train": unet.state_dict(),
        "scheduler_config": scheduler.config,
        "cfg": cfg_dict,
    }, save_path)
    print(f"Final model saved (EMA) -> {save_path}")


def train_one_stage(stage_name, dataloader, encoder, unet, noise_scheduler,
                    num_epochs, num_diffusion_steps, lr, device, ckpt_dir,
                    log_writer, extract_fn, ema_decay=0.999):
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(unet.parameters()), lr=lr)

    warmup_epochs = max(num_epochs // 10, 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema_enc = EMAModel(encoder, decay=ema_decay)
    ema_unet = EMAModel(unet, decay=ema_decay)

    best_loss = float("inf")
    encoder.train()
    unet.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n = 0

        pbar = tqdm(dataloader, desc=f"[{stage_name}] Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            cond_embeds, target = extract_fn(batch, encoder, device)

            noise = torch.randn_like(target)
            t = torch.randint(0, num_diffusion_steps, (target.size(0),),
                              device=device).long()
            noisy = noise_scheduler.add_noise(target, noise, t)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = unet(noisy, t, encoder_hidden_states=cond_embeds).sample
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(unet.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema_enc.update(encoder)
            ema_unet.update(unet)

            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        lr_scheduler.step()
        avg = epoch_loss / max(n, 1)
        log_writer.writerow([stage_name, epoch+1, f"{avg:.8f}",
                             f"{optimizer.param_groups[0]['lr']:.2e}"])
        print(f"[{stage_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 100 == 0:
            p = os.path.join(ckpt_dir, f"{stage_name}_epoch{epoch+1}.pt")
            torch.save({"encoder": ema_enc.state_dict(),
                        "unet": ema_unet.state_dict(),
                        "epoch": epoch+1, "loss": avg}, p)
            print(f"  Checkpoint (EMA) -> {p}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"encoder": ema_enc.state_dict(),
                        "unet": ema_unet.state_dict(),
                        "epoch": epoch+1, "loss": avg},
                       os.path.join(ckpt_dir, f"{stage_name}_best.pt"))

    return encoder, unet, ema_enc, ema_unet


def stage1_extract(batch, encoder, device):
    base_cond, _, om, _ = batch
    return encoder(base_cond.to(device)), om.to(device)

def stage2_extract(batch, encoder, device):
    _, mag_cond, om, br = batch
    return encoder(om.to(device), mag_cond.to(device)), br.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train metamaterial inverse design model")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs1", type=int, default=500)
    parser.add_argument("--epochs2", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--n_resample", type=int, default=64)
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable rotation augmentation")
    parser.add_argument("--resume_stage", type=int, default=0)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--stage2_ckpt", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Augmentation: {'OFF' if args.no_augment else 'ON (4x rotation)'}")

    print("\nLoading dataset...")
    dataset = MetamaterialDataset(args.data_dir,
                                  image_size=args.image_size,
                                  n_resample=args.n_resample,
                                  augment=not args.no_augment)
    print(f"Total training samples: {len(dataset)}")

    save_norm_stats(dataset, args.output_dir)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "n_samples": len(dataset), "device": device}, f, indent=2)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device == "cuda"))

    log_file = open(os.path.join(log_dir, "training_log.csv"), "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["stage", "epoch", "loss", "lr"])

    cfg = Config()

    # ============ Stage 1 ============
    if args.resume_stage < 1:
        print("\n" + "=" * 60)
        print("Stage 1: baseCompression -> Orientation Map")
        n_steps = len(dataset) // args.batch_size * args.epochs1
        print(f"  {len(dataset)} samples, {args.epochs1} epochs, ~{n_steps} steps")
        print("=" * 60)

        s1_enc = StressStrainEncoder(
            in_rows=cfg.s1_condition_rows, in_cols=args.n_resample,
            cross_attention_dim=cfg.cross_attention_dim,
            seq_len=cfg.condition_seq_len).to(device)
        s1_unet = build_stage1_unet(cfg.cross_attention_dim).to(device)
        s1_sched = DDPMScheduler(num_train_timesteps=args.diffusion_steps,
                                 beta_schedule="squaredcos_cap_v2")

        if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
            print(f"Resuming from checkpoint: {args.stage1_ckpt}")
            ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
            s1_enc.load_state_dict(ck["encoder"])
            s1_unet.load_state_dict(ck["unet"])

        s1_enc, s1_unet, ema_enc1, ema_unet1 = train_one_stage(
            "stage1", dataloader, s1_enc, s1_unet, s1_sched,
            args.epochs1, args.diffusion_steps, args.lr, device,
            ckpt_dir, log_writer, stage1_extract, cfg.ema_decay)

        save_final_checkpoint(s1_enc, s1_unet, ema_enc1, ema_unet1, s1_sched, {
            "in_rows": cfg.s1_condition_rows, "in_cols": args.n_resample,
            "cross_attention_dim": cfg.cross_attention_dim,
            "seq_len": cfg.condition_seq_len,
            "image_size": args.image_size,
            "diffusion_steps": args.diffusion_steps,
            "n_resample": args.n_resample,
        }, os.path.join(args.output_dir, "stage1_final.pt"))
    else:
        print(f"\nSkipping Stage 1 training")
        assert args.stage1_ckpt, "Skipping Stage 1 requires --stage1_ckpt"

    # ============ Stage 2 ============
    if args.resume_stage < 2:
        print("\n" + "=" * 60)
        print("Stage 2: OM + MagCompression -> Br Field")
        print(f"  {len(dataset)} samples, {args.epochs2} epochs")
        print("=" * 60)

        s2_enc = OMAndMagEncoder(
            om_channels=2, mag_rows=cfg.s2_mag_condition_rows,
            mag_cols=args.n_resample,
            cross_attention_dim=cfg.cross_attention_dim,
            om_seq_len=cfg.condition_seq_len // 2,
            mag_seq_len=cfg.condition_seq_len // 2).to(device)
        s2_unet = build_stage2_unet(cfg.cross_attention_dim).to(device)
        s2_sched = DDPMScheduler(num_train_timesteps=args.diffusion_steps,
                                 beta_schedule="squaredcos_cap_v2")

        if args.stage2_ckpt and os.path.exists(args.stage2_ckpt):
            print(f"Resuming from checkpoint: {args.stage2_ckpt}")
            ck = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
            s2_enc.load_state_dict(ck["encoder"])
            s2_unet.load_state_dict(ck["unet"])

        s2_enc, s2_unet, ema_enc2, ema_unet2 = train_one_stage(
            "stage2", dataloader, s2_enc, s2_unet, s2_sched,
            args.epochs2, args.diffusion_steps, args.lr, device,
            ckpt_dir, log_writer, stage2_extract, cfg.ema_decay)

        save_final_checkpoint(s2_enc, s2_unet, ema_enc2, ema_unet2, s2_sched, {
            "om_channels": 2, "mag_rows": cfg.s2_mag_condition_rows,
            "mag_cols": args.n_resample,
            "cross_attention_dim": cfg.cross_attention_dim,
            "om_seq_len": cfg.condition_seq_len // 2,
            "mag_seq_len": cfg.condition_seq_len // 2,
            "image_size": args.image_size,
            "diffusion_steps": args.diffusion_steps,
            "n_resample": args.n_resample,
        }, os.path.join(args.output_dir, "stage2_final.pt"))

    log_file.close()
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
