"""
test_model.py - Model testing with synthetic inputs and GT comparison
======================================================================
Usage:
    python test_model.py --model_dir ./output --data_dir ./data --output_dir ./test_results
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Synthetic stress-strain curve generator
# ============================================================
def make_synthetic_compression_curve(max_strain=0.08, peak_stress=-35.0,
                                      n_timesteps=40, stiffness_factor=1.0,
                                      noise_level=0.0):
    t_norm = np.linspace(0, 1, n_timesteps)
    strain_path = max_strain * np.tanh(3.0 * t_norm) / np.tanh(3.0)

    F11 = 1.0 - strain_path
    F12 = -strain_path
    F21 = -strain_path
    F22 = 1.0 - strain_path
    F = np.stack([F11, F12, F21, F22], axis=0)

    S22 = peak_stress * stiffness_factor * (strain_path / max_strain) ** 1.3
    S11 = np.zeros(n_timesteps)
    S12 = np.zeros(n_timesteps)
    S21 = 0.08 * S22
    Stress = np.stack([S11, S12, S21, S22], axis=0)

    if noise_level > 0:
        F += np.random.randn(*F.shape) * noise_level * max_strain
        Stress += np.random.randn(*Stress.shape) * noise_level * abs(peak_stress)
        F[:, 0] = [1, 0, 0, 1]
        Stress[:, 0] = 0

    return F, Stress


def make_synthetic_test_case(label, max_strain=0.08, peak_stress_base=-35.0,
                              peak_stress_mag=-36.0, stiffness_base=1.0,
                              stiffness_mag=1.05, n_timesteps_base=40,
                              n_timesteps_mag=35):
    F_base, S_base = make_synthetic_compression_curve(
        max_strain, peak_stress_base, n_timesteps_base, stiffness_base, 0.002)
    F_mag, S_mag = make_synthetic_compression_curve(
        max_strain * 1.02, peak_stress_mag, n_timesteps_mag, stiffness_mag, 0.002)
    return (np.concatenate([F_base, S_base], axis=0),
            np.concatenate([F_mag, S_mag], axis=0))


SYNTHETIC_CASES = {
    "soft_material": {
        "desc": "Soft material: low stiffness, small stress",
        "params": dict(max_strain=0.06, peak_stress_base=-20.0,
                       peak_stress_mag=-22.0, stiffness_base=0.7, stiffness_mag=0.75),
    },
    "stiff_material": {
        "desc": "Stiff material: high stiffness, large stress",
        "params": dict(max_strain=0.10, peak_stress_base=-45.0,
                       peak_stress_mag=-50.0, stiffness_base=1.3, stiffness_mag=1.5),
    },
    "strong_mag_coupling": {
        "desc": "Strong magnetic coupling: field significantly alters response",
        "params": dict(max_strain=0.08, peak_stress_base=-30.0,
                       peak_stress_mag=-42.0, stiffness_base=1.0, stiffness_mag=1.4),
    },
    "typical_sample": {
        "desc": "Typical sample: close to training data sample_1",
        "params": dict(max_strain=0.076, peak_stress_base=-34.0,
                       peak_stress_mag=-35.5, stiffness_base=1.0, stiffness_mag=1.05),
    },
}


# ============================================================
# Visualisation
# ============================================================
def plot_condition_curves(base_cond, mag_cond, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strain_base = 1.0 - base_cond[0]
    stress_base = base_cond[7]
    strain_mag = 1.0 - mag_cond[0]
    stress_mag = mag_cond[7]
    axes[0].plot(strain_base, stress_base, 'b-o', markersize=2, label='Base (no field)')
    axes[0].plot(strain_mag, stress_mag, 'r-s', markersize=2, label='Mag (with field)')
    axes[0].set_xlabel('Engineering Strain')
    axes[0].set_ylabel('Stress S22')
    axes[0].set_title('Stress-Strain Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    labels = ['F11','F12','F21','F22','S11','S12','S21','S22']
    y_labels = [f'Base {l}' for l in labels] + [f'Mag {l}' for l in labels]
    max_len = max(base_cond.shape[1], mag_cond.shape[1])
    from inverse_design_diffusion import resample_curves
    base_rs = resample_curves(base_cond, max_len)
    mag_rs = resample_curves(mag_cond, max_len)
    combined_rs = np.concatenate([base_rs, mag_rs], axis=0)
    im = axes[1].imshow(combined_rs, aspect='auto', cmap='RdBu_r')
    axes[1].set_yticks(range(16))
    axes[1].set_yticklabels(y_labels, fontsize=7)
    axes[1].set_xlabel('Timestep')
    axes[1].set_title('All Condition Channels')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_generation_results(results_list, title, save_path, gt=None):
    n_rows = len(results_list) + (1 if gt is not None else 0)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_titles = ['Material Mask', 'Orientation Angle', 'Br_x', 'Br_y']
    cmaps = ['gray', 'hsv', 'RdBu_r', 'RdBu_r']
    for i, res in enumerate(results_list):
        data = [res['om_mask'], res['om_angle'], res['br_x'], res['br_y']]
        row_label = f'Sample {i+1}' if len(results_list) > 1 else 'Generated'
        for j in range(4):
            im = axes[i, j].imshow(data[j], cmap=cmaps[j])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11)
            axes[i, j].axis('off')
            if j > 0:
                plt.colorbar(im, ax=axes[i, j], fraction=0.046)
        axes[i, 0].set_ylabel(row_label, fontsize=11, rotation=0, labelpad=60, va='center')
    if gt is not None:
        row_i = len(results_list)
        gt_data = [gt['om_mask'], gt['om_angle'], gt['br_x'], gt['br_y']]
        for j in range(4):
            im = axes[row_i, j].imshow(gt_data[j], cmap=cmaps[j])
            axes[row_i, j].axis('off')
            if j > 0:
                plt.colorbar(im, ax=axes[row_i, j], fraction=0.046)
        axes[row_i, 0].set_ylabel('Ground Truth', fontsize=11, fontweight='bold',
                                    rotation=0, labelpad=60, va='center', color='green')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_ground_truth(data_path, image_size=128):
    import torch
    import torch.nn.functional as Fn
    with open(data_path, 'rb') as f:
        d = pickle.load(f)
    om = d['orientationMap'].copy()
    mask = (~np.isnan(om)).astype(np.float32)
    angle = np.nan_to_num(om, nan=0.0) * mask
    om_2ch = np.stack([mask, angle], axis=0)
    om_t = Fn.interpolate(torch.from_numpy(om_2ch).unsqueeze(0).float(),
                           size=image_size, mode='bilinear', align_corners=False).squeeze(0).numpy()
    br_flat = d['Br']
    br = np.stack([br_flat[0::2].reshape(256, 256), br_flat[1::2].reshape(256, 256)], axis=0)
    br_t = Fn.interpolate(torch.from_numpy(br).unsqueeze(0).float(),
                           size=image_size, mode='bilinear', align_corners=False).squeeze(0).numpy()
    return {'om_mask': om_t[0], 'om_angle': om_t[1], 'br_x': br_t[0], 'br_y': br_t[1]}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Test model with synthetic and real inputs")
    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import TrainedPipeline, load_conditions_from_data
    pipe = TrainedPipeline.load(args.model_dir, device=args.device)

    # Test 1: Synthetic cases
    print("\n" + "=" * 65)
    print("Test 1: Synthetic stress-strain curves")
    print("=" * 65)
    for case_name, case_info in SYNTHETIC_CASES.items():
        print(f"\n--- {case_name}: {case_info['desc']} ---")
        base_c, mag_c = make_synthetic_test_case(case_name, **case_info['params'])
        print(f"  Input: base ({base_c.shape[1]} steps), mag ({mag_c.shape[1]} steps)")
        plot_condition_curves(base_c, mag_c, f"Input: {case_name} - {case_info['desc']}",
                              os.path.join(args.output_dir, f"input_{case_name}.png"))
        results = pipe.generate(base_c, mag_c, num_samples=args.num_samples, seed=args.seed)
        plot_generation_results(results, f"Generated: {case_name} - {case_info['desc']}",
                                os.path.join(args.output_dir, f"output_{case_name}.png"))
        for i, r in enumerate(results):
            np.savez_compressed(os.path.join(args.output_dir, f"{case_name}_sample{i}.npz"), **r)
        print(f"  Generated {len(results)} samples")

    # Test 2: Real data GT comparison
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.data")))
    if data_files:
        print("\n" + "=" * 65)
        print("Test 2: Real data Ground Truth comparison")
        print("=" * 65)
        indices = [0, len(data_files) // 2, len(data_files) - 1]
        for idx in indices:
            fpath = data_files[idx]
            fname = os.path.splitext(os.path.basename(fpath))[0]
            print(f"\n--- {fname} ---")
            base_c, mag_c = load_conditions_from_data(fpath)
            gt = load_ground_truth(fpath, image_size=pipe.image_size)
            plot_condition_curves(base_c, mag_c, f"Real Input: {fname}",
                                  os.path.join(args.output_dir, f"input_real_{fname}.png"))
            results = pipe.generate(base_c, mag_c, num_samples=args.num_samples, seed=args.seed)
            plot_generation_results(results, f"Real: {fname} - Generated vs GT",
                                    os.path.join(args.output_dir, f"output_real_{fname}.png"), gt=gt)
            print(f"  Generated {len(results)} samples + GT comparison")

    # Test 3: Diversity test
    print("\n" + "=" * 65)
    print("Test 3: Diversity (same input, different seeds)")
    print("=" * 65)
    base_c, mag_c = make_synthetic_test_case("diversity", **SYNTHETIC_CASES["typical_sample"]["params"])
    diversity_results = []
    for seed in [0, 42, 123, 999]:
        r = pipe.generate(base_c, mag_c, num_samples=1, seed=seed)
        diversity_results.append(r[0])
    plot_generation_results(diversity_results,
                            "Diversity Test: Same Input, Seeds=[0, 42, 123, 999]",
                            os.path.join(args.output_dir, "diversity_test.png"))
    print(f"  4 seeds complete")

    print(f"\nAll tests complete! -> {args.output_dir}/")


if __name__ == "__main__":
    main()
