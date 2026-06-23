# exe_bilateral_experiment.py
# Run expanded parameter sweep for bilateral filtering and k-level exponent parameters.
# Output is saved to test/bilateral_param_sweep/dataset_<folder_name>/

import os
import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add project src and experiment paths to sys.path
current_file = Path(__file__).resolve()
src_dir = current_file.parents[2]
experiment_dir = current_file.parents[1]
project_root = current_file.parents[3]
sys.path.append(str(src_dir))
sys.path.append(str(src_dir / "exe"))
sys.path.append(str(experiment_dir))

from fattal.fattal_tmo import pfstmo_fattal02
from fattal.fattal_tmo_bilateral import pfstmo_fattal02_bilateral
from processing.gamma_correction import Frame, apply_gamma_frame
from config.config import PARAM_GRID
import utils.utils as utils

DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "bilateral_strong_kge3"

# --- Dataset parameter configuration ---
DATASET_PARAMS = {
    "1": {"alpha": 0.9, "beta": 0.82},
    "2": {"alpha": 0.9, "beta": 0.80},
    "3": {"alpha": 0.9, "beta": 0.81},
    "4": {"alpha": 0.9, "beta": 0.84},
    "5": {"alpha": 0.3, "beta": 0.93},
    "6": {"alpha": 0.9, "beta": 0.81},
    "7": {"alpha": 0.9, "beta": 0.80}
}

# --- Fixed parameters (same across all datasets) ---
opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
HE_weight = 0.0
pre_gamma = 1.0
post_gamma = 1.0

def process_single_image(img_path, folder_name, alpha, base_beta):
    """Run parameter sweep for one image and save results to dataset folder."""
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = OUTPUT_DIR / f"dataset_{folder_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Processing] Folder {folder_name}: {img_path.name}")
    print(f"  Base Parameters -> alpha: {alpha}, beta: {base_beta}")

    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [ERROR] Cannot read image: {img_path}")
        return

    is_grayscale = (img.ndim == 2)
    if is_grayscale:
        img = np.stack([img, img, img], axis=-1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Pre-gamma correction
    pre_frame = Frame(R, G, B)
    apply_gamma_frame(pre_frame, pre_gamma)
    R_pre = pre_frame.x_channel.data
    G_pre = pre_frame.y_channel.data
    B_pre = pre_frame.z_channel.data

    # Define sweep ranges
    # 1. Beta shifts: base_beta +- 0.06 with 0.02 step
    beta_shifts = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
    betas = [np.round(base_beta + shift, 2) for shift in beta_shifts]

    # 2. Strong Bilateral filter settings: (d, sigmaColor, sigmaSpace)
    bilateral_settings = [
        (9, 0.15, 6.0),
        (15, 0.30, 10.0),
        (21, 0.45, 15.0)
    ]

    opt_saturation = 1.0 if is_grayscale else 0.8
    he_weight_clipped = np.clip(HE_weight, 0.0, 1.0)

    total_runs = len(betas) * len(bilateral_settings)
    run_idx = 0

    # Start sweep loop
    for beta in betas:
        for bd, bsc, bss in bilateral_settings:
            run_idx += 1
            # Format parameters for filename
            # Example: _a0.9_b0.82_bd9_sc0.15_ss6.0.png
            param_suffix = f"a{alpha}_b{beta}_bd{bd}_sc{bsc}_ss{bss}"
            
            # Execute tone mapping with fixed tmp_low = 1.0, tmp_high = 1.0
            noise = opt_noise if opt_noise > 0 else alpha * 0.01
            R_out, G_out, B_out = pfstmo_fattal02_bilateral(
                R_pre, G_pre, B_pre,
                alpha, beta, opt_saturation, noise,
                newfattal, fftsolver, detail_level, he_weight_clipped,
                bilateral_d=bd, bilateral_sigma_color=bsc, bilateral_sigma_space=bss,
                tmp_low=1.0, tmp_high=1.0, k_threshold=3
            )

            # Post-gamma and conversion
            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, post_gamma)
            out_rgb = np.clip(np.stack((post_frame.x_channel.data, post_frame.y_channel.data, post_frame.z_channel.data), axis=-1), 0.0, 1.0)
            out_bgr = cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if is_grayscale:
                out_bgr = out_bgr[:, :, 0]

            save_path = save_dir / f"{file_name}_{param_suffix}.png"
            cv2.imwrite(str(save_path), out_bgr)
            
            if run_idx % 7 == 0 or run_idx == total_runs:
                print(f"    [Progress] Dataset {folder_name}: Completed {run_idx}/{total_runs} combinations...")

def main():
    utils.start_timer()
    utils.print_elapsed("Bilateral param sweep start")

    data_folders = sorted([
        d for d in DATA_DIR.iterdir() if d.is_dir()
    ], key=lambda x: x.name)

    if not data_folders:
        print(f"No data folders found in: {DATA_DIR}")
        return

    print(f"Found {len(data_folders)} data folder(s) for the sweep experiment.")

    for folder in data_folders:
        hdr_files = list(folder.glob('*.hdr'))
        if not hdr_files:
            print(f"[{folder.name}] No .hdr files - skipping")
            continue

        # Get dataset specific parameters
        params = DATASET_PARAMS.get(folder.name, {"alpha": 0.9, "beta": 0.81})
        alpha = params["alpha"]
        base_beta = params["beta"]

        for hdr_file in hdr_files:
            process_single_image(hdr_file, folder.name, alpha, base_beta)

    utils.print_elapsed("Bilateral param sweep complete")
    print(f"\nAll sweep results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
