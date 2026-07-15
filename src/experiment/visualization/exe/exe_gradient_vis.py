# exe_gradient_vis.py
# Save log-gradient maps for each pyramid level per data in data/data_one.
# Output: test/gradient_pyramid/<dataset_id>/<img_name>/

import os
import cv2
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths and import packages
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
project_root = current_file.parents[4]  # .../Fattal_python root
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from fattal.fattal_tmo import createGaussianPyramids, calculate_gradient_mag
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# --- Path setup ---
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "gradient_pyramid"

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

opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
pre_gamma = 1.0
post_gamma = 1.0

MSIZE = 8 if fftsolver else 32


def visualize_log_gradient_map(log_grad_map, level, save_path, title=None):
    """Visualize a single log-gradient map as a colormap and save to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    im = ax.imshow(log_grad_map, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Log-Gradient Magnitude (ln(G + 1e-4))', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Log-Gradient Map - Pyramid Level {level}', fontsize=14, fontweight='bold')

    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)

    h, w = log_grad_map.shape
    stats_text = (
        f'Size: {w}x{h}  |  '
        f'Min: {log_grad_map.min():.4f}  |  '
        f'Max: {log_grad_map.max():.4f}  |  '
        f'Mean: {log_grad_map.mean():.4f}'
    )
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_image(img_path, dataset_id, opt_alpha, opt_beta):
    """Process a single HDR image: compute and save log-gradient maps for each pyramid level."""
    file_name = img_path.stem

    # Output folder: test/gradient_pyramid/<dataset_id>/<img_name>/
    save_dir = OUTPUT_DIR / dataset_id / file_name
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # Compute luminance (Rec. 709)
    Yr = 0.2126 * R_pre + 0.7152 * G_pre + 0.0722 * B_pre

    # Log-space transform
    maxLum = np.max(Yr)
    H = np.log(100.0 * Yr / maxLum + 1e-4)

    # Build Gaussian pyramid
    h, w = H.shape
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0:
        nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    print(f"  Pyramid levels: {nlevels} (original size: {w}x{h})")

    # Compute and save log-gradient map for each level
    for k in range(nlevels):
        G = calculate_gradient_mag(pyramids[k], k)
        
        # Apply logarithm to gradient magnitude map
        log_G = np.log(G + 1e-4)
        
        ph, pw = pyramids[k].shape

        # Save log-gradient map
        save_name = f"log_gradient_level_{k:02d}_{pw}x{ph}.png"
        save_path = save_dir / save_name
        title = f"Log-Gradient Map - Level {k}/{nlevels-1} (size={pw}x{ph})"
        visualize_log_gradient_map(log_G, k, str(save_path), title=title)

        print(f"    Level {k:2d}: saved log-gradient map ({pw}x{ph})")


def main():
    utils.start_timer()
    utils.print_elapsed("Log-gradient map visualization start")

    print(f"Input Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    for dataset_id in sorted(DATASET_PARAMS.keys(), key=int):
        params = DATASET_PARAMS[dataset_id]
        opt_alpha = params["alpha"]
        opt_beta = params["beta"]

        dataset_path = DATA_DIR / dataset_id
        if not dataset_path.exists():
            print(f"  [SKIP] Dataset folder {dataset_path} does not exist.")
            continue

        hdr_files = list(dataset_path.glob("*.hdr"))
        if not hdr_files:
            print(f"  [SKIP] No .hdr files in {dataset_path}.")
            continue

        print(f"\n--- Dataset [{dataset_id}] Processing (alpha={opt_alpha}, beta={opt_beta}) ---")
        for hdr_path in hdr_files:
            print(f"  Image: {hdr_path.name}")
            process_image(hdr_path, dataset_id, opt_alpha, opt_beta)

    utils.print_elapsed("Log-gradient map visualization complete")
    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
