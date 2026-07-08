# exe_attenuation_vis.py
# Visualize attenuation maps for each pyramid level per data in data/data_one.
# Output: test/attenuation/<data_folder_name>/attenuation_level_<k>.png

import os
import sys
import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# --- Path setup ---
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
project_root = current_file.parents[4]  # .../Fattal_python root
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from fattal.fattal_tmo import createGaussianPyramids, calculate_level_scaling_factor, calculate_attenuation, pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
from exe.config.config import INPUT_DIR, OUTPUT_DIR as CONFIG_OUTPUT_DIR, PARAM_GRID
import utils.utils as utils

DATA_DIR = Path(INPUT_DIR)
OUTPUT_DIR = Path(CONFIG_OUTPUT_DIR)

# --- Parameters (from config.py) ---
opt_alpha = PARAM_GRID['opt_alpha'][0]
opt_beta = PARAM_GRID['opt_beta'][0]
opt_noise = PARAM_GRID['opt_noise'][0]
newfattal = PARAM_GRID['newfattal'][0]
fftsolver = PARAM_GRID['fftsolver'][0]
detail_level = PARAM_GRID['detail_level'][0]
pre_gamma = PARAM_GRID['pre_gamma'][0]
post_gamma = PARAM_GRID['post_gamma'][0]

# --- Visualization options ---
# Y축 (컬러바 값의 범위)을 전체 피라미드 레벨에 대해 동일하게 고정할지 여부
FIX_Y_AXIS = False

MSIZE = 8 if fftsolver else 32


def visualize_attenuation_map(att_map, level, save_path, title=None, vmin=None, vmax=None):
    """Visualize a single attenuation map as a colormap and save to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    im = ax.imshow(att_map, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attenuation Value', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif level >= 0:
        ax.set_title(f'Attenuation Map - Pyramid Level {level}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Final Combined Attenuation Map', fontsize=14, fontweight='bold')

    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)

    h, w = att_map.shape
    stats_text = (
        f'Size: {w}x{h}  |  '
        f'Min: {att_map.min():.4f}  |  '
        f'Max: {att_map.max():.4f}  |  '
        f'Mean: {att_map.mean():.4f}'
    )
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_attenuation_opencv(att_map, save_path):
    """Save attenuation map as a grayscale image using OpenCV."""
    # Normalize to 0-255
    att_min = att_map.min()
    att_max = att_map.max()
    if att_max - att_min > 1e-8:
        normalized = ((att_map - att_min) / (att_max - att_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(att_map, dtype=np.uint8)

    cv2.imwrite(str(save_path), normalized)



def process_single_image(img_path, data_folder_name):
    """Process a single HDR image: compute and save attenuation maps for each pyramid level."""
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    # Output folder
    save_dir = OUTPUT_DIR if OUTPUT_DIR.name == data_folder_name else OUTPUT_DIR / data_folder_name
    os.makedirs(save_dir, exist_ok=True)

    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [SKIP] Cannot read image: {img_path}")
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
    print(f"  Pyramid levels: {nlevels}  (original size: {w}x{h})")

    # Noise calculation (same logic as pfstmo_fattal02)
    noise = opt_noise if opt_noise > 0 else opt_alpha * 0.01

    # Compute scaling factors for all levels
    scaling_factors = [None] * nlevels
    for k in range(nlevels):
        if k >= detail_level or k == nlevels - 1 or not newfattal:
            att = calculate_level_scaling_factor(pyramids[k], k, opt_alpha, opt_beta, noise)
            scaling_factors[k] = att

    # Compute Final Combined Attenuation Map
    final_att = calculate_attenuation(scaling_factors, pyramids, nlevels, newfattal)

    # Calculate global min/max across all valid maps for unified Y-axis (colorbar) range if FIX_Y_AXIS is True
    if FIX_Y_AXIS:
        all_maps = [s for s in scaling_factors if s is not None] + [final_att]
        global_vmin = min(m.min() for m in all_maps)
        global_vmax = max(m.max() for m in all_maps)
    else:
        global_vmin = None
        global_vmax = None

    # Visualize attenuation for each level
    for k in range(nlevels):
        if scaling_factors[k] is not None:
            att = scaling_factors[k]
            ph, pw = pyramids[k].shape
            save_name = f"attenuation_level_{k:02d}_{pw}x{ph}.png"
            save_path = save_dir / save_name

            title = (
                f'Attenuation Map - Level {k}/{nlevels-1}\n'
                f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise:.4f}, size={pw}x{ph})'
            )
            visualize_attenuation_map(att, k, str(save_path), title=title, vmin=global_vmin, vmax=global_vmax)

            # OpenCV raw image save
            cv_save_name = f"cv_attenuation_level_{k:02d}_{pw}x{ph}.png"
            cv_save_path = save_dir / cv_save_name
            save_attenuation_opencv(att, cv_save_path)

            print(f"    Level {k:2d}: {pw:5d}x{ph:<5d}  -> saved: {save_path.name}, {cv_save_path.name}")
        else:
            print(f"    Level {k:2d}: (skipped, below detail_level={detail_level})")

    # Visualize Final Combined Attenuation Map
    fh, fw = final_att.shape
    final_save_name = f"attenuation_final_combined_{fw}x{fh}.png"
    final_save_path = save_dir / final_save_name

    final_title = (
        f'Final Combined Attenuation Map (Phi)\n'
        f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise:.4f}, detail_level={detail_level}, size={fw}x{fh})'
    )
    visualize_attenuation_map(final_att, -1, str(final_save_path), title=final_title, vmin=global_vmin, vmax=global_vmax)

    cv_final_save_name = f"cv_attenuation_final_combined_{fw}x{fh}.png"
    cv_final_save_path = save_dir / cv_final_save_name
    save_attenuation_opencv(final_att, cv_final_save_path)

    print(f"    [FINAL ATT] Saved final attenuation map: {final_save_path.name}, {cv_final_save_path.name}")

    # --- Tone Mapping & Save ---
    print(f"  [TMO] Starting tone mapping...")
    L_out = pfstmo_fattal02(
        Yr,
        opt_alpha, opt_beta, opt_noise,
        newfattal, fftsolver, detail_level
    )

    out_img_8bit = (np.clip(L_out, 0.0, 1.0) * 255.0).astype(np.uint8)

    tmo_save_name = f"{file_name}_tonemapped.png"
    tmo_save_path = save_dir / tmo_save_name
    cv2.imwrite(str(tmo_save_path), out_img_8bit)
    print(f"    [TMO SAVE] Tone mapping result saved to: {tmo_save_path.name}")


def main():
    utils.start_timer()
    utils.print_elapsed("Attenuation visualization start")

    if not DATA_DIR.exists():
        print(f"Data directory does not exist: {DATA_DIR}")
        return

    # Check if DATA_DIR directly contains .hdr files
    direct_hdr_files = list(DATA_DIR.glob('*.hdr'))
    if direct_hdr_files:
        print(f"Processing single data folder: {DATA_DIR.name}")
        for hdr_file in direct_hdr_files:
            print(f"\n[Data {DATA_DIR.name}] Processing: {hdr_file.name}")
            process_single_image(hdr_file, DATA_DIR.name)
    else:
        # Scan sub-folders in DATA_DIR (1, 2, 3, ...)
        data_folders = sorted([
            d for d in DATA_DIR.iterdir() if d.is_dir()
        ], key=lambda x: x.name)

        if not data_folders:
            print(f"No data folders or .hdr files found in: {DATA_DIR}")
            return

        print(f"Found {len(data_folders)} data folder(s).\n")

        for folder in data_folders:
            hdr_files = list(folder.glob('*.hdr'))
            if not hdr_files:
                print(f"[{folder.name}] No .hdr files - skipping")
                continue

            for hdr_file in hdr_files:
                print(f"\n[Data {folder.name}] Processing: {hdr_file.name}")
                process_single_image(hdr_file, folder.name)

    utils.print_elapsed("Attenuation visualization complete")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
