# exe_attenuation_vis.py
# Visualize attenuation maps for each pyramid level per data in data/data_one.
# Output: test/attenuation/<data_folder_name>/attenuation_level_<k>.png

import os
import cv2
import numpy as np
import glob
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fattal.fattal_tmo import createGaussianPyramids, calculate_level_attenuation, pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
from config.config import PARAM_GRID
import utils.utils as utils
from pathlib import Path

# --- Path setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Fattal_python root
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "attenuation_AT_4_all"

# --- Parameters (from config.py) ---
opt_alpha = PARAM_GRID['opt_alpha'][0]
opt_beta = PARAM_GRID['opt_beta'][0]
opt_noise = PARAM_GRID['opt_noise'][0]
newfattal = PARAM_GRID['newfattal'][0]
fftsolver = PARAM_GRID['fftsolver'][0]
detail_level = PARAM_GRID['detail_level'][0]
pre_gamma = PARAM_GRID['pre_gamma'][0]
HE_weight = PARAM_GRID['HE_weight'][0]
post_gamma = PARAM_GRID['post_gamma'][0]

MSIZE = 8 if fftsolver else 32


def visualize_attenuation_map(att_map, level, save_path, title=None):
    """Visualize a single attenuation map as a colormap and save to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    im = ax.imshow(att_map, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attenuation Value', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Attenuation Map - Pyramid Level {level}', fontsize=14, fontweight='bold')

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

    # Output folder: test/attenuation/<data_folder_name>/
    save_dir = OUTPUT_DIR / data_folder_name
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

    # Compute and visualize attenuation for each level
    for k in range(nlevels):
        # Condition matching fattal_tmo.py lines 167-170
        if k >= detail_level or k == nlevels - 1 or not newfattal:
            att = calculate_level_attenuation(pyramids[k], k, opt_alpha, opt_beta, noise)

            ph, pw = pyramids[k].shape
            save_name = f"attenuation_level_{k:02d}_{pw}x{ph}.png"
            save_path = save_dir / save_name

            title = (
                f'Attenuation Map - Level {k}/{nlevels-1}\n'
                f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise:.4f}, size={pw}x{ph})'
            )
            visualize_attenuation_map(att, k, str(save_path), title=title)

            # OpenCV raw image save
            cv_save_name = f"cv_attenuation_level_{k:02d}_{pw}x{ph}.png"
            cv_save_path = save_dir / cv_save_name
            save_attenuation_opencv(att, cv_save_path)

            print(f"    Level {k:2d}: {pw:5d}x{ph:<5d}  -> saved: {save_path.name}, {cv_save_path.name}")
        else:
            print(f"    Level {k:2d}: (skipped, below detail_level={detail_level})")

    # --- Tone Mapping & Save ---
    # 그레이스케일일 경우 채도 복원 파라미터 무력화
    opt_saturation = 1.0 if is_grayscale else 0.8
    he_weight_clipped = np.clip(HE_weight, 0.0, 1.0)

    print(f"  [TMO] Starting tone mapping...")
    R_out, G_out, B_out = pfstmo_fattal02(
        R_pre, G_pre, B_pre,
        opt_alpha, opt_beta, opt_saturation, opt_noise,
        newfattal, fftsolver, detail_level, he_weight_clipped
    )

    # 후처리 감마 보정
    post_frame = Frame(R_out, G_out, B_out)
    apply_gamma_frame(post_frame, post_gamma)
    R_final = post_frame.x_channel.data
    G_final = post_frame.y_channel.data
    B_final = post_frame.z_channel.data

    # 채널 병합 및 포맷 변환
    out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
    out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
    out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

    # 원본이 그레이스케일이면 단채널로 변환
    if is_grayscale:
        out_img_bgr = out_img_bgr[:, :, 0]

    tmo_save_name = f"{file_name}_tonemapped.png"
    tmo_save_path = save_dir / tmo_save_name
    cv2.imwrite(str(tmo_save_path), out_img_bgr)
    print(f"    [TMO SAVE] Tone mapping result saved to: {tmo_save_path.name}")


def main():
    utils.start_timer()
    utils.print_elapsed("Attenuation visualization start")

    # Scan sub-folders in data/data_one (1, 2, 3, ...)
    data_folders = sorted([
        d for d in DATA_DIR.iterdir() if d.is_dir()
    ], key=lambda x: x.name)

    if not data_folders:
        print(f"No data folders found in: {DATA_DIR}")
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
