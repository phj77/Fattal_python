import os
import cv2
import numpy as np
import sys
import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup path to allow importing from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiment.fattal.fattal_tmo_exponent import createGaussianPyramids, calculate_level_attenuation, pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# --- Path setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # Fattal_python root
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "exponent_experiment"

# --- Dataset parameter configuration ---
dataset_configs = {
    1: {"alpha": 0.9, "beta": 0.82, "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
    2: {"alpha": 0.9, "beta": 0.8,  "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
    3: {"alpha": 0.9, "beta": 0.81, "row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
    4: {"alpha": 0.9, "beta": 0.84, "row": 974,  "highlight": [[457, 475], [590, 607]]},
    5: {"alpha": 0.3, "beta": 0.93, "row": 1170, "highlight": [[2073, 2188]]},
    6: {"alpha": 0.9, "beta": 0.81, "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
    7: {"alpha": 0.9, "beta": 0.8,  "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
}

# --- Fixed parameters ---
opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
HE_weight = 0.0
pre_gamma = 1.0
post_gamma = 1.0
MSIZE = 8 if fftsolver else 32

def main():
    utils.start_timer()
    utils.print_elapsed("Exponent scaling experiment start")

    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}\n")

    # Experiment parameter lists
    target_k_list = [3, 4, 5]
    tmp_list = [1.2, 1.5, 2.0]

    for k in sorted(dataset_configs.keys()):
        config = dataset_configs[k]
        opt_alpha = config["alpha"]
        opt_beta = config["beta"]
        scanline_row = config["row"]
        highlight_ranges = config["highlight"]

        input_dir = DATA_DIR / str(k)
        if not input_dir.exists():
            print(f"  [SKIP] Dataset folder {input_dir} does not exist.")
            continue

        hdr_files = list(input_dir.glob("*.hdr"))
        if not hdr_files:
            print(f"  [SKIP] No .hdr files in {input_dir}.")
            continue

        for img_path in hdr_files:
            file_name = img_path.stem
            print(f"\n--- Dataset [{k}] Processing: {file_name} ---")

            # Load HDR image
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"    [ERROR] Cannot read image: {img_path}")
                continue

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

            # Build Gaussian pyramid to verify level availability and compute specific attenuation maps
            h_h, w_h = H.shape
            mins = min(w_h, h_h)
            nlevels = 0
            temp_mins = mins
            while temp_mins >= MSIZE:
                nlevels += 1
                temp_mins //= 2
            if nlevels == 0:
                nlevels = 1

            pyramids = createGaussianPyramids(H, nlevels)
            noise = opt_noise if opt_noise > 0 else opt_alpha * 0.01

            for target_k in target_k_list:
                if target_k >= nlevels:
                    print(f"    [SKIP] target_k={target_k} is out of bounds (nlevels={nlevels}).")
                    continue

                for tmp in tmp_list:
                    print(f"    Running experiment: target_k={target_k}, tmp={tmp:.1f}")

                    # Define unique folder structure for this run
                    run_save_dir = OUTPUT_DIR / str(k) / f"k{target_k}_tmp{tmp:.1f}"
                    run_save_dir.mkdir(parents=True, exist_ok=True)

                    # 1. Compute and save Viridis colormap attenuation map at target_k
                    att_map = calculate_level_attenuation(pyramids[target_k], target_k, opt_alpha, opt_beta, noise, target_k, tmp)
                    
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    im = ax.imshow(att_map, cmap='viridis', aspect='auto')
                    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Attenuation Value', fontsize=12)
                    ax.set_title(
                        f'Attenuation Map - Level {target_k} (tmp={tmp:.1f})\n'
                        f'(alpha={opt_alpha}, beta={opt_beta}, size={att_map.shape[1]}x{att_map.shape[0]})',
                        fontsize=14, fontweight='bold'
                    )
                    ax.set_xlabel('Width (pixels)', fontsize=11)
                    ax.set_ylabel('Height (pixels)', fontsize=11)
                    
                    h_att, w_att = att_map.shape
                    stats_text = (
                        f'Size: {w_att}x{h_att}  |  '
                        f'Min: {att_map.min():.4f}  |  '
                        f'Max: {att_map.max():.4f}  |  '
                        f'Mean: {att_map.mean():.4f}'
                    )
                    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes, fontsize=10, ha='center', va='top', color='gray')
                    plt.tight_layout()
                    att_save_path = run_save_dir / f"attenuation_k{target_k}_viridis.png"
                    plt.savefig(str(att_save_path), dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # 2. Run Tone Mapping and save scanlines automatically
                    opt_saturation = 1.0 if is_grayscale else 0.8
                    R_out, G_out, B_out = pfstmo_fattal02(
                        R_pre, G_pre, B_pre,
                        opt_alpha, opt_beta, opt_saturation, opt_noise,
                        newfattal, fftsolver, detail_level, HE_weight,
                        scanline_row=scanline_row, highlight_ranges=highlight_ranges,
                        save_dir=str(run_save_dir), target_k=target_k, tmp=tmp
                    )

                    # 3. Post-gamma correction
                    post_frame = Frame(R_out, G_out, B_out)
                    apply_gamma_frame(post_frame, post_gamma)
                    R_final = post_frame.x_channel.data
                    G_final = post_frame.y_channel.data
                    B_final = post_frame.z_channel.data

                    # 4. Save tone-mapped result image
                    out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
                    out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
                    out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
                    out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)
                    if is_grayscale:
                        out_img_bgr = out_img_bgr[:, :, 0]

                    result_img_path = run_save_dir / "result_img.png"
                    cv2.imwrite(str(result_img_path), out_img_bgr)
                    print(f"      Saved attenuation map and tone-mapped result to {run_save_dir}")

    utils.print_elapsed("Exponent scaling experiment completed")
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
