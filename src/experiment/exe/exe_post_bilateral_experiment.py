# exe_post_bilateral_experiment.py
# Run parameter sweep for bilateral filtering on standard Fattal tone mapping results.
# Output is saved to test/post_bilateral_experiment/dataset_<name>/

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
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "post_bilateral_experiment"

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

def process_single_image(img_path, folder_name, alpha, beta):
    """Run post-TMO bilateral filtering sweep for one image and save results."""
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = OUTPUT_DIR / f"dataset_{folder_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Processing] Folder {folder_name}: {img_path.name}")
    print(f"  Parameters -> alpha: {alpha}, beta: {beta}")

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

    opt_saturation = 1.0 if is_grayscale else 0.8
    he_weight_clipped = np.clip(HE_weight, 0.0, 1.0)

    # 1. Run standard Fattal tone mapping
    R_out, G_out, B_out = pfstmo_fattal02(
        R_pre, G_pre, B_pre,
        alpha, beta, opt_saturation, opt_noise,
        newfattal, fftsolver, detail_level, he_weight_clipped
    )

    # Post-gamma and conversion
    post_frame = Frame(R_out, G_out, B_out)
    apply_gamma_frame(post_frame, post_gamma)
    out_rgb = np.clip(np.stack((post_frame.x_channel.data, post_frame.y_channel.data, post_frame.z_channel.data), axis=-1), 0.0, 1.0)
    out_bgr = cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if is_grayscale:
        out_bgr = out_bgr[:, :, 0]

    # Save standard Fattal result for reference
    ref_save_path = save_dir / f"{file_name}_fattal_only.png"
    cv2.imwrite(str(ref_save_path), out_bgr)
    print(f"  [Saved Reference] {ref_save_path.name}")

    # 2. Run post-processing Bilateral Filter parameter sweep
    # Bilateral filter parameters to combine:
    d_list = [5, 9, 15]
    sigma_color_list = [20, 50, 80]
    sigma_space_list = [20, 50, 80]

    total_runs = len(d_list) * len(sigma_color_list) * len(sigma_space_list)
    run_idx = 0

    for d in d_list:
        for sc in sigma_color_list:
            for ss in sigma_space_list:
                run_idx += 1
                
                # Apply bilateral filter to standard Fattal LDR result
                filtered_bgr = cv2.bilateralFilter(out_bgr, d, sc, ss)

                # Save the result
                param_suffix = f"d{d}_sc{sc}_ss{ss}"
                save_path = save_dir / f"{file_name}_post_bf_{param_suffix}.png"
                cv2.imwrite(str(save_path), filtered_bgr)

                if run_idx % 9 == 0 or run_idx == total_runs:
                    print(f"    [Progress] Completed {run_idx}/{total_runs} combinations...")

def main():
    utils.start_timer()
    utils.print_elapsed("Post-Fattal Bilateral sweep start")

    data_folders = sorted([
        d for d in DATA_DIR.iterdir() if d.is_dir()
    ], key=lambda x: x.name)

    if not data_folders:
        print(f"No data folders found in: {DATA_DIR}")
        return

    print(f"Found {len(data_folders)} data folder(s) for the experiment.")

    for folder in data_folders:
        hdr_files = list(folder.glob('*.hdr'))
        if not hdr_files:
            print(f"[{folder.name}] No .hdr files - skipping")
            continue

        # Get dataset specific parameters
        params = DATASET_PARAMS.get(folder.name, {"alpha": 0.9, "beta": 0.8})
        alpha = params["alpha"]
        beta = params["beta"]

        for hdr_file in hdr_files:
            process_single_image(hdr_file, folder.name, alpha, beta)

    utils.print_elapsed("Post-Fattal Bilateral sweep complete")
    print(f"\nAll results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
