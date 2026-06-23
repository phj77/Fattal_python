# exe_original_scanline_vis.py
# Save tone-mapped result images with scanline and highlight areas marked.
# Output: test/scanline/original_scanline_vis/<dataset_id>/[image_name]_tmo_marked.png

import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Setup paths and import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fattal.fattal_tmo import pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# --- Path setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Fattal_python root
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "scanline" / "original_scanline_vis"

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

# --- Fixed parameters (same across all datasets) ---
opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
HE_weight = 0.0
pre_gamma = 1.0
post_gamma = 1.0

def process_image(img_path, dataset_id, config):
    """Process a single HDR image: compute tone mapped LDR, mark scanline/highlight and save."""
    file_name = img_path.stem

    # Output folder: test/scanline/original_scanline_vis/<dataset_id>/
    save_dir = OUTPUT_DIR / str(dataset_id)
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

    opt_alpha = config["alpha"]
    opt_beta = config["beta"]
    scanline_row = config["row"]
    highlight_ranges = config["highlight"]
    opt_saturation = 1.0 if is_grayscale else 0.8

    # Tone Mapping
    R_out, G_out, B_out = pfstmo_fattal02(
        R_pre, G_pre, B_pre,
        opt_alpha, opt_beta, opt_saturation, opt_noise,
        newfattal, fftsolver, detail_level, HE_weight,
        scanline_row=None, highlight_ranges=None, save_dir=None
    )

    # Post-gamma correction
    post_frame = Frame(R_out, G_out, B_out)
    apply_gamma_frame(post_frame, post_gamma)
    R_final = post_frame.x_channel.data
    G_final = post_frame.y_channel.data
    B_final = post_frame.z_channel.data

    # Merge channels and convert to 8-bit
    out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
    out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
    out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

    h, w = out_img_bgr.shape[:2]
    img_draw = out_img_bgr.copy()

    # Draw semi-transparent green boxes for highlight ranges on img_draw (Green: BGR = (0, 255, 0))
    if highlight_ranges is not None:
        overlay = img_draw.copy()
        for rng in highlight_ranges:
            if len(rng) == 2:
                start_x = np.clip(rng[0], 0, w - 1)
                end_x = np.clip(rng[1], 0, w - 1)
                row_y = np.clip(scanline_row, 0, h - 1)
                cv2.rectangle(overlay, (start_x, row_y - 8), (end_x, row_y + 8), (0, 255, 0), thickness=-1)
        
        # Blend overlay with img_draw (alpha=0.3)
        cv2.addWeighted(overlay, 0.3, img_draw, 0.7, 0, img_draw)

    # Draw horizontal scanline (Red: BGR = (0, 0, 255))
    row_y = np.clip(scanline_row, 0, h - 1)
    cv2.line(img_draw, (0, row_y), (w - 1, row_y), (0, 0, 255), thickness=2)

    # Save tone-mapped image with markings
    save_name = f"{file_name}_tmo_marked.png"
    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), img_draw)
    print(f"    Saved: {save_path.name}")

def main():
    utils.start_timer()
    utils.print_elapsed("Tone-mapped scanline & highlight marking visualization start")

    print(f"Input Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    for dataset_id in sorted(dataset_configs.keys()):
        config = dataset_configs[dataset_id]
        
        dataset_path = DATA_DIR / str(dataset_id)
        if not dataset_path.exists():
            print(f"  [SKIP] Dataset folder {dataset_path} does not exist.")
            continue

        hdr_files = list(dataset_path.glob("*.hdr"))
        if not hdr_files:
            print(f"  [SKIP] No .hdr files in {dataset_path}.")
            continue

        print(f"\n--- Dataset [{dataset_id}] Processing (row={config['row']}) ---")
        for hdr_path in hdr_files:
            print(f"  Image: {hdr_path.name}")
            process_image(hdr_path, dataset_id, config)

    utils.print_elapsed("Tone-mapped scanline & highlight marking visualization complete")
    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
