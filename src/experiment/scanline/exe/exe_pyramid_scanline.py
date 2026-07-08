# exe_pyramid_scanline.py
# 입력 HDR 이미지의 가우시안 피라미드(Gaussian Pyramid) 각 레벨별 이미지에서 특정 행(Row)의 스캔라인 강도 데이터를 추출 및 시각화하여 저장하는 실행 스크립트입니다.
# Output: test/pyramid_scanline/<dataset_id>/<img_name>/

import os
import cv2
import numpy as np
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths and import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fattal.fattal_tmo import createGaussianPyramids
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# --- Path setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Fattal_python root
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "pyramid_scanline"

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

opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
pre_gamma = 1.0
post_gamma = 1.0
MSIZE = 8 if fftsolver else 32

def main():
    utils.start_timer()
    utils.print_elapsed("Gaussian pyramid scanline extraction start")

    print(f"Input Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

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

        print(f"\n--- Dataset [{k}] Processing (alpha={opt_alpha}, beta={opt_beta}, row={scanline_row}) ---")
        
        for img_path in hdr_files:
            file_name = img_path.stem
            print(f"  Processing image: {file_name}")

            # Read HDR image
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
            print(f"    Gaussian pyramid levels: {nlevels} (original size: {w}x{h})")

            # Output folder: test/pyramid_scanline/<dataset_id>/<img_name>/
            img_output_dir = OUTPUT_DIR / str(k) / file_name
            img_output_dir.mkdir(parents=True, exist_ok=True)

            # Save scanline for each level
            for level in range(nlevels):
                level_img = pyramids[level]
                lh, lw = level_img.shape

                # Scale coordinates by 2^level
                scale_factor = 2.0 ** level
                row_k = int(round(scanline_row / scale_factor))
                
                # Clip row index to be within bounds
                row_k = np.clip(row_k, 0, lh - 1)

                # Scale highlight ranges
                highlight_ranges_k = []
                if highlight_ranges is not None:
                    for rng in highlight_ranges:
                        if len(rng) == 2:
                            start_k = int(round(rng[0] / scale_factor))
                            end_k = int(round(rng[1] / scale_factor))
                            # Clip highlight ranges to be within bounds
                            start_k = np.clip(start_k, 0, lw - 1)
                            end_k = np.clip(end_k, 0, lw - 1)
                            highlight_ranges_k.append([start_k, end_k])

                # Stage name
                stage_name = f"pyramid_level_{level:02d}"

                # Save using utils.save_scanline
                utils.save_scanline(
                    level_img, 
                    row_index=row_k, 
                    stage_name=stage_name, 
                    highlight_ranges=highlight_ranges_k, 
                    save_dir=str(img_output_dir)
                )
                print(f"      Level {level:2d}: saved scanline at row {row_k} (size: {lw}x{lh})")

    utils.print_elapsed("Gaussian pyramid scanline extraction complete")
    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
