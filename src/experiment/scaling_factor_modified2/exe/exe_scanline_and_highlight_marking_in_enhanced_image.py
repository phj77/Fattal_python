# exe_scanline_and_highlight_marking_in_enhanced_image.py
# scaling_factor_modified2의 Fattal 톤 매핑을 수행한 후, 생성된 결과 이미지 상에 분석에 사용된 스캔라인(빨간색 가로선)과 주요 하이라이트 분석 구간(반투명 녹색 박스)을 직접 드로잉하여 시각적으로 표시 및 저장하는 실행 스크립트입니다.
# Output: test/scaling_factor_modified2_scanline_vis/<dataset_id>/[image_name]_tmo_marked.png

import os
import cv2
import numpy as np
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Setup paths and import packages
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
project_root = current_file.parents[4]  # .../Fattal_python root
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from experiment.scaling_factor_modified2.fattal.fattal_tmo_scanline import pfstmo_fattal02
import utils.utils as utils

# --- Path setup ---
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "tmp3" / "scaling_factor_modified2_scanline_vis"

# --- Dataset parameter configuration (y_0 parameter added for modified2) ---
dataset_configs = {
    1: {"alpha": 0.9, "beta": 0.82, "y_0": 1.2, "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
    2: {"alpha": 0.9, "beta": 0.8,  "y_0": 1.2, "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
    3: {"alpha": 0.9, "beta": 0.81, "y_0": 1.2, "row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
    4: {"alpha": 0.9, "beta": 0.84, "y_0": 1.2, "row": 974,  "highlight": [[457, 475], [590, 607]]},
    5: {"alpha": 0.3, "beta": 0.93, "y_0": 1.2, "row": 1170, "highlight": [[2073, 2188]]},
    6: {"alpha": 0.9, "beta": 0.81, "y_0": 1.2, "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
    7: {"alpha": 0.9, "beta": 0.8,  "y_0": 1.2, "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
}

# --- Fixed parameters (same across all datasets) ---
opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
hpf_sigma = 0.007
pyramid_top_size = 8

def process_image(img_path, dataset_id, config):
    """Process a single HDR image: compute tone mapped LDR, mark scanline/highlight and save."""
    file_name = img_path.stem

    # Output folder: test/scaling_factor_modified2_scanline_vis/<dataset_id>/
    save_dir = OUTPUT_DIR / str(dataset_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [ERROR] Cannot read image: {img_path}")
        return

    # 같은 intensity를 가진 3채널 이미지에서 1채널만 추출하여 사용
    if img.ndim == 3:
        img_single = img[:, :, 0]
    else:
        img_single = img

    opt_alpha = config["alpha"]
    opt_beta = config["beta"]
    opt_y_0 = config.get("y_0", 1.2)
    scanline_row = config["row"]
    highlight_ranges = config["highlight"]

    # Tone Mapping (do not save intermediate scanlines during marking process)
    L_out = pfstmo_fattal02(
        img_single,
        opt_alpha, opt_beta, opt_noise,
        newfattal, fftsolver, detail_level,
        hpf_sigma=hpf_sigma,
        pyramid_top_size=pyramid_top_size,
        opt_y_0=opt_y_0,
        scanline_row=None, highlight_ranges=None, save_dir=None
    )

    # 포맷 변환 및 클리핑 (8bit 단일 채널 이미지)
    out_img = np.clip(L_out, 0.0, 1.0)
    out_img_8bit = (out_img * 255.0).astype(np.uint8)
    # 마킹을 위해 BGR 3채널로 변환
    out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_GRAY2BGR)

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
