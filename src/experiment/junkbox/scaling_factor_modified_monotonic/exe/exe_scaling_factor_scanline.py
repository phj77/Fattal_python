# exe_scaling_factor_scanline.py
# Save scanline plots and data for level-wise scaling factors and final combined attenuation map for scaling_factor_modified_monotonic.

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
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import scaling_factor_modified_monotonic modules and configs
from experiment.scaling_factor_modified_monotonic.fattal.fattal_tmo import (
    apply_high_pass_filter,
    createGaussianPyramids,
    calculate_level_scaling_factor,
    calculate_attenuation
)
from experiment.scaling_factor_modified_monotonic.config.config import (
    INPUT_DIR, OUTPUT_DIR, get_parameter_combinations,
    CROP_Y_RANGE, CROP_X_RANGE
)
import utils.utils as utils

FIX_Y_AXIS = False
DRAW_SCANLINE = False

dataset_configs = {
    1: {"row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
    2: {"row": 1661, "highlight": [[300, 530], [1868, 1965]]},
    3: {"row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
    4: {"row": 974,  "highlight": [[457, 475], [590, 607]]},
    5: {"row": 1170, "highlight": [[2073, 2188]]},
    6: {"row": 1590, "highlight": [[400, 620], [2095, 2190]]},
    7: {"row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
}


def validate_and_get_dataset_dirs(input_dir):
    input_dir_abs = os.path.abspath(input_dir)
    norm_input_dir = os.path.normpath(input_dir_abs)

    if not os.path.exists(norm_input_dir) or not os.path.isdir(norm_input_dir):
        print(f"[오류] 지정한 INPUT_DIR('{input_dir}') 경로가 존재하지 않거나 디렉토리가 아닙니다.")
        sys.exit(1)

    direct_hdr = glob.glob(os.path.join(norm_input_dir, '*.hdr'))
    if direct_hdr:
        return [norm_input_dir]

    subdirs = [os.path.join(norm_input_dir, d) for d in os.listdir(norm_input_dir)
               if os.path.isdir(os.path.join(norm_input_dir, d))]

    valid_dataset_dirs = []
    for sd in subdirs:
        sub_hdr = glob.glob(os.path.join(sd, '*.hdr'))
        if sub_hdr:
            sub_name = os.path.basename(sd)
            key = int(sub_name) if sub_name.isdigit() else 9999
            valid_dataset_dirs.append((key, sd))

    if not valid_dataset_dirs:
        print(f"[오류] INPUT_DIR('{input_dir}') 및 하위 폴더에서 .hdr 파일을 찾지 못했습니다.")
        sys.exit(1)

    valid_dataset_dirs.sort(key=lambda x: x[0])
    return [path for key, path in valid_dataset_dirs]


def get_scanline_config(input_dir_path, img_shape):
    dir_name = os.path.basename(os.path.normpath(input_dir_path))
    if dir_name.isdigit() and int(dir_name) in dataset_configs:
        cfg = dataset_configs[int(dir_name)]
        return cfg["row"], cfg["highlight"]
    
    return img_shape[0] // 2, None


def visualize_attenuation_map_with_scanline(att_map, level, scanline_row, save_path, title=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    im = ax.imshow(att_map, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Scaling Factor (Attenuation) Value', fontsize=12)

    if DRAW_SCANLINE:
        ax.axhline(y=scanline_row, color='red', linestyle='--', linewidth=1.5, label=f'Scanline Row {scanline_row}')
        ax.legend(loc='upper right')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif level >= 0:
        ax.set_title(f'Scaling Factor Map - Level {level} (Row {scanline_row})', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Final Combined Attenuation Map (Row {scanline_row})', fontsize=14, fontweight='bold')

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


def process_single_image(img_path, ds_dir, base_output_dir, param):
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    dataset_name = os.path.basename(os.path.normpath(ds_dir))

    # Parse parameter values
    opt_alpha = param['opt_alpha']
    opt_beta = param['opt_beta']
    opt_noise = param.get('opt_noise', 0.001)
    new_fattal = param.get('newfattal', True)
    fft_solver = param.get('fftsolver', True)
    detail_level = param.get('detail_level', 0)
    
    pre_hpf_sigma = param.get('pre_hpf_sigma', 0.010)
    noise_val = opt_noise if opt_noise > 0 else opt_alpha * 0.01
    xp_ratio = param.get('xp_ratio', 0.05)
    y0 = param.get('y0', 6.0)
    
    # Save directory structure
    crop_suffix = ""
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        crop_suffix = f"_cropY{CROP_Y_RANGE[0]}-{CROP_Y_RANGE[1]}_X{CROP_X_RANGE[0]}-{CROP_X_RANGE[1]}"
    
    param_folder_name = f"preHPF{pre_hpf_sigma}_a{opt_alpha}_b{opt_beta}_n{opt_noise}_dl{detail_level}_xpRatio{xp_ratio}_y0{y0}{crop_suffix}"
    
    if dataset_name.isdigit():
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
    else:
        dataset_output_dir = base_output_dir

    save_dir = Path(os.path.join(dataset_output_dir, param_folder_name, file_name))
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- [Scaling Factor Scanline] Processing: {file_name} ---")
    print(f"  Params: alpha={opt_alpha}, beta={opt_beta}, noise={noise_val}, detail_level={detail_level}, pre_hpf_sigma={pre_hpf_sigma}")
    print(f"  Save Dir: {save_dir}")

    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [오류] 이미지를 읽을 수 없습니다: {img_path}")
        return

    if img.ndim == 3:
        img_single = img[:, :, 0]
    else:
        img_single = img

    # Get scanline config
    scanline_row, highlight_ranges = get_scanline_config(ds_dir, img.shape)
    print(f"  스캔라인 설정 (원본) -> Row: {scanline_row}, Highlight: {highlight_ranges}")

    # Crop image if ranges are defined
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        h_orig, w_orig = img_single.shape[:2]
        ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h_orig)
        xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w_orig)
        ymin, ymax = max(0, ymin), min(h_orig, ymax)
        xmin, xmax = max(0, xmin), min(w_orig, xmax)
        
        img_single = img_single[ymin:ymax, xmin:xmax]
        
        scanline_row = scanline_row - ymin
        if highlight_ranges is not None:
            adjusted_highlights = []
            for rng in highlight_ranges:
                if len(rng) == 2:
                    r_start, r_end = rng
                    r_start_adj = max(0, r_start - xmin)
                    r_end_adj = min(xmax - xmin, r_end - xmin)
                    adjusted_highlights.append([r_start_adj, r_end_adj])
            highlight_ranges = adjusted_highlights
        print(f"  스캔라인 설정 (크롭 반영) -> Row: {scanline_row}, Highlight: {highlight_ranges}")

    # 0. Original HDR map (pre highpass filter 적용 전) scanline 저장
    utils.save_scanline(
        img_single,
        row_index=scanline_row,
        stage_name="0_original_HDR_Y",
        highlight_ranges=highlight_ranges,
        save_dir=str(save_dir)
    )

    # 1. Apply Pre-HPF
    img_filtered = apply_high_pass_filter(img_single, pre_hpf_sigma=pre_hpf_sigma)

    # 2. Log-space transform
    max_lum = np.max(img_filtered)
    H = np.log(100.0 * img_filtered / max_lum + 1e-4)

    # 3. Create Gaussian pyramids
    h, w = H.shape
    msize = 32 if fft_solver else 32
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= msize:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0:
        nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    print(f"  피라미드 레벨 수: {nlevels}  (원본 크기: {w}x{h})")

    # 4. Calculate level-wise scaling factors
    scaling_factors = [None] * nlevels
    for k in range(nlevels):
        if k >= detail_level or k == nlevels - 1 or not new_fattal:
            att = calculate_level_scaling_factor(
                pyramids[k], k, opt_alpha, opt_beta, noise_val,
                xp_ratio=xp_ratio, y0=y0
            )
            scaling_factors[k] = att

    # 5. Calculate final combined attenuation map
    final_att = calculate_attenuation(scaling_factors, pyramids, nlevels, new_fattal)

    # 6. Global vmin/vmax calculation if FIX_Y_AXIS is True
    if FIX_Y_AXIS:
        valid_maps = [s for s in scaling_factors if s is not None] + [final_att]
        vmin = min(m.min() for m in valid_maps)
        vmax = max(m.max() for m in valid_maps)
    else:
        vmin = None
        vmax = None

    # 7. Generate level-wise scaling factor scanlines and 2D visualizations
    for k in range(nlevels):
        if scaling_factors[k] is not None:
            att = scaling_factors[k]
            ph, pw = pyramids[k].shape

            scale_factor = 2.0 ** k
            row_k = int(round(scanline_row / scale_factor))
            row_k = np.clip(row_k, 0, ph - 1)

            highlight_ranges_k = []
            if highlight_ranges is not None:
                for rng in highlight_ranges:
                    if len(rng) == 2:
                        start_k = int(round(rng[0] / scale_factor))
                        end_k = int(round(rng[1] / scale_factor))
                        start_k = np.clip(start_k, 0, pw - 1)
                        end_k = np.clip(end_k, 0, pw - 1)
                        highlight_ranges_k.append([start_k, end_k])

            map_name = f"scaling_factor_level_{k:02d}_{pw}x{ph}_with_line.png"
            map_path = save_dir / map_name
            title = (
                f'Pre-HPF({pre_hpf_sigma}) SF Map - Level {k}/{nlevels-1}\n'
                f'(alpha={opt_alpha}, beta={opt_beta}, size={pw}x{ph}, scanline_row={row_k})'
            )
            visualize_attenuation_map_with_scanline(att, k, row_k, str(map_path), title=title, vmin=vmin, vmax=vmax)

            stage_name = f"scaling_factor_level_{k:02d}"
            utils.save_scanline(
                att,
                row_index=row_k,
                stage_name=stage_name,
                highlight_ranges=highlight_ranges_k,
                save_dir=str(save_dir)
            )
            print(f"    Level {k:2d}: Saved scanline and map at row {row_k} (size: {pw}x{ph})")
        else:
            print(f"    Level {k:2d}: (skipped, below detail_level={detail_level})")

    # 8. Generate final combined attenuation map scanline and 2D visualization
    fh, fw = final_att.shape
    final_map_name = f"attenuation_final_combined_{fw}x{fh}_with_line.png"
    final_map_path = save_dir / final_map_name
    final_title = (
        f'Pre-HPF({pre_hpf_sigma}) Final Combined Attenuation Map (Phi)\n'
        f'(alpha={opt_alpha}, beta={opt_beta}, size={fw}x{fh}, scanline_row={scanline_row})'
    )
    visualize_attenuation_map_with_scanline(final_att, -1, scanline_row, str(final_map_path), title=final_title, vmin=vmin, vmax=vmax)

    utils.save_scanline(
        final_att,
        row_index=scanline_row,
        stage_name="final_combined_attenuation_map",
        highlight_ranges=highlight_ranges,
        save_dir=str(save_dir)
    )
    print(f"    [Final Combined] Saved scanline and map at row {scanline_row} (size: {fw}x{fh})")


def main():
    utils.start_timer()
    utils.print_elapsed("Pre-HPF Scaling Factor 스캔라인 작업 시작")

    dataset_dirs = validate_and_get_dataset_dirs(INPUT_DIR)
    param_combinations = get_parameter_combinations()
    base_output_dir = OUTPUT_DIR

    print(f"입력 경로: {INPUT_DIR}")
    print(f"출력 경로: {base_output_dir}")
    print(f"감지된 데이터셋 디렉토리 수: {len(dataset_dirs)}, 파라미터 조합 수: {len(param_combinations)}\n")

    for d_idx, ds_dir in enumerate(dataset_dirs, 1):
        dataset_name = os.path.basename(os.path.normpath(ds_dir))
        hdr_files = glob.glob(os.path.join(ds_dir, '*.hdr'))

        print("=" * 70)
        print(f"[{d_idx}/{len(dataset_dirs)}] 데이터셋 처리 시작: {dataset_name} ({ds_dir})")
        print(f"감지된 HDR 이미지 수: {len(hdr_files)}")
        print("=" * 70)

        for hdr_file in hdr_files:
            for p_idx, param in enumerate(param_combinations, 1):
                process_single_image(
                    hdr_file, ds_dir, base_output_dir, param
                )

    utils.print_elapsed("Pre-HPF Scaling Factor 스캔라인 작업 완료")
    print(f"\n최종 결과 저장 디렉토리: {base_output_dir}")


if __name__ == "__main__":
    main()
