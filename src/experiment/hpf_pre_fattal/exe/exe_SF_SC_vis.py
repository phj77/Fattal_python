# exe.SF_SC_vis.py
# Integrated visualization script for Pre-HPF + Fattal TMO:
# Generates Tone-Mapped Results, Level-wise Scaling Factor (Attenuation) Maps, and Scanline Profiles.

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

# Import Pre-HPF + Fattal algorithm modules and config
from experiment.hpf_pre_fattal.fattal.fattal_tmo import (
    apply_high_pass_filter,
    createGaussianPyramids,
    calculate_level_scaling_factor,
    calculate_attenuation,
    pfstmo_fattal02
)
from exe.config.config import INPUT_DIR, OUTPUT_DIR, PARAM_GRID, get_parameter_combinations
import utils.utils as utils

# ─── 실험 전용 사전 HPF (Pre-HPF) 및 시각화 설정 ───────────────────────────
# Original 이미지에 적용할 High-Pass Filter sigma 강도
PRE_HPF_SIGMA = 0.012

# Scaling Factor Colorbar Y축 범주 고정 여부
FIX_Y_AXIS = False
# ─────────────────────────────────────────────────────────────────────────────

# Dataset configs mapping scanline row and highlight ranges for default datasets (1~7)
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
    """
    INPUT_DIR 및 하위 디렉토리를 검증하고, .hdr 파일이 존재하는 폴더 경로 리스트를 반환합니다.
    """
    input_dir_abs = os.path.abspath(input_dir)
    norm_input_dir = os.path.normpath(input_dir_abs)

    if not os.path.exists(norm_input_dir) or not os.path.isdir(norm_input_dir):
        print(f"[오류] 지정한 INPUT_DIR('{input_dir}') 경로가 존재하지 않거나 디렉토리가 아닙니다.")
        sys.exit(1)

    # Case 1: INPUT_DIR 내에 .hdr 파일이 직접 존재하는 경우
    direct_hdr = glob.glob(os.path.join(norm_input_dir, '*.hdr'))
    if direct_hdr:
        return [norm_input_dir]

    # Case 2: 하위 디렉토리 탐색
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
    """Determine scanline row and highlight ranges based on input directory name."""
    dir_name = os.path.basename(os.path.normpath(input_dir_path))
    if dir_name.isdigit() and int(dir_name) in dataset_configs:
        cfg = dataset_configs[int(dir_name)]
        return cfg["row"], cfg["highlight"]
    
    return img_shape[0] // 2, None


def visualize_attenuation_map(att_map, level, save_path, title=None, vmin=None, vmax=None):
    """Visualize a single scaling factor / attenuation map as a colormap and save to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    im = ax.imshow(att_map, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Scaling Factor (Attenuation) Value', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif level >= 0:
        ax.set_title(f'Scaling Factor Map - Pyramid Level {level}', fontsize=14, fontweight='bold')
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
    """Save scaling factor map as a normalized 8-bit grayscale image using OpenCV."""
    att_min = att_map.min()
    att_max = att_map.max()
    if att_max - att_min > 1e-8:
        normalized = ((att_map - att_min) / (att_max - att_min) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(att_map, dtype=np.uint8)

    cv2.imwrite(str(save_path), normalized)


def process_single_image_integrated(img_path, ds_dir, save_dir, param):
    """
    Process single image for a specific parameter configuration:
    1. Runs Fattal TMO + outputs step-by-step Scanline profile graphs.
    2. Saves final 8-bit Tone-Mapped image.
    3. Computes and saves Scaling Factor (Attenuation) maps per pyramid level & combined map.
    """
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    # 파라미터 값 해제
    opt_alpha = param['opt_alpha']
    opt_beta = param['opt_beta']
    opt_noise = param.get('opt_noise', 0.001)
    newfattal = param.get('newfattal', True)
    fftsolver = param.get('fftsolver', True)
    detail_level = param.get('detail_level', 0)
    hpf_sigma = param.get('hpf_sigma', 0.007)
    pre_hpf_sigma = PRE_HPF_SIGMA

    print(f"\n--- Processing Image: {file_name} ---")
    print(f"  Params: alpha={opt_alpha}, beta={opt_beta}, noise={opt_noise}, detail_level={detail_level}, pre_hpf_sigma={pre_hpf_sigma}")
    print(f"  Save Directory: {save_dir}")

    # 이미지 읽기
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [오류] 이미지를 읽을 수 없습니다: {img_path}")
        return

    if img.ndim == 3:
        img_single = img[:, :, 0]
    else:
        img_single = img

    # Scanline 설정 파악
    scanline_row, highlight_ranges = get_scanline_config(ds_dir, img.shape)
    print(f"  스캔라인 설정 -> Row: {scanline_row}, Highlight: {highlight_ranges}")

    # 1. Tone Mapping 실행 및 Scanline 1~7번 그래프 자동 생성/저장
    print("  [1/3] Fattal 톤 매핑 및 Scanline 시각화 프로필 생성 중...")
    L_out = pfstmo_fattal02(
        img_single,
        opt_alpha, opt_beta, opt_noise,
        newfattal, fftsolver, detail_level,
        scanline_row=scanline_row,
        highlight_ranges=highlight_ranges,
        save_dir=str(save_dir),
        hpf_sigma=hpf_sigma,
        pre_hpf_sigma=pre_hpf_sigma
    )

    # 2. 톤매핑 완료된 8-bit 결과 이미지 저장
    out_img_8bit = (np.clip(L_out, 0.0, 1.0) * 255.0).astype(np.uint8)
    tmo_save_name = f"{file_name}_preHPF{pre_hpf_sigma}_a{opt_alpha}_b{opt_beta}_tonemapped.png"
    tmo_save_path = save_dir / tmo_save_name
    cv2.imwrite(str(tmo_save_path), out_img_8bit)
    print(f"  [2/3] Fattal 톤 매핑 결과 이미지 저장 완료: {tmo_save_path.name}")

    # 3. Scaling Factor Map (Attenuation Map) 계산 및 시각화 저장
    print("  [3/3] Scaling Factor (Attenuation) Map 계산 및 시각화 생성 중...")
    img_filtered = apply_high_pass_filter(img_single, pre_hpf_sigma=pre_hpf_sigma)
    maxLum = np.max(img_filtered)
    H = np.log(100.0 * img_filtered / maxLum + 1e-4)

    msize = 8 if fftsolver else 32
    h, w = H.shape
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= msize:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0:
        nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    noise_val = opt_noise if opt_noise > 0 else opt_alpha * 0.01

    scaling_factors = [None] * nlevels
    for k in range(nlevels):
        if k >= detail_level or k == nlevels - 1 or not newfattal:
            att = calculate_level_scaling_factor(pyramids[k], k, opt_alpha, opt_beta, noise_val)
            scaling_factors[k] = att

    final_att = calculate_attenuation(scaling_factors, pyramids, nlevels, newfattal)

    if FIX_Y_AXIS:
        valid_maps = [s for s in scaling_factors if s is not None] + [final_att]
        vmin = min(m.min() for m in valid_maps)
        vmax = max(m.max() for m in valid_maps)
    else:
        vmin = None
        vmax = None

    # 각 피라미드 레벨별 scaling factor map 저장
    for k in range(nlevels):
        if scaling_factors[k] is not None:
            att = scaling_factors[k]
            ph, pw = pyramids[k].shape
            save_name = f"scaling_factor_level_{k:02d}_{pw}x{ph}.png"
            sf_save_path = save_dir / save_name

            title = (
                f'Pre-HPF({pre_hpf_sigma}) Scaling Factor Map - Level {k}/{nlevels-1}\n'
                f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise_val:.4f}, size={pw}x{ph})'
            )
            visualize_attenuation_map(att, k, str(sf_save_path), title=title, vmin=vmin, vmax=vmax)

            cv_save_name = f"cv_scaling_factor_level_{k:02d}_{pw}x{ph}.png"
            cv_sf_save_path = save_dir / cv_save_name
            save_attenuation_opencv(att, cv_sf_save_path)
            print(f"    - Level {k:2d} SF Map: {save_name}, {cv_save_name}")

    # 최종 combined attenuation map 저장
    fh, fw = final_att.shape
    final_save_name = f"attenuation_final_combined_{fw}x{fh}.png"
    final_sf_save_path = save_dir / final_save_name
    final_title = (
        f'Pre-HPF({pre_hpf_sigma}) Final Combined Attenuation Map (Phi)\n'
        f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise_val:.4f}, detail_level={detail_level}, size={fw}x{fh})'
    )
    visualize_attenuation_map(final_att, -1, str(final_sf_save_path), title=final_title, vmin=vmin, vmax=vmax)

    cv_final_save_name = f"cv_attenuation_final_combined_{fw}x{fh}.png"
    cv_final_sf_save_path = save_dir / cv_final_save_name
    save_attenuation_opencv(final_att, cv_final_sf_save_path)
    print(f"    - Final Combined Attenuation Map: {final_save_name}, {cv_final_save_name}")


def main():
    utils.start_timer()
    utils.print_elapsed("Pre-HPF + Fattal 통합 시각화 (SF + SC + Result) 실행 시작")

    dataset_dirs = validate_and_get_dataset_dirs(INPUT_DIR)
    param_combinations = get_parameter_combinations()
    base_output_dir = Path(OUTPUT_DIR)

    print(f"입력 경로: {INPUT_DIR}")
    print(f"출력 경로: {base_output_dir}")
    print(f"사전 HPF Sigma: {PRE_HPF_SIGMA}")
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
                # 단일/다중 디렉토리 및 파라미터 조합별 저장 폴더 지정
                process_single_image_integrated(
                    hdr_file, ds_dir, base_output_dir, param
                )

    utils.print_elapsed("Pre-HPF + Fattal 통합 시각화 작업 완료")
    print(f"\n최종 결과 저장 디렉토리: {base_output_dir}")


if __name__ == "__main__":
    main()
