# exe_scaling_factor_vis.py
# Visualize scaling factor (attenuation) maps for each pyramid level for Pre-HPF + Fattal algorithm.

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

from experiment.hpf_pre_fattal.fattal.fattal_tmo import (
    apply_high_pass_filter,
    createGaussianPyramids,
    calculate_level_scaling_factor,
    calculate_attenuation,
    pfstmo_fattal02
)
from experiment.hpf_pre_fattal.config.config import INPUT_DIR, OUTPUT_DIR, PARAM_GRID
import utils.utils as utils

# ─── 실험 전용 사전 HPF (Pre-HPF) 및 시각화 설정 ───────────────────────────
# Original 이미지에 적용할 High-Pass Filter sigma 강도를 지정합니다.
PRE_HPF_SIGMA = PARAM_GRID.get('pre_hpf_sigma', [0.010])[0]

# Y축 (Colorbar 범주) 고정 여부 설정:
# True  : 전체 레벨 및 최종 맵의 글로벌 min/max로 Y축(Colorbar) 고정
# False : 각 scaling factor map의 min/max에 따라 Y축(Colorbar) 동적 설정
FIX_Y_AXIS = False
# ─────────────────────────────────────────────────────────────────────────────

# --- Parameters (from config.py) ---
opt_alpha = PARAM_GRID['opt_alpha'][0]
opt_beta = PARAM_GRID['opt_beta'][0]
opt_noise = PARAM_GRID['opt_noise'][0]
newfattal = PARAM_GRID['newfattal'][0]
fftsolver = PARAM_GRID['fftsolver'][0]
detail_level = PARAM_GRID['detail_level'][0]
hpf_sigma = PARAM_GRID['hpf_sigma'][0]

MSIZE = 2**5 if fftsolver else 32


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


def process_single_image(img_path, data_folder_name, base_output_dir):
    """Process a single HDR image: apply Pre-HPF, compute scaling factor maps per level, and run TMO."""
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    save_dir = base_output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- [Pre-HPF {PRE_HPF_SIGMA}] Processing Image: {file_name} ---")

    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [오류] 이미지를 읽을 수 없습니다: {img_path}")
        return

    if img.ndim == 3:
        img_single = img[:, :, 0]
    else:
        img_single = img

    # 1. Pre-HPF 적용
    img_filtered = apply_high_pass_filter(img_single, pre_hpf_sigma=PRE_HPF_SIGMA)

    # 2. 로그 공간 변환
    maxLum = np.max(img_filtered)
    H = np.log(100.0 * img_filtered / maxLum + 1e-4)

    # 3. 가우시안 피라미드 생성
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
    print(f"  피라미드 레벨 수: {nlevels}  (원본 크기: {w}x{h})")

    noise = opt_noise if opt_noise > 0 else opt_alpha * 0.01

    # 4. 각 레벨별 Scaling Factor (감쇠) 계산 및 수집
    scaling_factors = [None] * nlevels
    for k in range(nlevels):
        if k >= detail_level or k == nlevels - 1 or not newfattal:
            att = calculate_level_scaling_factor(pyramids[k], k, opt_alpha, opt_beta, noise)
            scaling_factors[k] = att

    # 4-1. Scaling factor를 결합한 최종 종합 Attenuation Map 계산
    final_att = calculate_attenuation(scaling_factors, pyramids, nlevels, newfattal)

    # 5. Y축 (Colorbar 범주) 고정 여부에 따른 vmin/vmax 설정
    if FIX_Y_AXIS:
        valid_maps = [s for s in scaling_factors if s is not None] + [final_att]
        vmin = min(m.min() for m in valid_maps)
        vmax = max(m.max() for m in valid_maps)
    else:
        vmin = None
        vmax = None

    # 5-1. 각 레벨별 Scaling Factor 시각화 및 저장
    for k in range(nlevels):
        if scaling_factors[k] is not None:
            att = scaling_factors[k]
            ph, pw = pyramids[k].shape
            save_name = f"scaling_factor_level_{k:02d}_{pw}x{ph}.png"
            save_path = save_dir / save_name

            title = (
                f'Pre-HPF({PRE_HPF_SIGMA}) Scaling Factor Map - Level {k}/{nlevels-1}\n'
                f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise:.4f}, size={pw}x{ph})'
            )
            visualize_attenuation_map(att, k, str(save_path), title=title, vmin=vmin, vmax=vmax)

            cv_save_name = f"cv_scaling_factor_level_{k:02d}_{pw}x{ph}.png"
            cv_save_path = save_dir / cv_save_name
            save_attenuation_opencv(att, cv_save_path)

            print(f"    Level {k:2d}: {pw:5d}x{ph:<5d}  -> saved: {save_path.name}, {cv_save_path.name}")
        else:
            print(f"    Level {k:2d}: (skipped, below detail_level={detail_level})")

    # 5-2. 최종 종합 Attenuation Map 시각화 및 저장
    fh, fw = final_att.shape
    final_save_name = f"attenuation_final_combined_{fw}x{fh}.png"
    final_save_path = save_dir / final_save_name

    final_title = (
        f'Pre-HPF({PRE_HPF_SIGMA}) Final Combined Attenuation Map (Phi)\n'
        f'(alpha={opt_alpha}, beta={opt_beta}, noise={noise:.4f}, detail_level={detail_level}, size={fw}x{fh})'
    )
    visualize_attenuation_map(final_att, -1, str(final_save_path), title=final_title, vmin=vmin, vmax=vmax)

    cv_final_save_name = f"cv_attenuation_final_combined_{fw}x{fh}.png"
    cv_final_save_path = save_dir / cv_final_save_name
    save_attenuation_opencv(final_att, cv_final_save_path)

    print(f"    [최종 ATT] 종합 Attenuation Map 저장 완료: {final_save_path.name}, {cv_final_save_path.name}")

    # 5. Tone Mapping 연산 및 저장
    print(f"  [TMO] Pre-HPF + Fattal 톤 매핑 실행 중...")
    L_out = pfstmo_fattal02(
        img_single,
        opt_alpha, opt_beta, opt_noise,
        newfattal, fftsolver, detail_level,
        hpf_sigma=hpf_sigma,
        pre_hpf_sigma=PRE_HPF_SIGMA
    )

    out_img_8bit = (np.clip(L_out, 0.0, 1.0) * 255.0).astype(np.uint8)

    tmo_save_name = f"{file_name}_preHPF{PRE_HPF_SIGMA}_a{opt_alpha}_b{opt_beta}_tonemapped.png"
    tmo_save_path = save_dir / tmo_save_name
    cv2.imwrite(str(tmo_save_path), out_img_8bit)
    print(f"    [TMO 저장 완료] {tmo_save_path.name}")


def main():
    utils.start_timer()
    utils.print_elapsed("Pre-HPF Scaling Factor 시각화 작업 시작")

    input_path = Path(INPUT_DIR)
    base_output_dir = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[오류] 입력 디렉토리가 존재하지 않습니다: {input_path}")
        return

    direct_hdr_files = list(input_path.glob('*.hdr'))
    if direct_hdr_files:
        print(f"단일 데이터 폴더 처리 중: {input_path.name}")
        for hdr_file in direct_hdr_files:
            process_single_image(hdr_file, input_path.name, base_output_dir)
    else:
        data_folders = sorted([
            d for d in input_path.iterdir() if d.is_dir()
        ], key=lambda x: x.name)

        if not data_folders:
            print(f"[오류] 입력 디렉토리에서 .hdr 파일이나 하위 디렉토리를 찾을 수 없습니다: {input_path}")
            return

        print(f"감지된 데이터 폴더 수: {len(data_folders)}\n")

        for folder in data_folders:
            hdr_files = list(folder.glob('*.hdr'))
            if not hdr_files:
                print(f"[{folder.name}] .hdr 파일이 없어 스킵합니다.")
                continue

            for hdr_file in hdr_files:
                process_single_image(hdr_file, folder.name, base_output_dir)

    utils.print_elapsed("Pre-HPF Scaling Factor 시각화 작업 완료")
    print(f"\n결과 저장 디렉토리: {base_output_dir}")


if __name__ == "__main__":
    main()
