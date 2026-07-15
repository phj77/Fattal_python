# exe_base_detail_scanline.py - Guided Filter + Fattal Base Layer Scanline Visualization
# PIPELINE: HDR → log → Guided Filter (base/detail 분리) → Fattal on base → 합성 → exp → 8bit LDR
# 추가 기능: 분리(H, base, detail) 및 합성(combined, base, scaled detail) 시의 스캔라인 멀티라인 비교 출력 및 저장

import cv2
import numpy as np
import os
import glob
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))

# src/ 추가 (fattal.*, utils.* 사용)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 실험 로컬 모듈 경로 추가 (config/, fattal/ 하위 모듈)
config_dir = os.path.join(exp_dir, 'config')
fattal_dir = os.path.join(exp_dir, 'fattal')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
if fattal_dir not in sys.path:
    sys.path.insert(0, fattal_dir)
# ─────────────────────────────────────────────────────────────────────────────

# 실험 전용 Fattal log domain 함수
from gf_fattal_tmo import tmo_fattal02_logdomain

# 실험 전용 설정
from config import INPUT_DIR, OUTPUT_DIR, CROP_Y_RANGE, CROP_X_RANGE, get_parameter_combinations, project_root

import utils.utils as utils

# 데이터셋별 스캔라인 위치 설정 및 하이라이트 구간 설정 (1~7 데이터셋 매핑)
dataset_configs = {
    1: {"row": 1100, "col": None, "highlight_row": [[2310, 2382], [1740, 1825]], "highlight_col": None},
    2: {"row": 1661, "col": 1909, "highlight_row": [[300, 530], [1868, 1965]], "highlight_col": [[238, 342]]},
    3: {"row": 955,  "col": 2311, "highlight_row": [[533, 622], [1380, 1490], [2260, 2355]], "highlight_col": [[206, 328], [1716, 1815]]},
    4: {"row": 974,  "col": 2486, "highlight_row": [[457, 475], [590, 607]], "highlight_col": [[275, 377], [1662, 1730]]},
    5: {"row": 1170, "col": 1348, "highlight_row": [[2073, 2188]], "highlight_col": [[695, 848]]},
    6: {"row": 1590, "col": None, "highlight_row": [[400, 620], [2095, 2190]], "highlight_col": None},
    7: {"row": 1338, "col": None, "highlight_row": [[1295, 1360], [2570, 2650]], "highlight_col": None}
}

def save_combined_scanline(lines, labels, colors, index, direction, stage_name, highlight_ranges=None, save_dir=None, ylim=None):
    """
    여러 개의 스캔라인 데이터를 하나의 그래프에 플롯하고 저장합니다.
    """
    from matplotlib.ticker import AutoMinorLocator

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    
    for line, label, color in zip(lines, labels, colors):
        ax.plot(line, color=color, linewidth=0.8, alpha=0.85, label=label)
        
    if highlight_ranges is not None:
        for rng in highlight_ranges:
            if len(rng) == 2:
                start, end = rng
                start = max(0, min(start, len(lines[0]) - 1))
                end = max(0, min(end, len(lines[0]) - 1))
                ax.axvspan(start, end, color='#ffa500', alpha=0.25)
                
    dir_str = "Row" if direction == 'row' else "Column"
    ax.set_title(f"Scanline Comparison at {dir_str} {index} - {stage_name}", fontsize=14, pad=15, fontweight='bold')
    xlabel_str = "Column Index (X)" if direction == 'row' else "Row Index (Y)"
    ax.set_xlabel(xlabel_str, fontsize=11, labelpad=8)
    ax.set_ylabel("Intensity Value", fontsize=11, labelpad=8)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', color='#e5e5e5', linestyle=':', linewidth=0.4)
    
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc')
    
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        all_min = min(np.min(line) for line in lines)
        all_max = max(np.max(line) for line in lines)
        y_margin = (all_max - all_min) * 0.05 if all_max > all_min else 1.0
        ax.set_ylim(all_min - y_margin, all_max + y_margin)
        
    ax.set_xlim(0, len(lines[0]) - 1)
    
    safe_stage = stage_name.replace(" ", "_").replace("/", "_")
    filename = f"scanline_{direction}{index}_{safe_stage}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # npy 파일로 스캔라인 수치 데이터 저장
    npy_filename = f"scanline_{direction}{index}_{safe_stage}.npy"
    npy_path = os.path.join(save_dir, npy_filename)
    npy_data = {label: line for label, line in zip(labels, lines)}
    np.save(npy_path, npy_data)
    print(f"  [저장 완료] 스캔라인 그래프 및 데이터: {save_path}")

def main():
    utils.start_timer()
    utils.print_elapsed("Guided Filter + Base/Detail Separation Scanline 실험 시작")
    
    scanline_output_root = OUTPUT_DIR
    
    # 입력 디렉토리 경로 정규화 및 데이터셋 번호 파악
    norm_input = os.path.normpath(INPUT_DIR)
    base_name = os.path.basename(norm_input)

    dataset_dirs = []
    if base_name.isdigit():
        num_key = int(base_name)
        dataset_dirs.append((num_key, norm_input))
    else:
        for k in range(1, 8):
            sub_dir = os.path.join(norm_input, str(k))
            if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
                dataset_dirs.append((k, sub_dir))

    if not dataset_dirs:
        print(f"경고: '{INPUT_DIR}'에서 유효한 데이터셋 디렉토리를 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    
    print(f"출력 루트 경로: {scanline_output_root}")
    print(f"총 {len(dataset_dirs)}개의 데이터셋 디렉토리가 감지되었습니다.")
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        print(f"크롭 설정 적용 - Y축: {CROP_Y_RANGE}, X축: {CROP_X_RANGE}")

    for k, input_dir in dataset_dirs:
        config = dataset_configs.get(k, {"row": None, "col": None, "highlight_row": None, "highlight_col": None})
        
        search_pattern = os.path.join(input_dir, '*.hdr')
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{input_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        print(f"\n--- 데이터셋 [{k}] 처리 시작 (이미지 개수: {len(hdr_files)}) ---")
        
        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"이미지 로딩 중: {file_name}")
            
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue
                
            if img.ndim == 3:
                img_single = img[:, :, 0]
            else:
                img_single = img
                
            # 이미지 크롭 범위 설정 및 좌표 클리핑
            h, w = img_single.shape
            ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h)
            xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w)
            
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            
            img_cropped = img_single[ymin:ymax, xmin:xmax]
            crop_suffix = f"_cropY{ymin}-{ymax}_X{xmin}-{xmax}" if (CROP_Y_RANGE is not None or CROP_X_RANGE is not None) else ""
            
            for p_idx, p in enumerate(param_combinations, 1):
                newfattal = p['newfattal']
                if p['fftsolver']:
                    newfattal = True
                opt_noise = p['opt_noise']
                if opt_noise <= 0.0:
                    opt_noise = p['opt_alpha'] * 0.01

                # ── Step 1: Log domain 변환 ──
                Y = img_cropped.astype(np.float64)
                maxLum = np.max(Y)
                H = np.log(100.0 * Y / maxLum + 1e-4)

                # ── Step 2: Guided Filter로 base/detail 분리 ──
                H_float32 = H.astype(np.float32)
                base_layer = cv2.ximgproc.guidedFilter(
                    guide=H_float32, 
                    src=H_float32, 
                    radius=p['gf_radius'], 
                    eps=p['gf_eps']
                ).astype(np.float64)
                detail_layer = H - base_layer

                # ── Step 3: Base layer에 Fattal TMO 적용 (log domain) ──
                tone_mapped_base = tmo_fattal02_logdomain(
                    base_layer,
                    p['opt_alpha'], p['opt_beta'], opt_noise,
                    newfattal, p['fftsolver'], p['detail_level'],
                    hpf_sigma=p.get('hpf_sigma', 0.007)
                )

                # ── Step 4: 합성 ──
                detail_factor = p['detail_factor']
                combined = tone_mapped_base + detail_factor * detail_layer

                # ── Step 5: Exp & 정규화 → 8bit LDR ──
                L = np.exp(combined)
                cut_min = 0.01 * 0.1
                cut_max = 1.0 - 0.01 * 0.5
                
                min_val = np.percentile(L, cut_min * 100)
                max_val = np.percentile(L, cut_max * 100)
                
                L = (L - min_val) / (max_val - min_val)
                L = np.clip(L, 0.0, 1.0)
                out_img_8bit = (L * 255.0).astype(np.uint8)

                # 저장 폴더 구성 (파라미터 조건별 서브디렉토리 생성)
                param_folder = (
                    f"a{p['opt_alpha']}_b{p['opt_beta']}_dl{p['detail_level']}"
                    f"_gfr{p['gf_radius']}_gfe{p['gf_eps']}_df{detail_factor}"
                )
                
                # 단일 데이터셋 & 단일 파라미터 조합인 경우 OUTPUT_DIR 직하위에 파일명 폴더를 생성하여 저장
                if len(dataset_dirs) == 1 and len(param_combinations) == 1:
                    save_dir = os.path.join(scanline_output_root, file_name)
                elif len(dataset_dirs) == 1:
                    save_dir = os.path.join(scanline_output_root, param_folder, file_name)
                else:
                    save_dir = os.path.join(scanline_output_root, str(k), param_folder, file_name)
                    
                os.makedirs(save_dir, exist_ok=True)

                # LDR 이미지 결과 저장
                img_save_name = f"result_{file_name}{crop_suffix}.png"
                cv2.imwrite(os.path.join(save_dir, img_save_name), out_img_8bit)
                
                # ── Step 6: 스캔라인 추출 및 멀티라인 플롯 비교 시각화 ──
                scanline_row = config["row"]
                scanline_col = config["col"]
                highlight_row = config["highlight_row"]
                highlight_col = config["highlight_col"]

                # 1. 가로 스캔라인 (Row) 처리
                if scanline_row is not None:
                    # 크롭 영역 경계 검사 및 상대 인덱스 변환
                    if not (ymin <= scanline_row < ymax):
                        print(f"  [경고] 설정된 row {scanline_row}가 Y 크롭 범위 [{ymin}, {ymax}]를 벗어납니다.")
                        r_idx = (ymin + ymax) // 2 - ymin
                        orig_r = (ymin + ymax) // 2
                        print(f"  대신 크롭 영역 중심 Row {orig_r} (크롭 내 인덱스 {r_idx})를 사용합니다.")
                    else:
                        r_idx = scanline_row - ymin
                        orig_r = scanline_row

                    # 가로 방향 하이라이트 구간의 x좌표 시프트 (크롭 기준)
                    highlight_ranges_shifted = []
                    if highlight_row is not None:
                        for rng in highlight_row:
                            start_shifted = max(0, rng[0] - xmin)
                            end_shifted = min(xmax - xmin - 1, rng[1] - xmin)
                            if start_shifted < end_shifted:
                                highlight_ranges_shifted.append([start_shifted, end_shifted])

                    # 1-1. 분리 단계 스캔라인 (H, base, detail)
                    sep_lines = [H[r_idx, :], base_layer[r_idx, :], detail_layer[r_idx, :]]
                    sep_labels = ["H (Original Log)", "Base Layer (Guided Filter)", "Detail Layer"]
                    sep_colors = ["black", "blue", "red"]
                    save_combined_scanline(
                        sep_lines, sep_labels, sep_colors,
                        index=orig_r, direction="row",
                        stage_name="separation",
                        highlight_ranges=highlight_ranges_shifted,
                        save_dir=save_dir
                    )

                    # 1-2. 합성 단계 스캔라인 (combined, base, scaled detail)
                    comb_lines = [combined[r_idx, :], tone_mapped_base[r_idx, :], detail_factor * detail_layer[r_idx, :]]
                    comb_labels = ["Combined", "Tone-mapped Base", f"Scaled Detail (df={detail_factor})"]
                    comb_colors = ["black", "blue", "red"]
                    save_combined_scanline(
                        comb_lines, comb_labels, comb_colors,
                        index=orig_r, direction="row",
                        stage_name="combination",
                        highlight_ranges=highlight_ranges_shifted,
                        save_dir=save_dir
                    )

                # 2. 세로 스캔라인 (Col) 처리
                if scanline_col is not None:
                    # 크롭 영역 경계 검사 및 상대 인덱스 변환
                    if not (xmin <= scanline_col < xmax):
                        print(f"  [경고] 설정된 col {scanline_col}이 X 크롭 범위 [{xmin}, {xmax}]를 벗어납니다.")
                        c_idx = (xmin + xmax) // 2 - xmin
                        orig_c = (xmin + xmax) // 2
                        print(f"  대신 크롭 영역 중심 Col {orig_c} (크롭 내 인덱스 {c_idx})를 사용합니다.")
                    else:
                        c_idx = scanline_col - xmin
                        orig_c = scanline_col

                    # 세로 방향 하이라이트 구간의 y좌표 시프트 (크롭 기준)
                    highlight_ranges_shifted = []
                    if highlight_col is not None:
                        for rng in highlight_col:
                            start_shifted = max(0, rng[0] - ymin)
                            end_shifted = min(ymax - ymin - 1, rng[1] - ymin)
                            if start_shifted < end_shifted:
                                highlight_ranges_shifted.append([start_shifted, end_shifted])

                    # 2-1. 분리 단계 스캔라인 (H, base, detail)
                    sep_lines = [H[:, c_idx], base_layer[:, c_idx], detail_layer[:, c_idx]]
                    sep_labels = ["H (Original Log)", "Base Layer (Guided Filter)", "Detail Layer"]
                    sep_colors = ["black", "blue", "red"]
                    save_combined_scanline(
                        sep_lines, sep_labels, sep_colors,
                        index=orig_c, direction="col",
                        stage_name="separation",
                        highlight_ranges=highlight_ranges_shifted,
                        save_dir=save_dir
                    )

                    # 2-2. 합성 단계 스캔라인 (combined, base, scaled detail)
                    comb_lines = [combined[:, c_idx], tone_mapped_base[:, c_idx], detail_factor * detail_layer[:, c_idx]]
                    comb_labels = ["Combined", "Tone-mapped Base", f"Scaled Detail (df={detail_factor})"]
                    comb_colors = ["black", "blue", "red"]
                    save_combined_scanline(
                        comb_lines, comb_labels, comb_colors,
                        index=orig_c, direction="col",
                        stage_name="combination",
                        highlight_ranges=highlight_ranges_shifted,
                        save_dir=save_dir
                    )

    utils.print_elapsed("모든 작업 종료")

if __name__ == "__main__":
    main()
