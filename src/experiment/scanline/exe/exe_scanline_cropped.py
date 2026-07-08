# exe_scanline_cropped.py
# 연산 중간 단계의 특정 신호(오리지널 Y, HDR GX, log-domain GX 등)들에 대해, 급격한 값 변화가 일어나는 하이라이트 영역 등의 국소적인 값 변동을 상세히 분석할 수 있도록 지정한 Y축 범위 또는 백분위수 기준으로 크롭된 스캔라인 그래프를 생성 및 저장하는 실행 스크립트입니다.
import cv2
import numpy as np
import os
import glob
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# --- 스캔라인별 Y축 범위 설정 파라미터 (수정 가능) ---

# 1. 1_original_HDR_Y 파라미터
# 하위 X% 백분위수 값(V)을 기준으로, [V - Y_VAL, V] 범위를 출력합니다.
Y1_X_PCT = 90
Y1_Y_VAL = 8000

# 2. 2_original_HDR_GX 파라미터
# 구체적인 출력 intensity 범위를 [Y2_MIN_VAL, Y2_MAX_VAL]로 설정합니다.
Y2_MIN_VAL = -400
Y2_MAX_VAL = 400

# 4. 4_original_log_domain_GX 파라미터
# 구체적인 출력 intensity 범위를 [Y4_MIN_VAL, Y4_MAX_VAL]로 설정합니다.
Y4_MIN_VAL = -0.01
Y4_MAX_VAL = 0.01

# Setup paths and import packages
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # Fattal_python root
sys.path.append(str(project_root / "src"))

from fattal.fattal_tmo import pfstmo_fattal02
import utils.utils as utils

def save_scanline_cropped(image, row_index, stage_name, highlight_ranges=None, save_dir=None, 
                          y_min_val=None, y_max_val=None, pct_x=None, offset_y=None, dataset_id=None):
    """
    특정 행(row)의 스캔라인을 추출하여 Y축 범위를 크롭하여 플롯으로 저장합니다.
    - pct_x와 offset_y가 지정된 경우: Y축 상한은 하위 pct_x%의 백분위수 값(V), 하한은 V - offset_y로 설정합니다.
    - y_min_val과 y_max_val이 지정된 경우: Y축 범위를 [y_min_val, y_max_val]로 직접 설정합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    h = image.shape[0]
    row_index = np.clip(row_index, 0, h - 1)
    
    scanline = image[row_index, :]
    
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    
    ax.plot(scanline, color='#1f77b4', linewidth=0.8, alpha=0.9, label='Intensity')
    
    if highlight_ranges is not None:
        for rng in highlight_ranges:
            if len(rng) == 2:
                start, end = rng
                start = max(0, min(start, len(scanline) - 1))
                end = max(0, min(end, len(scanline) - 1))
                label = f"Highlight [{start}-{end}]"
                ax.axvspan(start, end, color='#ffa500', alpha=0.25, label=label)
                
    ax.set_xlabel("Column Index (X)", fontsize=11, labelpad=8)
    ax.set_ylabel("Intensity (Y)", fontsize=11, labelpad=8)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.grid(True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', color='#e5e5e5', linestyle=':', linewidth=0.4)
    
    s_min = np.min(scanline)
    s_max = np.max(scanline)
    s_mean = np.mean(scanline)
    
    if pct_x is not None and offset_y is not None:
        y_max = np.percentile(scanline, pct_x)
        y_min = y_max - offset_y
        title_range_str = f"Bottom {pct_x}% - {offset_y} offset"
        stats_text = (
            f"Min: {s_min:.6f}\n"
            f"Max (Global): {s_max:.6f}\n"
            f"Mean (Global): {s_mean:.6f}\n"
            f"Bottom {pct_x}% (Ref): {y_max:.6f}\n"
            f"Range: [{y_min:.6f}, {y_max:.6f}]"
        )
    elif y_min_val is not None and y_max_val is not None:
        y_min = y_min_val
        y_max = y_max_val
        title_range_str = f"Fixed Range [{y_min:.3f}, {y_max:.3f}]"
        stats_text = (
            f"Min: {s_min:.6f}\n"
            f"Max (Global): {s_max:.6f}\n"
            f"Mean (Global): {s_mean:.6f}\n"
            f"Range: [{y_min:.6f}, {y_max:.6f}]"
        )
    else:
        y_min = s_min
        y_max = s_max
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        y_min -= y_margin
        y_max += y_margin
        title_range_str = "Full Range"
        stats_text = (
            f"Min: {s_min:.6f}\n"
            f"Max (Global): {s_max:.6f}\n"
            f"Mean (Global): {s_mean:.6f}\n"
            f"Range: [{y_min:.6f}, {y_max:.6f}]"
        )
        
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, len(scanline) - 1)
    
    dataset_title_prefix = f"Dataset {dataset_id} - " if dataset_id is not None else ""
    ax.set_title(f"{dataset_title_prefix}Scanline Intensity at Row {row_index} - {stage_name} ({title_range_str})", fontsize=14, pad=15, fontweight='bold')
    
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc')
    
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.8),
            fontsize=9, family='monospace')
            
    plt.tight_layout()
    
    safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    dataset_file_prefix = f"dataset_{dataset_id}_" if dataset_id is not None else ""
    file_path_png = os.path.join(save_dir, f"{dataset_file_prefix}scanline_row{row_index}_{safe_stage_name}_cropped.png")
    plt.savefig(file_path_png, dpi=300)
    plt.close()
    
    file_path_npy = os.path.join(save_dir, f"{dataset_file_prefix}scanline_row{row_index}_{safe_stage_name}_cropped.npy")
    np.save(file_path_npy, scanline)
    print(f"[{stage_name}] Cropped Scanline at row {row_index} (Dataset {dataset_id}) saved to {save_dir}.")

def main():
    utils.start_timer()
    utils.print_elapsed("크롭된 스캔라인 배치 생성 시작")

    data_path = project_root / "data" / "data_one"
    output_dir = project_root / "test" / "scanline" / "scanline_cropped"

    # dataset별 설정 매핑
    dataset_configs = {
        1: {"alpha": 0.9, "beta": 0.82, "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
        2: {"alpha": 0.9, "beta": 0.8,  "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
        3: {"alpha": 0.9, "beta": 0.81, "row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
        4: {"alpha": 0.9, "beta": 0.84, "row": 974,  "highlight": [[457, 475], [590, 607]]},
        5: {"alpha": 0.3, "beta": 0.93, "row": 1170, "highlight": [[2073, 2188]]},
        6: {"alpha": 0.9, "beta": 0.81, "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
        7: {"alpha": 0.9, "beta": 0.8,  "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
    }

    # 1부터 7까지 순회
    for k in range(1, 8):
        config = dataset_configs[k]
        input_dir = data_path / str(k)
        
        # 각 데이터셋별 폴더 경로 설정
        dataset_output_dir = output_dir / str(k)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        hdr_files = list(input_dir.glob('*.hdr'))
        
        if not hdr_files:
            print(f"경고: '{input_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        print(f"\n--- 데이터셋 [{k}] 처리 시작 (이미지 개수: {len(hdr_files)}) ---")
        
        for img_path in hdr_files:
            file_name = img_path.stem
            print(f"이미지 로딩 중: {file_name}")
            
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue
                
            is_grayscale = (img.ndim == 2)
            # 같은 intensity를 가진 3채널 이미지에서 1채널만 추출하여 사용
            if img.ndim == 3:
                img_single = img[:, :, 0]
            else:
                img_single = img
            
            # 파라미터 매핑
            opt_alpha = config["alpha"]
            opt_beta = config["beta"]
            scanline_row = config["row"]
            highlight_ranges = config["highlight"]
            
            opt_noise = 0.001
            newfattal = True
            fftsolver = True
            detail_level = 0
            
            # 오리지널 Y(휘도) = 단일 채널 입력
            Yr = img_single.astype(np.float64)
            
            # 로그 공간 변환 H
            maxLum = np.max(Yr)
            H = np.log(100.0 * Yr / maxLum + 1e-4)
            
            # 1. 1_original_HDR_Y 크롭 그래프 저장
            # 하위 Y1_X_PCT % 부터 아래로 Y1_Y_VAL 만큼 출력
            save_scanline_cropped(
                Yr, scanline_row, "1_original_HDR_Y", 
                highlight_ranges=highlight_ranges, 
                save_dir=str(dataset_output_dir),
                pct_x=Y1_X_PCT,
                offset_y=Y1_Y_VAL,
                dataset_id=k
            )
            
            # 2. 2_original_HDR_GX 연산 및 크롭 그래프 저장
            # 구체적인 출력 범위를 [Y2_MIN_VAL, Y2_MAX_VAL]로 설정
            Gx_hdr = np.empty_like(Yr)
            Gx_hdr[:, :-1] = (Yr[:, 1:] - Yr[:, :-1]) * 0.5
            Gx_hdr[:, -1] = (Yr[:, -2] - Yr[:, -1]) * 0.5
            
            save_scanline_cropped(
                Gx_hdr, scanline_row, "2_original_HDR_GX",
                highlight_ranges=highlight_ranges,
                save_dir=str(dataset_output_dir),
                y_min_val=Y2_MIN_VAL,
                y_max_val=Y2_MAX_VAL,
                dataset_id=k
            )
            
            # 4. 4_original_log_domain_GX 연산 및 크롭 그래프 저장
            # 구체적인 출력 범위를 [Y4_MIN_VAL, Y4_MAX_VAL]로 설정
            Gx_un = np.empty_like(H)
            Gx_un[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5
            Gx_un[:, -1] = (H[:, -2] - H[:, -1]) * 0.5
            
            save_scanline_cropped(
                Gx_un, scanline_row, "4_original_log_domain_GX", 
                highlight_ranges=highlight_ranges, 
                save_dir=str(dataset_output_dir),
                y_min_val=Y4_MIN_VAL,
                y_max_val=Y4_MAX_VAL,
                dataset_id=k
            )
            
            # 톤 매핑 실행 (save_scanline 중복 방지를 위해 scanline_row=None으로 전달)
            L_out = pfstmo_fattal02(
                img_single,
                opt_alpha, opt_beta, opt_noise,
                newfattal, fftsolver, detail_level,
                scanline_row=None, highlight_ranges=None,
                save_dir=None
            )
            
            # 포맷 변환 및 클리핑 (8bit 단일 채널 이미지)
            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)
                
            save_name = f"dataset_{k}_{file_name}_a{opt_alpha}_b{opt_beta}.png"
            save_path = dataset_output_dir / save_name
            cv2.imwrite(str(save_path), out_img_8bit)
            print(f"결과 이미지 저장 완료: {save_path}")
 
    utils.print_elapsed("모든 작업 종료")

if __name__ == "__main__":
    main()
