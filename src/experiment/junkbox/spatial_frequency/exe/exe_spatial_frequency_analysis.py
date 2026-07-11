# exe_spatial_frequency_analysis.py
# Execution and visualization script for image spatial frequency analysis.

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# stdout/stderr UTF-8 설정 (Windows 환경 한글 출력 보호)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# --- Path setup ---
CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CURRENT_DIR.parent
EXPERIMENTS_ROOT = EXPERIMENT_DIR.parent
SRC_DIR = EXPERIMENTS_ROOT.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.spatial_frequency.core.spatial_frequency_analyzer import analyze_spatial_frequency

# Config로부터 기본 경로 가져오기 (실패 시 기본 경로 적용)
try:
    from exe.config.config import INPUT_DIR, test_path
    DEFAULT_INPUT_DIR = Path(INPUT_DIR)
    DEFAULT_OUTPUT_DIR = Path(test_path) / "spatial_frequency_analysis"
except Exception:
    DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "data_one" / "3"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "spatial_frequency_analysis"


def visualize_and_save_spatial_frequency(
    results: dict,
    output_dir: Path,
    show_plot: bool = False
):
    """
    공간 주파수 분석 결과를 6분할 종합 시각화 대시보드로 생성하고 저장합니다.
    """
    meta = results["meta"]
    filename = meta["filename"]
    img_stem = Path(filename).stem

    img_rgb = results["img_rgb"]
    Y = results["luminance"]
    fft_res = results["fft"]
    psd_res = results["psd"]
    band_res = results["bands"]
    sf_res = results["sf"]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{img_stem}_spatial_freq_analysis.png"

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Spatial Frequency Analysis Report - {filename}", fontsize=16, fontweight='bold', y=0.98)

    # 1. 원본 이미지 / Luminance 프리뷰
    ax1 = plt.subplot(2, 3, 1)
    if meta["log_domain"]:
        display_img = Y
        title_str = f"Luminance (Log Domain)\nRange: [{meta['min_val']:.2e}, {meta['max_val']:.2e}]"
    else:
        # 간단한 뷰어 확인용 Log 프리뷰
        display_img = np.log1p(np.maximum(Y, 0))
        title_str = f"Luminance (Log Preview)\nRange: [{meta['min_val']:.2e}, {meta['max_val']:.2e}]"
    
    im1 = ax1.imshow(display_img, cmap='gray')
    ax1.set_title(title_str, fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. 2D FFT Magnitude Spectrum (Centering 0Hz)
    ax2 = plt.subplot(2, 3, 2)
    mag_spec = fft_res["magnitude_spectrum"]
    im2 = ax2.imshow(mag_spec, cmap='viridis')
    ax2.set_title("2D FFT Magnitude Spectrum\nlog(1 + |F(u,v)|)", fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. 1D Radial Power Spectral Density (RPSD) & 1/f^alpha Slope
    ax3 = plt.subplot(2, 3, 3)
    freqs = psd_res["freqs"]
    psd_1d = psd_res["psd_1d"]
    fit_line = psd_res["fit_line"]
    alpha = psd_res["slope_alpha"]
    r2 = psd_res["fit_r2"]

    valid = freqs > 0
    ax3.loglog(freqs[valid], psd_1d[valid], 'b.-', label='Radial PSD', alpha=0.7)
    if np.any(fit_line > 0):
        ax3.loglog(freqs[valid], fit_line[valid], 'r--', linewidth=2,
                   label=f'Fit: 1/f^({alpha:.2f}) (R²={r2:.2f})')

    ax3.set_title(f"1D Radial Power Spectrum\nFalloff Exponent α = {alpha:.3f}", fontsize=12)
    ax3.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=10)
    ax3.set_ylabel("Power Spectral Density", fontsize=10)
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    ax3.legend(fontsize=10, loc='upper right')

    # 4. 3대역 주파수 분해 (Low, Mid, High Bands)
    ax4 = plt.subplot(2, 3, 4)
    lf_img = band_res["lf_image"]
    mf_img = band_res["mf_image"]
    hf_img = band_res["hf_image"]
    pct = band_res["energy_pct"]

    # 3개 대역을 1x3 나란히 결합하여 서브 그리드로 표시
    h, w = Y.shape
    combined_bands = np.zeros((h, w * 3), dtype=np.float32)
    
    # 각 대역 가시화를 위한 정규화
    def norm_vis(arr):
        a_min, a_max = np.min(arr), np.max(arr)
        if a_max > a_min:
            return (arr - a_min) / (a_max - a_min)
        return arr

    combined_bands[:, :w] = norm_vis(lf_img)
    combined_bands[:, w:2*w] = norm_vis(mf_img)
    combined_bands[:, 2*w:] = norm_vis(hf_img)

    ax4.imshow(combined_bands, cmap='inferno')
    ax4.set_title(f"Bandpass Decomposition (LF / MF / HF)\nEnergy: LF({pct['low']:.1f}%), MF({pct['mid']:.1f}%), HF({pct['high']:.1f}%)", fontsize=12)
    ax4.axis('off')

    # 5. Local Spatial Frequency Map (SF Heatmap)
    ax5 = plt.subplot(2, 3, 5)
    sf_map = sf_res["sf_map"]
    sf_index = sf_res["sf_index"]
    vmax_sf = float(np.percentile(sf_map, 99)) if np.percentile(sf_map, 99) > 0 else float(np.max(sf_map))

    im5 = ax5.imshow(sf_map, cmap='jet', vmin=0, vmax=vmax_sf)
    ax5.set_title(f"Local Spatial Frequency Map (SF)\nOverall SF RMS Score: {sf_index:.4f}", fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Frequency Energy Ratio & Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # (1) Energy Distribution Bar Chart (서브 차트)
    pos = [0.15, 0.45, 0.70, 0.40] # [left, bottom, width, height]
    sub_ax = fig.add_axes(pos)
    bands_keys = ['Low', 'Mid', 'High']
    energy_vals = [pct['low'], pct['mid'], pct['high']]
    colors = ['#4CAF50', '#FF9800', '#F44336']

    bars = sub_ax.bar(bands_keys, energy_vals, color=colors, alpha=0.85, edgecolor='black')
    sub_ax.set_ylabel("Energy Share (%)", fontsize=9)
    sub_ax.set_ylim(0, 100)
    sub_ax.set_title("Spectral Energy Distribution", fontsize=11, fontweight='bold')
    sub_ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        sub_ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (2) Summary Info Text Box (아래쪽 위치)
    summary_text = (
        f" [Summary Statistics]\n"
        f" -------------------------------------\n"
        f" • Image Resolution : {meta['width']} x {meta['height']}\n"
        f" • Dynamic Range    : {meta['dynamic_range_db']:.2f} dB\n"
        f" • Spectral Slope α : {alpha:.4f} (1/f^α)\n"
        f" • Fitting R² Score : {r2:.4f}\n"
        f" • Overall SF Index : {sf_index:.4f}\n"
        f" • Row Freq Mean    : {sf_res['rf_mean']:.4f}\n"
        f" • Col Freq Mean    : {sf_res['cf_mean']:.4f}\n"
    )
    ax6.text(0.12, 0.05, summary_text, transform=ax6.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0F4F8', edgecolor='#334155', alpha=0.9))

    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.92, wspace=0.25, hspace=0.25)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f" [Saved] Result report saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def run_spatial_frequency_analysis(
    input_path: Path = None,
    output_dir: Path = None,
    log_domain: bool = True,
    show_plot: bool = False
):
    """
    지정된 입력 경로(디렉토리 또는 단일 파일)에서 이미지를 불러와 공간 주파수 분석을 실행합니다.
    """
    if input_path is None:
        input_path = DEFAULT_INPUT_DIR

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print(f"\n=======================================================")
    print(f"  Spatial Frequency Analysis Experiment Initialized")
    print(f"  Input Path : {input_path}")
    print(f"  Output Path: {output_dir}")
    print(f"  Log Domain : {log_domain}")
    print(f"=======================================================\n")

    # 대상 파일 수집 (.hdr, .exr, .png, .jpg, .bmp 등)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            list(input_path.glob("**/*.hdr")) +
            list(input_path.glob("**/*.exr")) +
            list(input_path.glob("**/*.png")) +
            list(input_path.glob("**/*.jpg"))
        )
    else:
        print(f"[오류] 입력 경로가 존재하지 않습니다: {input_path}")
        return

    if not image_files:
        print(f"[경고] {input_path} 경로에서 분석 대상 이미지 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지 파일 분석을 시작합니다.")

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
        try:
            results = analyze_spatial_frequency(img_path, log_domain=log_domain)
            visualize_and_save_spatial_frequency(results, output_dir, show_plot=show_plot)
        except Exception as e:
            print(f"  [ERROR] {img_path.name} 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[완료] 모든 이미지의 공간 주파수 분석 및 시각화 리포트 생성이 완료되었습니다.")


if __name__ == "__main__":
    # 기본 디렉토리에 대한 분석 및 저장 실행 (show_plot=False로 저장 전용 실행)
    run_spatial_frequency_analysis(
        input_path=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        log_domain=True,
        show_plot=False
    )
