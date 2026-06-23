"""
fattal_improved_laplacian_decomposition.py

방법론 1: Multi-scale Laplacian Decomposition + Selective Reconstruction

[핵심 원리]
기존 Fattal 알고리즘의 FFT solver에서 HPF(High-Pass Filter)를 사용하면
DCT 스펙트럼 도메인의 급격한 차단으로 인해 Gibbs 현상(동심원형 ringing)이 발생한다.

본 방법론은 HPF를 완전히 제거하고, 대신 Poisson 풀이 결과 U를
**Laplacian Pyramid**로 분해하여 저주파 편향(DC bias)만 선택적으로 제거한다.

Laplacian Pyramid는 Gaussian 기반의 부드러운 주파수 응답을 사용하므로,
DCT HPF와 달리 sharp cutoff가 없어 Gibbs 현상이 원천적으로 방지된다.

[파이프라인]
1. 표준 Fattal: 로그 변환 → 가우시안 피라미드 → 감쇠맵(FI) → 감쇠 그래디언트
2. Divergence 계산 → FFT Poisson 풀이 (HPF 비활성화)
3. Poisson 해 U를 Laplacian Pyramid로 분해 (7레벨)
4. 저주파 레벨(L5, L6, Residual) 감쇠, 중주파(L1, L2) 미세 부스트
5. 피라미드 재합성 → exp → 정규화 → 최종 LDR 출력
"""

import numpy as np
import cv2
import sys
import os
import time
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import utils
from fattal import pde_fft
from processing.gamma_correction import Frame, apply_gamma_frame


# ===================================================================
# Fattal 파이프라인 핵심 함수 (fattal_tmo.py에서 복제)
# ===================================================================

def gaussianBlur(I):
    h, w = I.shape
    if w < 3 or h < 3:
        return I.copy()
    T = np.zeros_like(I)
    T[:, 1:w-1] = (2.0 * I[:, 1:w-1] + I[:, 0:w-2] + I[:, 2:w]) * 0.25
    T[:, 0] = (3.0 * I[:, 0] + I[:, 1]) * 0.25
    T[:, w-1] = (3.0 * I[:, w-1] + I[:, w-2]) * 0.25
    L = np.zeros_like(I)
    L[1:h-1, :] = (2.0 * T[1:h-1, :] + T[0:h-2, :] + T[2:h, :]) * 0.25
    L[0, :] = (3.0 * T[0, :] + T[1, :]) * 0.25
    L[h-1, :] = (3.0 * T[h-1, :] + T[h-2, :]) * 0.25
    return L


def downSample(A):
    h, w = A.shape
    nh, nw = h // 2, w // 2
    B = (A[0:2*nh:2, 0:2*nw:2] + A[1:2*nh:2, 0:2*nw:2] +
         A[0:2*nh:2, 1:2*nw:2] + A[1:2*nh:2, 1:2*nw:2]) * 0.25
    return B


def createGaussianPyramids(H, nlevels):
    pyramids = [H]
    L = gaussianBlur(H)
    for k in range(1, nlevels):
        down = downSample(L)
        pyramids.append(down)
        if k < nlevels - 1:
            L = gaussianBlur(down)
    return pyramids


def upSample(A, target_shape):
    th, tw = target_shape
    ah, aw = A.shape
    y_idx = np.clip(np.arange(th) // 2, 0, ah - 1)
    x_idx = np.clip(np.arange(tw) // 2, 0, aw - 1)
    return A[np.ix_(y_idx, x_idx)]


def calculate_gradient_mag(H, k):
    divider = 2.0 ** (k + 1)
    gx = np.empty_like(H)
    gx[:, 0] = H[:, 0] - H[:, 1]
    gx[:, -1] = H[:, -2] - H[:, -1]
    gx[:, 1:-1] = H[:, :-2] - H[:, 2:]
    gx /= divider
    gy = np.empty_like(H)
    gy[0, :] = H[0, :] - H[1, :]
    gy[-1, :] = H[-2, :] - H[-1, :]
    gy[1:-1, :] = H[:-2, :] - H[2:, :]
    gy /= divider
    return np.sqrt(gx**2 + gy**2)


def calculate_attenuation(gradient, alfa, beta, noise):
    avgGrad = np.mean(gradient)
    grad_safe = np.maximum(gradient, 1e-4)
    a = alfa * avgGrad
    return ((grad_safe + noise) / a) ** (beta - 1.0)


def calculate_level_attenuation(H, k, alfa, beta, noise):
    G = calculate_gradient_mag(H, k)
    return calculate_attenuation(G, alfa, beta, noise)


def calculateFiMatrix(values, pyramids, nlevels):
    h, w = pyramids[-1].shape
    fi = [None] * nlevels
    fi[-1] = np.ones((h, w), dtype=np.float32)
    for k in range(nlevels - 1, -1, -1):
        if values[k] is not None:
            fi[k] *= values[k]
        if k > 0:
            target_shape = pyramids[k-1].shape
            up = upSample(fi[k], target_shape)
            fi[k-1] = gaussianBlur(up)
    return fi[0]


# ===================================================================
# Laplacian Pyramid 분해 및 재합성
# ===================================================================

def build_laplacian_pyramid(img, levels):
    """
    OpenCV의 pyrDown/pyrUp를 사용하여 Laplacian Pyramid 구축.
    
    반환값: [L0, L1, ..., L_{levels-1}, Residual]
    - L0: 최고 주파수 (원본 해상도 디테일)
    - Residual: 최저 주파수 (DC + 저주파 공간 편향)
    """
    # 가우시안 피라미드 구축
    gaussian = [img.astype(np.float64)]
    current = img.astype(np.float64)
    for i in range(levels):
        down = cv2.pyrDown(current.astype(np.float32)).astype(np.float64)
        gaussian.append(down)
        current = down

    # 라플라시안 피라미드 = 가우시안 차분
    laplacian = []
    for i in range(levels):
        h, w = gaussian[i].shape[:2]
        up = cv2.pyrUp(gaussian[i + 1].astype(np.float32),
                       dstsize=(w, h)).astype(np.float64)
        laplacian.append(gaussian[i] - up)

    # 잔차(Residual) = 가장 거친 가우시안 레벨
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct_from_laplacian(pyramid):
    """Laplacian Pyramid를 재합성하여 원본 크기 이미지 복원."""
    current = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[:2]
        up = cv2.pyrUp(current.astype(np.float32),
                       dstsize=(w, h)).astype(np.float64)
        current = up + pyramid[i]
    return current


# ===================================================================
# 톤 매핑 (Laplacian Decomposition 적용)
# ===================================================================

def tmo_laplacian(Y, alfa, beta, noise, n_pyramid_levels=7, level_gains=None):
    """
    Fattal 톤 매핑 + Laplacian Pyramid를 이용한 선택적 주파수 복원.

    DCT 도메인 HPF 대신, 공간 도메인의 Laplacian Pyramid로
    저주파 편향을 부드럽게 제거하여 ringing artifact를 방지한다.
    
    Args:
        Y: 입력 휘도 (float32)
        alfa, beta, noise: Fattal 감쇠 파라미터
        n_pyramid_levels: Laplacian 피라미드 레벨 수 (기본 7)
        level_gains: 각 레벨별 가중치 리스트 (None이면 자동 설정)
    """
    h, w = Y.shape
    maxLum = np.max(Y)
    H = np.log(100.0 * Y / maxLum + 1e-4)

    # --- 표준 Fattal: 가우시안 피라미드 → 감쇠맵 ---
    MSIZE = 8
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0:
        nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)

    attenuation_vals = [None] * nlevels
    with ThreadPoolExecutor(max_workers=min(nlevels, 8)) as executor:
        futures = []
        for k in range(nlevels):
            futures.append((k, executor.submit(
                calculate_level_attenuation, pyramids[k], k, alfa, beta, noise)))
        for k, future in futures:
            attenuation_vals[k] = future.result()

    FI = calculateFiMatrix(attenuation_vals, pyramids, nlevels)

    # --- 감쇠 그래디언트 (FFT solver 방식) ---
    Gx = np.empty_like(H)
    Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (FI[:, 1:] + FI[:, :-1])
    Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (FI[:, -2] + FI[:, -1])

    Gy = np.empty_like(H)
    Gy[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5 * (FI[1:, :] + FI[:-1, :])
    Gy[-1, :] = (H[-2, :] - H[-1, :]) * 0.5 * (FI[-2, :] + FI[-1, :])

    # 다이버전스
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]
    DivG[:, 0] += Gx[:, 0]
    DivG[0, :] += Gy[0, :]

    # --- Poisson 풀이 (HPF 비활성화) ---
    print("    [Poisson] FFT solver 실행 (HPF=OFF) ...")
    U = pde_fft.solve_pde_fft(DivG, hpf_sigma=0.0)

    # --- Laplacian Pyramid 분해 ---
    actual_levels = min(n_pyramid_levels, int(np.log2(min(h, w))) - 2)
    actual_levels = max(actual_levels, 3)

    print(f"    [Laplacian] 피라미드 분해: {actual_levels} 레벨 + Residual")
    lap_pyramid = build_laplacian_pyramid(U, actual_levels)

    # 레벨별 가중치 설정
    if level_gains is None:
        n_total = len(lap_pyramid)  # actual_levels + 1 (Residual 포함)
        level_gains = np.ones(n_total)
        # Residual (DC bias): 완전 제거
        level_gains[-1] = 0.0
        # 최저주파 Laplacian 레벨: 강하게 감쇠
        if n_total >= 2:
            level_gains[-2] = 0.15
        # 그 다음 저주파: 중간 감쇠
        if n_total >= 3:
            level_gains[-3] = 0.3
        # 그 다음: 약한 감쇠
        if n_total >= 4:
            level_gains[-4] = 0.6
        # 중주파 (결함 디테일 포함 대역): 미세 부스트
        if n_total >= 6:
            level_gains[1] = 1.1
            level_gains[2] = 1.1

    for i, lap in enumerate(lap_pyramid):
        gain = level_gains[i] if i < len(level_gains) else 1.0
        print(f"      L{i}: {lap.shape[1]}x{lap.shape[0]}  gain={gain:.2f}")

    # 가중치 적용
    for i in range(len(lap_pyramid)):
        gain = level_gains[i] if i < len(level_gains) else 1.0
        lap_pyramid[i] = lap_pyramid[i] * gain

    # 재합성
    U_enhanced = reconstruct_from_laplacian(lap_pyramid)
    print("    [Laplacian] 피라미드 재합성 완료")

    # --- exp 복원 및 정규화 ---
    L = np.exp(U_enhanced)

    min_val = np.percentile(L, 0.1)
    max_val = np.percentile(L, 99.5)
    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1)

    return L.astype(np.float32)


# ===================================================================
# Scanline 저장
# ===================================================================

def save_scanline_plot(image, row_index, save_path, title_suffix="",
                       highlight_ranges=None):
    """특정 행(row)의 스캔라인을 플롯으로 저장."""
    h = image.shape[0]
    row_index = np.clip(row_index, 0, h - 1)
    scanline = image[row_index, :].astype(np.float64)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
    ax.plot(scanline, color='#1f77b4', linewidth=0.8, alpha=0.9,
            label='Intensity')

    if highlight_ranges:
        for rng in highlight_ranges:
            if len(rng) == 2:
                s, e = max(0, rng[0]), min(len(scanline) - 1, rng[1])
                ax.axvspan(s, e, color='#ff4444', alpha=0.2,
                           label=f'Defect [{s}-{e}]')

    ax.set_title(f"Scanline at Row {row_index} {title_suffix}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Column Index", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', color='#e5e5e5', linestyle=':', linewidth=0.3)

    s_min, s_max, s_mean = np.min(scanline), np.max(scanline), np.mean(scanline)
    stats = f"Min: {s_min:.4f}\nMax: {s_max:.4f}\nMean: {s_mean:.4f}"
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.8),
            fontsize=9, family='monospace')

    y_margin = (s_max - s_min) * 0.05 if s_max > s_min else 0.1
    ax.set_ylim(s_min - y_margin, s_max + y_margin)
    ax.set_xlim(0, len(scanline) - 1)
    ax.legend(loc='upper left', frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ===================================================================
# 메인 실행
# ===================================================================

def run_methodology():
    utils.start_timer()
    print("=" * 60)
    print(" 방법론 1: Multi-scale Laplacian Decomposition")
    print("=" * 60)

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "data_one")

    date_str = time.strftime("%Y%m%d")
    output_base = os.path.join(
        project_root, "test", f"{date_str}_LaplacianDecomposition")
    output_images = os.path.join(output_base, "enhanced_images")
    output_scanlines = os.path.join(output_base, "scanlines")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_scanlines, exist_ok=True)

    # 데이터셋별 파라미터
    dataset_configs = {
        1: {"alpha": 0.9, "beta": 0.82,
            "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
        2: {"alpha": 0.9, "beta": 0.8,
            "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
        3: {"alpha": 0.9, "beta": 0.81,
            "row": 955, "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
        4: {"alpha": 0.9, "beta": 0.84,
            "row": 974, "highlight": [[457, 475], [590, 607]]},
        5: {"alpha": 0.3, "beta": 0.93,
            "row": 1170, "highlight": [[2073, 2188]]},
        6: {"alpha": 0.9, "beta": 0.81,
            "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
        7: {"alpha": 0.9, "beta": 0.8,
            "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]},
    }

    report_lines = []
    report_lines.append(
        "Methodology: Multi-scale Laplacian Decomposition "
        "+ Selective Reconstruction")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("[Parameters]")
    report_lines.append("- PDE Solver: FFT (DCT-I, Neumann BC)")
    report_lines.append("- HPF: DISABLED (hpf_sigma=0.0)")
    report_lines.append("- Laplacian Pyramid Levels: 7 (auto-adjusted)")
    report_lines.append("- Level Gains (auto-configured):")
    report_lines.append("    L0 (finest):    1.00  (preserve)")
    report_lines.append("    L1:             1.10  (slight boost)")
    report_lines.append("    L2:             1.10  (slight boost)")
    report_lines.append("    L3:             1.00  (preserve)")
    report_lines.append("    L4:             0.60  (attenuate)")
    report_lines.append("    L5:             0.30  (strong attenuate)")
    report_lines.append("    L6:             0.15  (very strong attenuate)")
    report_lines.append("    Residual (DC):  0.00  (complete removal)")
    report_lines.append("- noise: 0.001")
    report_lines.append("- pre_gamma / post_gamma: 1.0 / 1.0")
    report_lines.append("- HE_weight: 0.0")
    report_lines.append("")

    total_start = time.perf_counter()

    for k in range(1, 8):
        config = dataset_configs[k]
        input_dir = os.path.join(data_path, str(k))

        hdr_files = glob.glob(os.path.join(input_dir, '*.hdr'))
        if not hdr_files:
            print(f"  경고: 데이터셋 {k}에서 HDR 파일 없음")
            continue

        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\n{'─'*50}")
            print(f"  Image{k}: {file_name}")
            print(f"  alpha={config['alpha']}, beta={config['beta']}")
            print(f"{'─'*50}")
            img_start = time.perf_counter()

            # 이미지 로드
            img = cv2.imread(
                img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"  오류: 이미지를 읽을 수 없음 - {img_path}")
                continue

            is_grayscale = (img.ndim == 2)
            if is_grayscale:
                img = np.stack([img, img, img], axis=-1)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            R = img_rgb[:, :, 0]
            G_ch = img_rgb[:, :, 1]
            B = img_rgb[:, :, 2]

            # 전처리 감마 (identity)
            pre_frame = Frame(R, G_ch, B)
            apply_gamma_frame(pre_frame, 1.0)
            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            # 휘도 계산
            Yr = 0.2126 * R_pre + 0.7152 * G_pre + 0.0722 * B_pre

            # 톤 매핑 (Laplacian Decomposition)
            L = tmo_laplacian(
                Yr,
                config["alpha"],
                config["beta"],
                noise=0.001,
                n_pyramid_levels=7,
            )

            # 그레이스케일 출력
            epsilon = 1e-4
            Y_safe = np.maximum(Yr, epsilon)
            L_safe = np.maximum(L, epsilon)
            Gray_out = np.maximum(R_pre / Y_safe, 0) * L_safe

            out_img = np.clip(Gray_out, 0.0, 1.0)
            out_8bit = (out_img * 255.0).astype(np.uint8)

            # 결과 이미지 저장
            save_name = (f"Image{k}_{file_name}"
                         f"_a{config['alpha']}_b{config['beta']}.png")
            save_path = os.path.join(output_images, save_name)
            cv2.imwrite(save_path, out_8bit)

            elapsed = time.perf_counter() - img_start
            print(f"  저장: {save_name}  ({elapsed:.1f}s)")

            # 스캔라인 저장
            scanline_row = config["row"]
            scanline_name = f"Image{k}_scanline_row{scanline_row}.png"
            scanline_path = os.path.join(output_scanlines, scanline_name)
            save_scanline_plot(
                out_8bit.astype(np.float32) / 255.0,
                scanline_row,
                scanline_path,
                title_suffix=f"— Image{k} (Laplacian Decomposition)",
                highlight_ranges=config["highlight"],
            )
            print(f"  스캔라인: {scanline_name}")

            report_lines.append(
                f"[Image{k}] {file_name} | "
                f"alpha={config['alpha']}, beta={config['beta']} | "
                f"처리시간: {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    report_lines.append("")
    report_lines.append(f"총 처리 시간: {total_elapsed:.1f}s")
    report_lines.append("")
    report_lines.append("[정성적 평가]")
    report_lines.append("- (실험 완료 후 결과 이미지를 확인하여 작성)")

    report_path = os.path.join(output_base, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*60}")
    print(f" 전체 완료 (총 {total_elapsed:.1f}초)")
    print(f" 결과: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_methodology()
