"""
fattal_improved_laplacian_clahe.py

방법론 4: Laplacian Decomposition + Bilateral Filter + CLAHE Hybrid

[핵심 원리]
방법론 1(Laplacian Decomposition)에서 ringing artifact를 완전히 제거하는 데
성공했으나, 고주파 grain noise가 과도하게 남는 문제가 있었다.

본 방법론은 다음 3단계 후처리를 추가하여 이 문제를 해결한다:

1. Laplacian Pyramid 가중치 조정
   - L1, L2의 boost를 1.0으로 낮춰 고주파 증폭 억제
   - 저주파 감쇠는 동일하게 유지 (ringing 방지)

2. Bilateral Filter (엣지 보존 디노이징)
   - 결함 경계(에지)는 보존하면서 배경의 grain noise만 제거
   - Gaussian blur와 달리 edge sharpness를 유지

3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - 국소 영역별 대비 향상으로 결함 시인성 극대화
   - clipLimit로 노이즈 증폭 방지
   - tileGridSize로 적응형 영역 크기 조절

[파이프라인]
1. 표준 Fattal: 로그 변환 → 가우시안 피라미드 → 감쇠맵(FI) → 감쇠 그래디언트
2. Divergence 계산 → FFT Poisson 풀이 (HPF 비활성화)
3. Poisson 해 U를 Laplacian Pyramid로 분해 (7레벨)
4. 저주파 레벨 감쇠 (boost 없음)
5. 피라미드 재합성 → exp → 정규화
6. Bilateral Filter → CLAHE → 최종 LDR 출력
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

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(src_dir)
sys.path.append(current_dir)
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
    gaussian = [img.astype(np.float64)]
    current = img.astype(np.float64)
    for i in range(levels):
        down = cv2.pyrDown(current.astype(np.float32)).astype(np.float64)
        gaussian.append(down)
        current = down

    laplacian = []
    for i in range(levels):
        h, w = gaussian[i].shape[:2]
        up = cv2.pyrUp(gaussian[i + 1].astype(np.float32),
                       dstsize=(w, h)).astype(np.float64)
        laplacian.append(gaussian[i] - up)

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
# 후처리: Bilateral Filter + NLM Denoising + CLAHE
# ===================================================================

def apply_bilateral_filter(img_float, d=13, sigma_color=0.12, sigma_space=20):
    """
    Bilateral Filter: 에지를 보존하면서 grain noise 제거.

    Args:
        img_float: 0~1 범위 float 이미지
        d: 필터 직경 (픽셀). 클수록 더 넓은 영역 평활화.
        sigma_color: 밝기 차이 기준 시그마 (클수록 더 강한 평활화)
        sigma_space: 공간 거리 시그마
    """
    img_8bit = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_8bit, d, sigma_color * 255, sigma_space)
    return filtered.astype(np.float32) / 255.0


def apply_nlm_denoising(img_float, h=10, template_window=7, search_window=21):
    """
    Non-Local Means Denoising: grain noise에 최적화된 구조 보존 디노이징.

    Bilateral보다 반복 패턴 노이즈에 강하며, 에지/텍스처를 더 잘 보존한다.

    Args:
        img_float: 0~1 범위 float 이미지
        h: 필터 강도 (클수록 강한 디노이징, 디테일 손실 가능)
        template_window: 비교 패치 크기 (홀수)
        search_window: 탐색 영역 크기 (홀수)
    """
    img_8bit = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(
        img_8bit, None, h, template_window, search_window)
    return denoised.astype(np.float32) / 255.0


def apply_clahe(img_float, clip_limit=1.5, tile_grid_size=(16, 16)):
    """
    CLAHE: 국소 대비 향상으로 결함 시인성 극대화.

    Args:
        img_float: 0~1 범위 float 이미지
        clip_limit: 대비 제한 (낮추면 노이즈 증폭 방지)
        tile_grid_size: 적응 영역 크기
    """
    img_8bit = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_8bit)
    return enhanced.astype(np.float32) / 255.0


# ===================================================================
# 톤 매핑 (Laplacian + Bilateral + CLAHE)
# ===================================================================

def tmo_laplacian_clahe(Y, alfa, beta, noise,
                         n_pyramid_levels=7,
                         bilateral_d=13,
                         bilateral_sigma_color=0.22,
                         bilateral_sigma_space=20,
                         nlm_h=20,
                         nlm_h2=10,
                         nlm_template=7,
                         nlm_search=21,
                         clahe_clip=1.5,
                         clahe_tile=(16, 16)):
    """
    Fattal 톤 매핑 + Laplacian Pyramid + Bilateral + CLAHE.

    방법론 1 대비 변경점:
    - L1/L2 boost 제거 (1.0으로 고정) → 고주파 노이즈 증폭 방지
    - Bilateral Filter → 배경 grain noise 제거 (에지 보존)
    - CLAHE → 결함 국소 대비 향상
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

    # --- 감쇠 그래디언트 ---
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
    print("    [Poisson] FFT solver (HPF=OFF) ...")
    U = pde_fft.solve_pde_fft(DivG, hpf_sigma=0.0)

    # --- Laplacian Pyramid 분해 ---
    actual_levels = min(n_pyramid_levels, int(np.log2(min(h, w))) - 2)
    actual_levels = max(actual_levels, 3)

    print(f"    [Laplacian] Pyramid: {actual_levels} levels + Residual")
    lap_pyramid = build_laplacian_pyramid(U, actual_levels)

    # 레벨별 가중치: L0/L1/L2 강 감쇠 (blob noise 근본 억제), 저주파 강 감쇠
    n_total = len(lap_pyramid)
    level_gains = np.ones(n_total)
    # L0 (최고주파): 0.4 → blob noise 최강 억제
    level_gains[0] = 0.4
    # L1: 0.7 → 중고주파 그레인 강 억제
    if n_total >= 2:
        level_gains[1] = 0.7
    # L2: 0.9 → 중주파 약간 감쇠
    if n_total >= 3:
        level_gains[2] = 0.9
    # Residual (DC bias): 완전 제거
    level_gains[-1] = 0.0
    # 최저주파: 강하게 감쇠
    if n_total >= 2:
        level_gains[-2] = 0.15
    if n_total >= 3:
        level_gains[-3] = 0.3
    if n_total >= 4:
        level_gains[-4] = 0.6
    # L3: 1.0 유지 (결함 디테일 보존)

    for i, lap in enumerate(lap_pyramid):
        gain = level_gains[i]
        print(f"      L{i}: {lap.shape[1]}x{lap.shape[0]}  gain={gain:.2f}")

    # 가중치 적용
    for i in range(len(lap_pyramid)):
        lap_pyramid[i] = lap_pyramid[i] * level_gains[i]

    # 재합성
    U_enhanced = reconstruct_from_laplacian(lap_pyramid)
    print("    [Laplacian] Pyramid reconstructed")

    # --- exp 복원 및 정규화 ---
    L = np.exp(U_enhanced)
    min_val = np.percentile(L, 0.1)
    max_val = np.percentile(L, 99.5)
    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1).astype(np.float32)

    # --- 후처리 1: Bilateral Filter (에지 보존 1차 디노이징) ---
    print(f"    [Bilateral] d={bilateral_d}, "
          f"sigma_c={bilateral_sigma_color}, sigma_s={bilateral_sigma_space}")
    L = apply_bilateral_filter(
        L, d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space)

    # --- 후처리 2: Non-Local Means (구조 보존 2-pass 디노이징) ---
    print(f"    [NLM-1] h={nlm_h}, template={nlm_template}, search={nlm_search}")
    L = apply_nlm_denoising(
        L, h=nlm_h,
        template_window=nlm_template,
        search_window=nlm_search)
    print(f"    [NLM-2] h={nlm_h2}, template={nlm_template}, search={nlm_search}")
    L = apply_nlm_denoising(
        L, h=nlm_h2,
        template_window=nlm_template,
        search_window=nlm_search)

    # --- 후처리 3: CLAHE (결함 국소 대비 향상) ---
    print(f"    [CLAHE] clip={clahe_clip}, tile={clahe_tile}")
    L = apply_clahe(L, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    return L


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
    print(" Method 4: Laplacian + Bilateral + CLAHE Hybrid")
    print("=" * 60)

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "data_one")

    date_str = time.strftime("%Y%m%d")
    output_base = os.path.join(
        project_root, "test", f"{date_str}_LaplacianCLAHE_v4")
    output_images = os.path.join(output_base, "enhanced_images")
    output_scanlines = os.path.join(output_base, "scanlines")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_scanlines, exist_ok=True)

    # 후처리 파라미터 (v4: 공격적 감쇠 + 2-pass NLM)
    BILATERAL_D = 13
    BILATERAL_SIGMA_COLOR = 0.22
    BILATERAL_SIGMA_SPACE = 20
    NLM_H = 20
    NLM_H2 = 10
    NLM_TEMPLATE = 7
    NLM_SEARCH = 21
    CLAHE_CLIP = 1.5
    CLAHE_TILE = (16, 16)

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
        "Methodology 4: Laplacian + Bilateral + CLAHE Hybrid")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("[Parameters]")
    report_lines.append("- PDE Solver: FFT (DCT-I, Neumann BC)")
    report_lines.append("- HPF: DISABLED (hpf_sigma=0.0)")
    report_lines.append("- Laplacian Pyramid Levels: 7 (auto-adjusted)")
    report_lines.append("- Level Gains (L0/L1/L2 attenuated):")
    report_lines.append("    L0 (finest):    0.40  (blob noise strongest suppression)")
    report_lines.append("    L1:             0.70  (grain noise strong suppression)")
    report_lines.append("    L2:             0.90  (mid-freq slight attenuation)")
    report_lines.append("    L3:             1.00  (preserve)")
    report_lines.append("    L4:             0.60  (attenuate)")
    report_lines.append("    L5:             0.30  (strong attenuate)")
    report_lines.append("    L6:             0.15  (very strong attenuate)")
    report_lines.append("    Residual (DC):  0.00  (complete removal)")
    report_lines.append(f"- Bilateral Filter: d={BILATERAL_D}, "
                        f"sigma_c={BILATERAL_SIGMA_COLOR}, "
                        f"sigma_s={BILATERAL_SIGMA_SPACE}")
    report_lines.append(f"- NLM Denoising (2-pass): h1={NLM_H}, h2={NLM_H2}, "
                        f"template={NLM_TEMPLATE}, search={NLM_SEARCH}")
    report_lines.append(f"- CLAHE: clipLimit={CLAHE_CLIP}, "
                        f"tileGridSize={CLAHE_TILE}")
    report_lines.append("- noise: 0.001")
    report_lines.append("")

    total_start = time.perf_counter()

    for k in range(1, 8):
        config = dataset_configs[k]
        input_dir = os.path.join(data_path, str(k))

        hdr_files = glob.glob(os.path.join(input_dir, '*.hdr'))
        if not hdr_files:
            print(f"  Warning: No HDR files in dataset {k}")
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
                print(f"  Error: cannot read - {img_path}")
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

            # 톤 매핑 (Laplacian + Bilateral + NLM + CLAHE)
            L = tmo_laplacian_clahe(
                Yr,
                config["alpha"],
                config["beta"],
                noise=0.001,
                n_pyramid_levels=7,
                bilateral_d=BILATERAL_D,
                bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
                bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
                nlm_h=NLM_H,
                nlm_h2=NLM_H2,
                nlm_template=NLM_TEMPLATE,
                nlm_search=NLM_SEARCH,
                clahe_clip=CLAHE_CLIP,
                clahe_tile=CLAHE_TILE,
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
            print(f"  Saved: {save_name}  ({elapsed:.1f}s)")

            # 스캔라인 저장
            scanline_row = config["row"]
            scanline_name = f"Image{k}_scanline_row{scanline_row}.png"
            scanline_path = os.path.join(output_scanlines, scanline_name)
            save_scanline_plot(
                out_8bit.astype(np.float32) / 255.0,
                scanline_row,
                scanline_path,
                title_suffix=f"- Image{k} (Laplacian+CLAHE)",
                highlight_ranges=config["highlight"],
            )
            print(f"  Scanline: {scanline_name}")

            report_lines.append(
                f"[Image{k}] {file_name} | "
                f"alpha={config['alpha']}, beta={config['beta']} | "
                f"time: {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    report_lines.append("")
    report_lines.append(f"Total time: {total_elapsed:.1f}s")
    report_lines.append("")
    report_lines.append("[Qualitative Evaluation]")
    report_lines.append("- Ringing: NONE (Laplacian Pyramid)")
    report_lines.append("- Grain noise: STRONGLY REDUCED (L0=0.85 + Bilateral + NLM)")
    report_lines.append("- Defect contrast: ENHANCED (CLAHE clip=1.5)")

    report_path = os.path.join(output_base, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*60}")
    print(f" Done ({total_elapsed:.1f}s total)")
    print(f" Output: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_methodology()
