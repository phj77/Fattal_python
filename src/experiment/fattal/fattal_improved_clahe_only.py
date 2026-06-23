"""
fattal_improved_clahe_only.py

방법론 3: CLAHE 기반 직접적 로컬 대비 향상 (Fattal 후처리)

[핵심 원리]
Fattal 알고리즘의 기본 Poisson 풀이(HPF 없음)로 글로벌 톤 압축을 수행한 후,
CLAHE (Contrast Limited Adaptive Histogram Equalization)를 최종 결과에 적용하여
국소적 대비를 향상시킨다.

방법론 4(하이브리드)와의 차이점:
- Laplacian Pyramid 분해/재합성을 하지 않음
- Bilateral Filter / NLM 디노이징을 사용하지 않음
- 순수 CLAHE + Unsharp Masking만으로 후처리

[파이프라인]
1. 표준 Fattal: 로그 변환 → 가우시안 피라미드 → 감쇠맵(FI) → 감쇠 그래디언트
2. Divergence 계산 → FFT Poisson 풀이 (HPF 비활성화)
3. exp 복원 → 정규화
4. CLAHE 적용 (국소 대비 향상)
5. Unsharp Masking (결함 에지 선명도 향상)
6. 최종 LDR 출력
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
# Fattal 파이프라인 핵심 함수
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
# 후처리: CLAHE + Unsharp Masking
# ===================================================================

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


def apply_unsharp_mask(img_float, sigma=2.0, strength=0.5):
    """
    Unsharp Masking: 에지 선명도 향상.

    Args:
        img_float: 0~1 범위 float 이미지
        sigma: 가우시안 블러 시그마 (클수록 넓은 범위 선명화)
        strength: 샤프닝 강도 (0~1, 1이면 100% 적용)
    """
    # Gaussian blur로 저주파 추출
    ksize = int(sigma * 6) | 1  # 항상 홀수
    blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
    # 원본 - 블러 = 고주파 디테일
    detail = img_float - blurred
    # 원본 + strength * 고주파
    sharpened = img_float + strength * detail
    return np.clip(sharpened, 0, 1).astype(np.float32)


# ===================================================================
# 톤 매핑 (Fattal + CLAHE + Unsharp)
# ===================================================================

def tmo_clahe_only(Y, alfa, beta, noise,
                   clahe_clip=1.5,
                   clahe_tile=(32, 32),
                   unsharp_sigma=2.0,
                   unsharp_strength=0.5):
    """
    Fattal 톤 매핑 + CLAHE + Unsharp Masking.

    방법론 4 대비 변경점:
    - Laplacian Pyramid 분해/재합성 없음
    - Bilateral Filter / NLM 디노이징 없음
    - 순수 CLAHE + Unsharp Masking만 사용
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

    # --- exp 복원 및 정규화 (Laplacian Pyramid 없음) ---
    L = np.exp(U)
    min_val = np.percentile(L, 0.1)
    max_val = np.percentile(L, 99.5)
    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1).astype(np.float32)

    # --- 후처리 1: CLAHE (국소 대비 향상) ---
    print(f"    [CLAHE] clip={clahe_clip}, tile={clahe_tile}")
    L = apply_clahe(L, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    # --- 후처리 2: Unsharp Masking (에지 선명도 향상) ---
    print(f"    [Unsharp] sigma={unsharp_sigma}, strength={unsharp_strength}")
    L = apply_unsharp_mask(L, sigma=unsharp_sigma, strength=unsharp_strength)

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
    print(" Method 3: CLAHE-based Direct Local Contrast Enhancement")
    print("=" * 60)

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "data_one")

    date_str = time.strftime("%Y%m%d")
    output_base = os.path.join(
        project_root, "test", f"{date_str}_CLAHE_Only")
    output_images = os.path.join(output_base, "enhanced_images")
    output_scanlines = os.path.join(output_base, "scanlines")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_scanlines, exist_ok=True)

    # 후처리 파라미터
    CLAHE_CLIP = 1.5
    CLAHE_TILE = (32, 32)
    UNSHARP_SIGMA = 2.0
    UNSHARP_STRENGTH = 0.5

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
        "Methodology 3: CLAHE-based Direct Local Contrast Enhancement")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("[Parameters]")
    report_lines.append("- PDE Solver: FFT (DCT-I, Neumann BC)")
    report_lines.append("- HPF: DISABLED (hpf_sigma=0.0)")
    report_lines.append("- Laplacian Pyramid: NOT USED")
    report_lines.append("- Bilateral Filter: NOT USED")
    report_lines.append("- NLM Denoising: NOT USED")
    report_lines.append(f"- CLAHE: clipLimit={CLAHE_CLIP}, "
                        f"tileGridSize={CLAHE_TILE}")
    report_lines.append(f"- Unsharp Masking: sigma={UNSHARP_SIGMA}, "
                        f"strength={UNSHARP_STRENGTH}")
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

            # 톤 매핑 (CLAHE + Unsharp Only)
            L = tmo_clahe_only(
                Yr,
                config["alpha"],
                config["beta"],
                noise=0.001,
                clahe_clip=CLAHE_CLIP,
                clahe_tile=CLAHE_TILE,
                unsharp_sigma=UNSHARP_SIGMA,
                unsharp_strength=UNSHARP_STRENGTH,
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
                title_suffix=f"- Image{k} (CLAHE Only)",
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
    report_lines.append("- Ringing: May appear (no Laplacian decomposition)")
    report_lines.append("- Grain noise: Not addressed (no Bilateral/NLM)")
    report_lines.append("- Defect contrast: ENHANCED (CLAHE + Unsharp)")

    report_path = os.path.join(output_base, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*60}")
    print(f" Done ({total_elapsed:.1f}s total)")
    print(f" Output: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_methodology()
