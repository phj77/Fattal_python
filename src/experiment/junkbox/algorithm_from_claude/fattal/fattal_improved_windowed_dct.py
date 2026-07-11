"""
fattal_improved_windowed_dct.py

방법론 2: Windowed DCT (Overlap-Add) + Smooth HPF

[핵심 원리]
전체 이미지를 한 번에 DCT하면 글로벌 고유 모드가 이미지 전체에 전파되어
ringing artifact를 유발한다. 본 방법론은 이를 방지하기 위해:

1. Divergence 행렬을 겹침 윈도잉(Overlap-Add) 방식으로 블록 단위 분할
2. 각 블록에 Hann/Tukey 윈도우를 적용하여 블록 경계 불연속 제거
3. 블록별 DCT Poisson 풀이 + smooth Butterworth HPF 적용
4. Overlap-Add로 블록 결과를 합성하여 전체 U 복원

이렇게 하면 먼 거리의 강한 에지가 다른 영역에 미치는 영향을 최소화하고,
윈도잉이 자연스러운 anti-aliasing 역할을 하여 Gibbs 현상을 방지한다.

[파이프라인]
1. 표준 Fattal: 로그 변환 → 가우시안 피라미드 → 감쇠맵(FI) → 감쇠 그래디언트
2. Divergence 계산
3. Windowed DCT Overlap-Add Poisson 풀이
4. exp 복원 → 정규화
5. (선택) CLAHE로 국소 대비 미세 향상
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

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(src_dir)
sys.path.append(current_dir)
from utils import utils
from fattal import pde_fft
from processing.gamma_correction import Frame, apply_gamma_frame

# PyFFTW imports (pde_fft에서 이미 사용 중)
import pyfftw
import pyfftw.interfaces.scipy_fft as fftw_fft
import multiprocessing

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()


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
# Windowed DCT Overlap-Add Poisson Solver
# ===================================================================

def solve_block_poisson_dct(block):
    """
    단일 블록에 대해 DCT-I 기반 Poisson 풀이를 수행한다.
    pde_fft.solve_pde_fft의 핵심 로직을 블록 단위로 적용.
    """
    h, w = block.shape

    # 경계 호환성 조정
    block = pde_fft.make_compatible_boundary(block)

    # normal -> eigenvalue space (DCT-I)
    F_tr = pde_fft.transform_normal2ev(block)

    # 고유값 계산
    l1 = pde_fft.get_lambda(h)
    l2 = pde_fft.get_lambda(w)
    denom = l1[:, None] + l2[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        F_tr = np.where(denom == 0, 0, F_tr / denom)
    F_tr[0, 0] = 0.0

    # eigenvalue -> normal space (inverse DCT-I)
    U_block = pde_fft.transform_ev2normal(F_tr)
    U_block -= np.max(U_block)

    return U_block


def butterworth_hpf_2d(h, w, cutoff_ratio=0.02, order=2):
    """
    2D Butterworth High-Pass Filter.
    Gaussian HPF보다 transition band가 완만하여 Gibbs 현상 감소.

    Args:
        h, w: 블록 크기
        cutoff_ratio: 차단 주파수 비율 (0~1, 작을수록 더 많은 저주파 제거)
        order: 필터 차수 (클수록 급격한 차단)
    """
    ky = np.arange(h, dtype=np.float64) / h
    kx = np.arange(w, dtype=np.float64) / w
    D = np.sqrt(ky[:, None]**2 + kx[None, :]**2)
    D[0, 0] = 1e-10  # 0으로 나누기 방지

    H_filter = 1.0 / (1.0 + (cutoff_ratio / D) ** (2 * order))
    H_filter[0, 0] = 0.0  # DC 완전 제거
    return H_filter.astype(np.float32)


def solve_poisson_windowed_ola(DivG, block_size=512, overlap_ratio=0.5,
                                hpf_cutoff=0.02, hpf_order=2):
    """
    Windowed DCT Overlap-Add Poisson Solver.

    1. DivG를 겹침 블록으로 분할
    2. 각 블록에 2D Hann 윈도우 적용
    3. 블록별 DCT Poisson 풀이 + Butterworth HPF
    4. Overlap-Add로 전체 U 복원

    Args:
        DivG: 전체 이미지의 divergence 행렬
        block_size: 블록 크기 (정사각형)
        overlap_ratio: 오버랩 비율 (0.5 = 50%)
        hpf_cutoff: Butterworth HPF 차단 주파수 비율
        hpf_order: Butterworth HPF 차수
    """
    H, W = DivG.shape
    step = int(block_size * (1.0 - overlap_ratio))

    # 2D Hann 윈도우 생성
    hann_1d = np.hanning(block_size).astype(np.float64)
    window_2d = np.outer(hann_1d, hann_1d)

    # Butterworth HPF 생성 (블록 크기에 맞게)
    hpf = butterworth_hpf_2d(block_size, block_size,
                              cutoff_ratio=hpf_cutoff, order=hpf_order)

    # 출력 버퍼
    U_acc = np.zeros((H, W), dtype=np.float64)
    W_acc = np.zeros((H, W), dtype=np.float64)  # 윈도우 가중치 누적

    # 블록 위치 계산
    y_starts = list(range(0, H - block_size + 1, step))
    x_starts = list(range(0, W - block_size + 1, step))

    # 마지막 블록이 이미지 끝에 닿지 않으면 추가
    if y_starts[-1] + block_size < H:
        y_starts.append(H - block_size)
    if x_starts[-1] + block_size < W:
        x_starts.append(W - block_size)

    total_blocks = len(y_starts) * len(x_starts)
    print(f"    [OLA] blocks: {len(y_starts)}x{len(x_starts)} = {total_blocks}, "
          f"block_size={block_size}, step={step}")

    block_count = 0
    for y0 in y_starts:
        for x0 in x_starts:
            # 블록 추출 + 윈도잉
            block = DivG[y0:y0+block_size, x0:x0+block_size].copy()
            block_windowed = block * window_2d

            # 블록 Poisson 풀이
            U_block = solve_block_poisson_dct(block_windowed)

            # Butterworth HPF 적용 (DCT 도메인에서 직접)
            # U_block을 다시 DCT → HPF → IDCT
            F_tr = pde_fft.transform_normal2ev(U_block)
            F_tr *= hpf
            U_block = pde_fft.transform_ev2normal(F_tr)

            # 윈도우 가중치 적용 후 누적
            U_acc[y0:y0+block_size, x0:x0+block_size] += U_block * window_2d
            W_acc[y0:y0+block_size, x0:x0+block_size] += window_2d ** 2

            block_count += 1
            if block_count % 10 == 0 or block_count == total_blocks:
                print(f"    [OLA] {block_count}/{total_blocks} blocks done",
                      end='\r')

    print()

    # 가중치로 정규화 (0 방지)
    W_acc = np.maximum(W_acc, 1e-10)
    U = U_acc / W_acc

    # 최대값이 0이 되도록 정규화
    U -= np.max(U)

    return U.astype(np.float32)


# ===================================================================
# 후처리: CLAHE
# ===================================================================

def apply_clahe(img_float, clip_limit=1.5, tile_grid_size=(16, 16)):
    """CLAHE: 국소 대비 향상."""
    img_8bit = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_8bit)
    return enhanced.astype(np.float32) / 255.0


# ===================================================================
# 톤 매핑 (Fattal + Windowed DCT OLA)
# ===================================================================

def tmo_windowed_dct(Y, alfa, beta, noise,
                     block_size=512,
                     overlap_ratio=0.5,
                     hpf_cutoff=0.02,
                     hpf_order=2,
                     clahe_clip=1.5,
                     clahe_tile=(32, 32)):
    """
    Fattal 톤 매핑 + Windowed DCT Overlap-Add.
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

    # --- Windowed DCT Overlap-Add Poisson 풀이 ---
    print(f"    [Poisson] Windowed DCT OLA (block={block_size}, "
          f"overlap={overlap_ratio}, hpf_cutoff={hpf_cutoff}) ...")
    U = solve_poisson_windowed_ola(
        DivG,
        block_size=block_size,
        overlap_ratio=overlap_ratio,
        hpf_cutoff=hpf_cutoff,
        hpf_order=hpf_order,
    )

    # --- exp 복원 및 정규화 ---
    L = np.exp(U)
    min_val = np.percentile(L, 0.1)
    max_val = np.percentile(L, 99.5)
    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1).astype(np.float32)

    # --- 후처리: CLAHE ---
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
    print(" Method 2: Windowed DCT (Overlap-Add) + Smooth HPF")
    print("=" * 60)

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "data_one")

    date_str = time.strftime("%Y%m%d")
    output_base = os.path.join(
        project_root, "test", f"{date_str}_WindowedDCT_OLA")
    output_images = os.path.join(output_base, "enhanced_images")
    output_scanlines = os.path.join(output_base, "scanlines")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_scanlines, exist_ok=True)

    # OLA 파라미터
    BLOCK_SIZE = 512
    OVERLAP_RATIO = 0.5
    HPF_CUTOFF = 0.02
    HPF_ORDER = 2
    CLAHE_CLIP = 1.5
    CLAHE_TILE = (32, 32)

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
        "Methodology 2: Windowed DCT (Overlap-Add) + Smooth HPF")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("[Parameters]")
    report_lines.append("- PDE Solver: Windowed DCT Overlap-Add")
    report_lines.append(f"- Block Size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    report_lines.append(f"- Overlap Ratio: {OVERLAP_RATIO} ({int(OVERLAP_RATIO*100)}%)")
    report_lines.append(f"- HPF: Butterworth (cutoff={HPF_CUTOFF}, order={HPF_ORDER})")
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

            # 톤 매핑 (Windowed DCT OLA)
            L = tmo_windowed_dct(
                Yr,
                config["alpha"],
                config["beta"],
                noise=0.001,
                block_size=BLOCK_SIZE,
                overlap_ratio=OVERLAP_RATIO,
                hpf_cutoff=HPF_CUTOFF,
                hpf_order=HPF_ORDER,
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
                title_suffix=f"- Image{k} (Windowed DCT OLA)",
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
    report_lines.append("- Ringing: Expected REDUCED (windowed blocks)")
    report_lines.append("- Low-freq bias: Expected REDUCED (Butterworth HPF per block)")
    report_lines.append("- Defect contrast: Enhanced (CLAHE)")

    report_path = os.path.join(output_base, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*60}")
    print(f" Done ({total_elapsed:.1f}s total)")
    print(f" Output: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_methodology()
