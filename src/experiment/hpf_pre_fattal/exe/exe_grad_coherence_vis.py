# exe_grad_coherence_vis.py
# Multi-Perspective Local Gradient Coherence & Consistency Analysis Script across Gaussian Pyramid Levels
#
# == 관점별 측정 기준 ==
# P1. Structure Tensor Orientation Coherence  : 고유값 비율 기반 선형 구조 일관성 (방향 부호 독립)
# P2. Vector Alignment Coherence              : 단위 방향 벡터 국소 평균 크기 (부호 포함)
# P3. Circular Variance Coherence             : 2배각 순환 분산 기반 순수 위상 집중도 (크기 가중치 無)
# P4. Magnitude-Weighted Coherence            : Percentile 크기 가중 오리엔테이션 일관성 (강한 엣지 강조)
# P5. Cross-Level Inter-Scale Consistency     : 인접 피라미드 레벨 간 Gradient 방향 코사인 유사도
# P6. Gradient Field Curl (비보존성)          : curl^2 / (div^2 + curl^2) — 작을수록 보존장에 가까움
# P7. Angular Entropy Coherence               : 국소 각도 분포 엔트로피 역수 기반 집중도 (낮을수록 혼잡)
# P8. Dominant Direction Concentration        : 국소 Orientation Histogram의 최빈 방향 집중도
# P9. Gradient Self-Similarity (Scale-Space)  : 연속 피라미드 레벨 간 Magnitude 가중 방향 일치도

import os
import sys
import cv2
import numpy as np
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
    createGaussianPyramids
)
from experiment.hpf_pre_fattal.config.config import INPUT_DIR, OUTPUT_DIR, PARAM_GRID
import utils.utils as utils

# ─── 실험 전용 설정 ─────────────────────────────────────────────────────────
PRE_HPF_SIGMA = PARAM_GRID.get('pre_hpf_sigma', [0.010])[0]

# Angular Entropy Histogram 분할 수 (P7, P8)
N_BINS = 36   # 10° 단위 (360°/36 = 10°/bin)
# ─────────────────────────────────────────────────────────────────────────────

# --- Parameters (from config.py) ---
fftsolver = PARAM_GRID['fftsolver'][0]
MSIZE = 8 if fftsolver else 32


# ──────────────────────────────────────────────────────────────────────────────
# [공통 유틸 함수]
# ──────────────────────────────────────────────────────────────────────────────

def calculate_gradient_components(H: np.ndarray, k: int):
    """
    Fattal TMO와 동일한 방식으로 x/y 방향 Gradient (gx, gy) 및 Magnitude를 계산.
    divider = 2^(k+1) 은 Fattal 알고리즘과의 호환성을 위한 레벨 스케일 normalization.
    """
    divider = 2.0 ** (k + 1)

    gx = np.empty_like(H, dtype=np.float32)
    gx[:, 0]    = (H[:, 1]  - H[:, 0])    / divider
    gx[:, -1]   = (H[:, -1] - H[:, -2])   / divider
    gx[:, 1:-1] = (H[:, 2:] - H[:, :-2])  / (2.0 * divider)

    gy = np.empty_like(H, dtype=np.float32)
    gy[0, :]    = (H[1, :]  - H[0, :])    / divider
    gy[-1, :]   = (H[-1, :] - H[-2, :])   / divider
    gy[1:-1, :] = (H[2:, :] - H[:-2, :])  / (2.0 * divider)

    grad_mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, grad_mag


def get_gaussian_kernel_size(sigma: float) -> int:
    ksize = int(np.ceil(sigma * 3) * 2 + 1)
    ksize = max(3, ksize)
    return ksize | 1  # 홀수 보장


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    ksize = get_gaussian_kernel_size(sigma)
    return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# [P1] Structure Tensor Orientation Coherence
#  · 정의: C = sqrt((Jxx-Jyy)^2 + 4*Jxy^2) / (Jxx+Jyy+eps)
#  · 범위: [0, 1] — 1일수록 국소 영역의 gradient가 단일 방향으로 정렬됨
#  · 특징: 방향 부호와 무관 (180° 회전 불변), 선형 엣지/텍스처 감지에 최적
# ──────────────────────────────────────────────────────────────────────────────
def compute_p1_structure_tensor(gx: np.ndarray, gy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    Jxx = gaussian_smooth(gx**2, sigma)
    Jyy = gaussian_smooth(gy**2, sigma)
    Jxy = gaussian_smooth(gx * gy, sigma)
    coherence = np.sqrt((Jxx - Jyy)**2 + 4.0 * Jxy**2) / (Jxx + Jyy + 1e-8)
    return np.clip(coherence, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P2] Vector Alignment Coherence
#  · 정의: R = ||mean(u_x, u_y)||, 여기서 (u_x, u_y) = (gx, gy) / ||(gx,gy)||
#  · 범위: [0, 1] — 1일수록 같은 방향의 벡터가 모임, 반대 방향 혼재 시 감소
#  · 특징: 방향 부호 고려 (0°~360°), P1과 달리 방향의 sign을 구분
# ──────────────────────────────────────────────────────────────────────────────
def compute_p2_vector_alignment(gx: np.ndarray, gy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    ux, uy = gx / mag, gy / mag
    mean_ux = gaussian_smooth(ux, sigma)
    mean_uy = gaussian_smooth(uy, sigma)
    return np.clip(np.sqrt(mean_ux**2 + mean_uy**2), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P3] Circular Variance Coherence (2배각 순환 통계)
#  · 정의: A = ||mean(cos(2θ), sin(2θ))|| (θ = arctan2(gy, gx))
#  · 범위: [0, 1] — gradient 크기와 무관, 순수 위상/각도의 집중도
#  · 특징: 모든 픽셀이 동일 가중치, 평탄 영역 노이즈 각도에 민감할 수 있음
# ──────────────────────────────────────────────────────────────────────────────
def compute_p3_circular_variance(gx: np.ndarray, gy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    theta = np.arctan2(gy, gx)
    mean_cos2 = gaussian_smooth(np.cos(2.0 * theta), sigma)
    mean_sin2 = gaussian_smooth(np.sin(2.0 * theta), sigma)
    return np.clip(np.sqrt(mean_cos2**2 + mean_sin2**2), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P4] Magnitude-Weighted Coherence (Percentile 크기 가중)
#  · 정의: C_weighted = C_tensor * (mag / percentile_99(mag))
#  · 범위: [0, 1] — 강한 엣지 영역에서만 일관성 값이 높게 나타남
#  · 개선: 전역 max 대신 99th percentile을 사용하여 이상치 영향 제거
# ──────────────────────────────────────────────────────────────────────────────
def compute_p4_magnitude_weighted(gx: np.ndarray, gy: np.ndarray, grad_mag: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    c_tensor = compute_p1_structure_tensor(gx, gy, sigma)
    p99 = np.percentile(grad_mag, 99) + 1e-8
    mag_norm = np.clip(grad_mag / p99, 0.0, 1.0)
    return np.clip(c_tensor * mag_norm, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P5] Cross-Level Inter-Scale Consistency (피라미드 레벨 간 방향 일관성)
#  · 정의: CosSim(k, k+1) = u_k · upsample(u_{k+1})
#  · 범위: [-1, 1] — +1: 동일 방향, -1: 정반대 방향
#  · 특징: 스케일 변화 시 gradient 방향의 공간적 보존도 측정
# ──────────────────────────────────────────────────────────────────────────────
def compute_p5_cross_level(gx_k: np.ndarray, gy_k: np.ndarray,
                            gx_next: np.ndarray, gy_next: np.ndarray) -> np.ndarray:
    th, tw = gx_k.shape
    gx_up = cv2.resize(gx_next, (tw, th), interpolation=cv2.INTER_LINEAR)
    gy_up = cv2.resize(gy_next, (tw, th), interpolation=cv2.INTER_LINEAR)

    ux_k  = gx_k  / (np.sqrt(gx_k**2  + gy_k**2)  + 1e-8)
    uy_k  = gy_k  / (np.sqrt(gx_k**2  + gy_k**2)  + 1e-8)
    ux_up = gx_up / (np.sqrt(gx_up**2 + gy_up**2) + 1e-8)
    uy_up = gy_up / (np.sqrt(gx_up**2 + gy_up**2) + 1e-8)

    return np.clip(ux_k * ux_up + uy_k * uy_up, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P6] Gradient Field Curl Ratio (Curl 기반 비보존성)
#  · 정의: curl_ratio = curl^2 / (div^2 + curl^2 + eps)
#  · 범위: [0, 1] — 0에 가까울수록 curl이 없음 → 완전 보존장 (이상적 gradient 필드)
#  · BUG FIX: 이전 버전에서 분자/분모가 반전되어 있었음. curl이 작을수록 0에 가까워야 함
# ──────────────────────────────────────────────────────────────────────────────
def compute_p6_curl_ratio(gx: np.ndarray, gy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    dgx_dx = np.gradient(gx.astype(np.float64), axis=1).astype(np.float32)
    dgy_dy = np.gradient(gy.astype(np.float64), axis=0).astype(np.float32)
    dgy_dx = np.gradient(gy.astype(np.float64), axis=1).astype(np.float32)
    dgx_dy = np.gradient(gx.astype(np.float64), axis=0).astype(np.float32)

    div  = dgx_dx + dgy_dy   # 발산: 확산/수렴 구조
    curl = dgy_dx - dgx_dy   # 회전: 이상적 gradient 필드에서는 0

    div_sq  = gaussian_smooth(div**2,  sigma)
    curl_sq = gaussian_smooth(curl**2, sigma)

    # curl_ratio: 0이면 완전 보존장, 1이면 완전 회전장
    ratio = curl_sq / (div_sq + curl_sq + 1e-8)
    return np.clip(ratio, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P7] Angular Entropy Coherence (국소 각도 분포 엔트로피 기반 집중도)
#  · 정의: H_normalized = Shannon Entropy / log(N_bins), Coherence = 1 - H_normalized
#  · 범위: [0, 1] — 1일수록 하나의 방향으로 집중 (낮은 엔트로피)
#  · 특징: Histogram 기반이므로 각도 분포 전체 모양을 반영, P3보다 다봉(multi-modal) 감지 우수
# ──────────────────────────────────────────────────────────────────────────────
def compute_p7_angular_entropy(gx: np.ndarray, gy: np.ndarray,
                                sigma: float = 1.5, n_bins: int = N_BINS,
                                patch_size: int = 9) -> np.ndarray:
    """
    벡터화된 box filter 방식으로 국소 각도 엔트로피를 계산합니다.
    각 angle bin에 대한 one-hot 마스크를 만들고 cv2.blur로 국소 빈도를 구한 뒤 엔트로피를 산출합니다.
    """
    h, w = gx.shape
    theta = np.arctan2(gy, gx)  # [-pi, pi]
    theta_idx = ((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int32)
    theta_idx = np.clip(theta_idx, 0, n_bins - 1)

    log_n_bins = np.log(float(n_bins))
    ksize = (patch_size, patch_size)

    # 각 bin별 one-hot → box filter → 국소 히스토그램 스택 [n_bins, h, w]
    local_hist = np.zeros((n_bins, h, w), dtype=np.float32)
    for b in range(n_bins):
        mask = (theta_idx == b).astype(np.float32)
        local_hist[b] = cv2.blur(mask, ksize, borderType=cv2.BORDER_REFLECT)

    # 정규화 → 확률
    total = local_hist.sum(axis=0, keepdims=True) + 1e-8
    p = local_hist / total

    # Shannon Entropy: -sum(p * log(p))  (p=0 항은 0으로 처리)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log(p), 0.0)
    entropy = -np.sum(p * log_p, axis=0) / log_n_bins  # 정규화 [0, 1]

    coherence = 1.0 - np.clip(entropy, 0.0, 1.0)
    return coherence


# ──────────────────────────────────────────────────────────────────────────────
# [P8] Dominant Direction Concentration (최빈 방향 집중도)
#  · 정의: max_bin_count / total_count (국소 히스토그램에서 최빈 방향의 비율)
#  · 범위: [0, 1] — 1에 가까울수록 한 방향으로 집중
#  · 특징: 가장 직관적인 방향 집중도, 다봉 분포에도 강건
# ──────────────────────────────────────────────────────────────────────────────
def compute_p8_dominant_direction(gx: np.ndarray, gy: np.ndarray,
                                   n_bins: int = N_BINS,
                                   patch_size: int = 9) -> np.ndarray:
    """
    벡터화된 box filter 방식으로 국소 최빈 방향 집중도를 계산합니다.
    """
    h, w = gx.shape
    theta = np.arctan2(gy, gx)
    theta_idx = ((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int32)
    theta_idx = np.clip(theta_idx, 0, n_bins - 1)

    ksize = (patch_size, patch_size)

    # 각 bin별 one-hot → box filter → 국소 히스토그램
    local_hist = np.zeros((n_bins, h, w), dtype=np.float32)
    for b in range(n_bins):
        mask = (theta_idx == b).astype(np.float32)
        local_hist[b] = cv2.blur(mask, ksize, borderType=cv2.BORDER_REFLECT)

    total = local_hist.sum(axis=0) + 1e-8
    max_bin = local_hist.max(axis=0)

    return np.clip(max_bin / total, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [P9] Magnitude-Weighted Inter-Scale Consistency (크기 가중 스케일 간 일치도)
#  · 정의: (mag_k * abs(CosSim)) 의 국소 가중 평균
#  · 범위: [0, 1]
#  · 특징: P5(CosSim)에서 강한 엣지에만 가중치를 주어, 약한 gradient 노이즈 영향 억제
# ──────────────────────────────────────────────────────────────────────────────
def compute_p9_mag_weighted_inter_scale(gx_k: np.ndarray, gy_k: np.ndarray,
                                         mag_k: np.ndarray,
                                         gx_next: np.ndarray, gy_next: np.ndarray,
                                         sigma: float = 1.5) -> np.ndarray:
    cos_sim = compute_p5_cross_level(gx_k, gy_k, gx_next, gy_next)
    abs_sim = np.abs(cos_sim)  # 방향 부호 제거: 반대 방향도 강한 일치로 인정

    p99 = np.percentile(mag_k, 99) + 1e-8
    mag_norm = np.clip(mag_k / p99, 0.0, 1.0)

    numerator   = gaussian_smooth(mag_norm * abs_sim, sigma)
    denominator = gaussian_smooth(mag_norm,           sigma) + 1e-8
    return np.clip(numerator / denominator, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [시각화 & 저장 헬퍼]
# ──────────────────────────────────────────────────────────────────────────────

def visualize_and_save_map(data_map: np.ndarray, save_path: Path,
                           title: str, cmap: str = 'plasma',
                           vmin: float = 0.0, vmax: float = 1.0,
                           cbar_label: str = "Index"):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_map, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)

    h, w = data_map.shape
    stats_text = (
        f'Size: {w}x{h}  |  '
        f'Min: {data_map.min():.4f}  |  '
        f'Max: {data_map.max():.4f}  |  '
        f'Mean: {data_map.mean():.4f}  |  '
        f'Std: {data_map.std():.4f}'
    )
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color='gray')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_opencv_normalized(map_data: np.ndarray, save_path: Path):
    m_min, m_max = map_data.min(), map_data.max()
    if m_max - m_min > 1e-8:
        norm = ((map_data - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(map_data, dtype=np.uint8)
    cv2.imwrite(str(save_path), norm)


def plot_overall_summary(perspectives_means: dict, nlevels: int, save_path: Path):
    levels = list(range(nlevels))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: 레벨 내 일관성 지표 (P1~P4, P6~P8)
    level_styles = [
        ('01_ST_Coherence',         'crimson',     'o-',  'P1 Structure Tensor C'),
        ('02_Vec_Alignment',        'royalblue',   's--', 'P2 Vector Alignment R'),
        ('03_Circular_Variance',    'darkorange',  '^-.',  'P3 Circular Variance A'),
        ('04_Mag_Weighted',         'forestgreen', 'd:',   'P4 Magnitude-Weighted'),
        ('06_Curl_Ratio',           'saddlebrown', 'p--', 'P6 Curl Ratio (↓=better)'),
        ('07_Angular_Entropy',      'teal',        'h-.',  'P7 Angular Entropy Coherence'),
        ('08_Dominant_Direction',   'purple',      'v:',   'P8 Dominant Direction'),
    ]
    for key, color, style, label in level_styles:
        if key in perspectives_means and perspectives_means[key]:
            axes[0].plot(levels, perspectives_means[key], style, color=color,
                         linewidth=2.0, label=label, markersize=6)

    axes[0].set_title('Intra-Level Coherence Metrics (P1~P4, P6~P8)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Pyramid Level (0: Fine → N: Coarse)', fontsize=11)
    axes[0].set_ylabel('Mean Metric Value', fontsize=11)
    axes[0].set_xticks(levels)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=9, loc='best')

    # 오른쪽: 레벨 간 일관성 지표 (P5, P9)
    inter_styles = [
        ('05_Cross_Level_CosSim',   'purple',      'v-',  'P5 Cross-Level CosSim [-1,1]'),
        ('09_Mag_Inter_Scale',      'darkcyan',    'D--', 'P9 Mag-Weighted Inter-Scale [0,1]'),
    ]
    for key, color, style, label in inter_styles:
        if key in perspectives_means and perspectives_means[key]:
            x_vals = levels[:len(perspectives_means[key])]
            axes[1].plot(x_vals, perspectives_means[key], style, color=color,
                         linewidth=2.0, label=label, markersize=6)

    axes[1].set_title('Inter-Level Scale Consistency (P5, P9)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Pyramid Level k → k+1', fontsize=11)
    axes[1].set_ylabel('Mean Metric Value', fontsize=11)
    axes[1].set_xticks(levels[:-1])
    axes[1].set_xticklabels([f'{k}→{k+1}' for k in levels[:-1]], fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9, loc='best')

    plt.suptitle('Comprehensive Multi-Perspective Pyramid Gradient Coherence Analysis (9 Metrics)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# [메인 처리 함수]
# ──────────────────────────────────────────────────────────────────────────────

def process_single_image(img_path: Path, data_folder_name: str, base_output_dir: Path):
    file_name = img_path.stem
    print(f"\n{'='*75}")
    print(f" [Pre-HPF {PRE_HPF_SIGMA}] 9-Perspective Local Gradient Analysis: {file_name}")
    print(f"{'='*75}")

    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [오류] 이미지를 읽을 수 없습니다: {img_path}")
        return

    img_single = img[:, :, 0] if img.ndim == 3 else img

    img_filtered = apply_high_pass_filter(img_single, pre_hpf_sigma=PRE_HPF_SIGMA)

    maxLum = np.max(img_filtered)
    H = np.log(100.0 * img_filtered / maxLum + 1e-4)

    h, w = H.shape
    nlevels = 0
    temp_mins = min(w, h)
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    nlevels = max(nlevels, 1)

    pyramids = createGaussianPyramids(H, nlevels)
    print(f"  피라미드 레벨 수: {nlevels}  (원본 크기: {w}x{h})\n")

    # 관점별 저장 서브 폴더 정의
    dirs = {
        'p1':  base_output_dir / "01_structure_tensor_coherence",
        'p2':  base_output_dir / "02_vector_alignment_coherence",
        'p3':  base_output_dir / "03_circular_variance_coherence",
        'p4':  base_output_dir / "04_magnitude_weighted_coherence",
        'p5':  base_output_dir / "05_cross_level_cos_similarity",
        'p6':  base_output_dir / "06_gradient_field_curl_ratio",
        'p7':  base_output_dir / "07_angular_entropy_coherence",
        'p8':  base_output_dir / "08_dominant_direction_concentration",
        'p9':  base_output_dir / "09_mag_weighted_inter_scale",
        'sum': base_output_dir / "00_summary_reports",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    means = {
        '01_ST_Coherence':       [],
        '02_Vec_Alignment':      [],
        '03_Circular_Variance':  [],
        '04_Mag_Weighted':       [],
        '05_Cross_Level_CosSim': [],
        '06_Curl_Ratio':         [],
        '07_Angular_Entropy':    [],
        '08_Dominant_Direction': [],
        '09_Mag_Inter_Scale':    [],
    }

    grads = []

    # ── 레벨별 관점 1~4, 6~8 계산 ──
    print(f"  {'Lvl':>4} {'Size':>12} | {'P1':>6} {'P2':>6} {'P3':>6} {'P4':>6} {'P6':>6} {'P7':>6} {'P8':>6}")
    print(f"  {'-'*4} {'-'*12} | {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for k in range(nlevels):
        H_k = pyramids[k]
        ph, pw = H_k.shape
        gx, gy, grad_mag = calculate_gradient_components(H_k, k)
        grads.append((gx, gy, grad_mag))

        p1 = compute_p1_structure_tensor(gx, gy)
        p2 = compute_p2_vector_alignment(gx, gy)
        p3 = compute_p3_circular_variance(gx, gy)
        p4 = compute_p4_magnitude_weighted(gx, gy, grad_mag)
        p6 = compute_p6_curl_ratio(gx, gy)
        p7 = compute_p7_angular_entropy(gx, gy, patch_size=min(9, min(ph, pw)))
        p8 = compute_p8_dominant_direction(gx, gy, patch_size=min(9, min(ph, pw)))

        means['01_ST_Coherence'].append(float(p1.mean()))
        means['02_Vec_Alignment'].append(float(p2.mean()))
        means['03_Circular_Variance'].append(float(p3.mean()))
        means['04_Mag_Weighted'].append(float(p4.mean()))
        means['06_Curl_Ratio'].append(float(p6.mean()))
        means['07_Angular_Entropy'].append(float(p7.mean()))
        means['08_Dominant_Direction'].append(float(p8.mean()))

        fname_base = f"level_{k:02d}_{pw}x{ph}"
        visualize_and_save_map(p1, dirs['p1'] / f"p1_st_coherence_{fname_base}.png",
                               f"P1. Structure Tensor Coherence - L{k}/{nlevels-1} ({pw}x{ph})", cmap='plasma', cbar_label="Coherence C [0,1]")
        save_opencv_normalized(p1, dirs['p1'] / f"cv_p1_{fname_base}.png")

        visualize_and_save_map(p2, dirs['p2'] / f"p2_vec_alignment_{fname_base}.png",
                               f"P2. Vector Alignment Coherence - L{k}/{nlevels-1} ({pw}x{ph})", cmap='magma', cbar_label="Alignment R [0,1]")
        save_opencv_normalized(p2, dirs['p2'] / f"cv_p2_{fname_base}.png")

        visualize_and_save_map(p3, dirs['p3'] / f"p3_circular_var_{fname_base}.png",
                               f"P3. Circular Variance Coherence - L{k}/{nlevels-1} ({pw}x{ph})", cmap='inferno', cbar_label="Circ. Var. A [0,1]")
        save_opencv_normalized(p3, dirs['p3'] / f"cv_p3_{fname_base}.png")

        visualize_and_save_map(p4, dirs['p4'] / f"p4_mag_weighted_{fname_base}.png",
                               f"P4. Magnitude-Weighted Coherence - L{k}/{nlevels-1} ({pw}x{ph})", cmap='viridis', cbar_label="Mag-Weighted [0,1]")
        save_opencv_normalized(p4, dirs['p4'] / f"cv_p4_{fname_base}.png")

        visualize_and_save_map(p6, dirs['p6'] / f"p6_curl_ratio_{fname_base}.png",
                               f"P6. Curl Ratio (↓=보존장) - L{k}/{nlevels-1} ({pw}x{ph})", cmap='hot', vmin=0, vmax=1, cbar_label="Curl Ratio [0→1 : conservative→rotational]")
        save_opencv_normalized(p6, dirs['p6'] / f"cv_p6_{fname_base}.png")

        visualize_and_save_map(p7, dirs['p7'] / f"p7_entropy_{fname_base}.png",
                               f"P7. Angular Entropy Coherence - L{k}/{nlevels-1} ({pw}x{ph})", cmap='YlOrRd', cbar_label="Entropy Coherence [0,1]")
        save_opencv_normalized(p7, dirs['p7'] / f"cv_p7_{fname_base}.png")

        visualize_and_save_map(p8, dirs['p8'] / f"p8_dominant_{fname_base}.png",
                               f"P8. Dominant Direction Concentration - L{k}/{nlevels-1} ({pw}x{ph})", cmap='cool', cbar_label="Dominant Ratio [0,1]")
        save_opencv_normalized(p8, dirs['p8'] / f"cv_p8_{fname_base}.png")

        print(f"  {k:>4d} {pw:>5d}x{ph:<5d}  | {p1.mean():6.3f} {p2.mean():6.3f} {p3.mean():6.3f} {p4.mean():6.3f} {p6.mean():6.3f} {p7.mean():6.3f} {p8.mean():6.3f}")

    # ── 레벨 간 관점 5, 9 계산 ──
    print(f"\n  {'':>4} Cross-Level Consistency (P5, P9)")
    print(f"  {'-'*55}")

    for k in range(nlevels - 1):
        gx_k, gy_k, mag_k  = grads[k]
        gx_n, gy_n, _       = grads[k + 1]
        ph, pw = gx_k.shape
        fname_base = f"L{k:02d}_to_L{k+1:02d}_{pw}x{ph}"

        p5 = compute_p5_cross_level(gx_k, gy_k, gx_n, gy_n)
        p9 = compute_p9_mag_weighted_inter_scale(gx_k, gy_k, mag_k, gx_n, gy_n)

        means['05_Cross_Level_CosSim'].append(float(p5.mean()))
        means['09_Mag_Inter_Scale'].append(float(p9.mean()))

        visualize_and_save_map(p5, dirs['p5'] / f"p5_cross_level_{fname_base}.png",
                               f"P5. Cross-Level CosSim - L{k} vs L{k+1} ({pw}x{ph})", cmap='coolwarm', vmin=-1.0, vmax=1.0, cbar_label="Cos Sim [-1, 1]")
        save_opencv_normalized(p5, dirs['p5'] / f"cv_p5_{fname_base}.png")

        visualize_and_save_map(p9, dirs['p9'] / f"p9_mag_inter_scale_{fname_base}.png",
                               f"P9. Mag-Weighted Inter-Scale - L{k} vs L{k+1} ({pw}x{ph})", cmap='copper', cbar_label="Weighted Consistency [0,1]")
        save_opencv_normalized(p9, dirs['p9'] / f"cv_p9_{fname_base}.png")

        print(f"    L{k:02d}→L{k+1:02d} ({pw:5d}x{ph:<5d}):  P5={p5.mean():.4f}  P9={p9.mean():.4f}")

    # ── 종합 요약 리포트 ──
    summary_path = dirs['sum'] / "summary_9perspectives_pyramid_coherence.png"
    plot_overall_summary(means, nlevels, summary_path)
    print(f"\n  [종합 리포트 저장] {summary_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# [main]
# ──────────────────────────────────────────────────────────────────────────────

def main():
    utils.start_timer()
    utils.print_elapsed("9-Perspective Pyramid Gradient Coherence Analysis 시작")

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
        data_folders = sorted([d for d in input_path.iterdir() if d.is_dir()], key=lambda x: x.name)
        if not data_folders:
            print(f"[오류] .hdr 파일이나 하위 디렉토리를 찾을 수 없습니다: {input_path}")
            return
        print(f"감지된 데이터 폴더 수: {len(data_folders)}\n")
        for folder in data_folders:
            hdr_files = list(folder.glob('*.hdr'))
            if not hdr_files:
                print(f"[{folder.name}] .hdr 파일 없음, 스킵")
                continue
            for hdr_file in hdr_files:
                process_single_image(hdr_file, folder.name, base_output_dir)

    utils.print_elapsed("9-Perspective Pyramid Gradient Coherence Analysis 완료")
    print(f"\n모든 관점 결과 저장 루트 디렉토리: {base_output_dir}")


if __name__ == "__main__":
    main()
