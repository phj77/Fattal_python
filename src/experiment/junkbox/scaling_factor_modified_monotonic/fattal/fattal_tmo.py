import numpy as np
import scipy.fft as fft
import cv2
import sys
import os
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import utils
from fattal import pde_multigrid
from fattal import pde_fft


def apply_high_pass_filter(img: np.ndarray, pre_hpf_sigma: float = 0.007) -> np.ndarray:
    """
    Original 이미지에 High-Pass Filter(HPF)를 먼저 적용합니다.
    
    Args:
        img (np.ndarray): 입력 2D 이미지 (단일 채널)
        pre_hpf_sigma (float): High-Pass Filter 표준편차/강도 (0 이하일 경우 필터 미적용)
        
    Returns:
        np.ndarray: HPF가 적용된 이미지
    """
    if pre_hpf_sigma <= 0.0:
        return img.copy()

    utils.print_elapsed(f"   [Pre-HPF] High-Pass Filter 적용 시작 (sigma_ratio={pre_hpf_sigma})")
    h, w = img.shape
    sigma_spatial = pre_hpf_sigma * max(h, w)
    if sigma_spatial < 0.5:
        sigma_spatial = 0.5

    ksize = int(np.ceil(sigma_spatial * 3) * 2 + 1)
    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1

    low_pass = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma_spatial, sigmaY=sigma_spatial)
    high_pass = img.astype(np.float32) - low_pass

    # 평균 밝기를 보존하여 로그 변환 시 음수 발생 방지 및 DC offset 유지
    mean_val = np.mean(img)
    img_hpf = high_pass + mean_val

    # 0 이하의 값 클리핑 (로그 연산 안전성 확보)
    positive_mask = img > 0
    min_val = np.min(img[positive_mask]) if np.any(positive_mask) else 1e-4
    img_hpf = np.maximum(img_hpf, min_val * 0.1)

    utils.print_elapsed("   [Pre-HPF] High-Pass Filter 적용 완료")
    return img_hpf


def gaussianBlur(I):
    h, w = I.shape
    if w < 3 or h < 3:
        return I.copy()

    T = np.zeros_like(I)
    # X blur
    T[:, 1:w-1] = (2.0 * I[:, 1:w-1] + I[:, 0:w-2] + I[:, 2:w]) * 0.25
    T[:, 0] = (3.0 * I[:, 0] + I[:, 1]) * 0.25
    T[:, w-1] = (3.0 * I[:, w-1] + I[:, w-2]) * 0.25

    L = np.zeros_like(I)
    # Y blur
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


def createGaussianPyramids(H, n_pyramid_levels):
    pyramids = [H]
    L = gaussianBlur(H)

    for k in range(1, n_pyramid_levels):
        down = downSample(L)
        pyramids.append(down)
        if k < n_pyramid_levels - 1:
            L = gaussianBlur(down)

    return pyramids


def upSample(A, target_shape):
    th, tw = target_shape
    ah, aw = A.shape
    y_idx = np.clip(np.arange(th) // 2, 0, ah - 1)
    x_idx = np.clip(np.arange(tw) // 2, 0, aw - 1)
    return A[np.ix_(y_idx, x_idx)]


def calculate_gradient_mag(H, k):
    h, w = H.shape
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

    G = np.sqrt(gx**2 + gy**2)
    return G


def calculate_scaling_factor(gradient, alfa, beta, noise):
    avgGrad = np.mean(gradient)
    a = alfa * avgGrad

    mask = gradient > 0
    safe_grad = np.where(mask, gradient, 1.0)
    scaling_factor = np.where(mask, ((safe_grad + noise) / a) ** (beta - 1.0), 0.0)
    return scaling_factor


def calculate_piecewise_scaling_factor(gradient, alfa, beta, xp_ratio, y0):
    """
    구간별(Piecewise) 스케일링 팩터를 계산하는 함수.
    두 번째 구간과 세 번째 구간이 x = a 지점에서 매끄럽게(미분 가능하게) 연결됩니다.
    
    Parameters:
      gradient : np.ndarray, 기울기(gradient) 크기 배열 (x에 해당)
      xp_ratio : float, 첫 번째 구간과 두 번째 구간의 경계값 비율 (xp = xp_ratio * a)
      alfa     : float, 원본 Fattal 알고리즘의 alfa 파라미터
      y0       : float, x = 0 에서의 함수값
      beta     : float, 원본 Fattal 알고리즘의 감쇠 지수 파라미터
    """
    
    # 0 이하의 값 방지
    x = np.maximum(gradient, 1e-6)
    y = np.zeros_like(x)
    
    # 원본 알고리즘의 기준에 따라 파라미터 a를 내부에서 자동 계산
    avg_grad = np.mean(x)
    a = alfa * avg_grad
    
    # xp가 a의 xp_ratio가 되도록 계산
    xp = xp_ratio * a
    
    # --- 1. 첫 번째 구간: 0 < x < xp ---
    mask1 = x < xp
    if np.any(mask1):
        x1 = x[mask1]
        y[mask1] = -(y0 / (2.0 * xp**2)) * (x1**2) + y0

    # --- 2. 두 번째 구간: xp <= x < a ---
    mask2 = (xp <= x) & (x < a)
    if np.any(mask2):
        x2 = x[mask2]
        dx = a - xp
        
        # 정규화된 매개변수 t
        t = (x2 - xp) / dx
        
        # x = a 에서의 C1 연속성을 위해 경계 조건을 세 번째 구간의 값으로 강제 지정
        A = y0 / 2.0
        C = 1.0
        dy = C - A
        
        # 정규화된 미분값 (세 번째 구간의 x = a 미분값 기반)
        v0 = -(y0 / xp) * dx
        v1 = ((beta - 1.0) / a) * dx
        
        # 유리 함수의 내부 파라미터 계산
        w = (v0 + v1) / (2.0 * dy)
        B = A + v0 / (2.0 * w)
        
        # t를 이용한 유리 2차 함수 계산
        inv_t = 1.0 - t
        numerator = A * (inv_t**2) + 2.0 * w * B * t * inv_t + C * (t**2)
        denominator = (inv_t**2) + 2.0 * w * t * inv_t + (t**2)
        
        y[mask2] = numerator / denominator

    # --- 3. 세 번째 구간: x >= a ---
    # noise 항이 제거된 기존 Fattal 스케일링 수식 적용
    mask3 = x >= a
    if np.any(mask3):
        y[mask3] = (x[mask3] / a) ** (beta - 1.0)

    return y


def calculate_level_scaling_factor(H, k, alfa, beta, noise, xp_ratio, y0):
    G = calculate_gradient_mag(H, k)
    if k == 3:
        scaling_factor = calculate_piecewise_scaling_factor(G, alfa, beta, xp_ratio, y0)
    else:
        scaling_factor = calculate_scaling_factor(G, alfa, beta, noise)

    return scaling_factor


def calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal):
    h, w = pyramids[-1].shape
    attenuation = [None] * n_pyramid_levels

    if newfattal:
        attenuation[-1] = np.ones((h, w), dtype=np.float32)
    else:
        attenuation[-1] = np.empty((h, w), dtype=np.float32) 

    for k in range(n_pyramid_levels - 1, -1, -1):
        if scaling_factor[k] is not None:
            if newfattal:
                attenuation[k] *= scaling_factor[k]
            else:
                attenuation[k] = scaling_factor[k]
                
        if k > 0:
            target_shape = pyramids[k-1].shape
            if newfattal:
                up = upSample(attenuation[k], target_shape)
                attenuation[k-1] = gaussianBlur(up)
            else:
                attenuation[k-1] = np.empty(target_shape, dtype=np.float32)

    return attenuation[0]


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007, xp_ratio=0.05, y0=6.0):
    utils.print_elapsed("     [tmo] 시작")
    h, w = Y.shape
    
    TOP_SIZE = 2**5 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_after_high_pass_filter_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
        if fftsolver:
            Gx_hdr = np.empty_like(Y)
            Gx_hdr[:, :-1] = (Y[:, 1:] - Y[:, :-1]) * 0.5
            Gx_hdr[:, -1] = (Y[:, -2] - Y[:, -1]) * 0.5
        else:
            e = np.minimum(np.arange(w) + 1, w - 1)
            Gx_hdr = (Y[:, e] - Y)
        utils.save_scanline(Gx_hdr, scanline_row, "2_original_HDR_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)
    utils.print_elapsed("     [tmo] 로그 공간 변환 완료")

    if scanline_row is not None:
        utils.save_scanline(H, scanline_row, "3_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 가우시안 피라미드 구성 
    mins = min(w, h)
    n_pyramid_levels = 0
    temp_mins = mins
    while temp_mins >= TOP_SIZE:
        n_pyramid_levels += 1
        temp_mins //= 2
    if n_pyramid_levels == 0: n_pyramid_levels = 1

    pyramids = createGaussianPyramids(H, n_pyramid_levels)
    utils.print_elapsed("     [tmo] 가우시안 피라미드 구성 완료")

    # value 행렬 병렬 계산
    scaling_factor = [None] * n_pyramid_levels
    with ThreadPoolExecutor(max_workers=n_pyramid_levels) as executor:
        futures = []
        for k in range(n_pyramid_levels):
            if k >= detail_level or k == n_pyramid_levels - 1 or not newfattal:
                futures.append((k, executor.submit(calculate_level_scaling_factor, pyramids[k], k, alfa, beta, noise, xp_ratio, y0)))
            else:
                scaling_factor[k] = None
                
        for k, future in futures:
            scaling_factor[k] = future.result()

    # FI 행렬 계산
    attenuation_map = calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal)
    utils.print_elapsed("     [tmo] FI 행렬 및 그래디언트 계산 완료")

    if scanline_row is not None:
        if fftsolver:
            Gx_un = np.empty_like(H)
            Gx_un[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5
            Gx_un[:, -1] = (H[:, -2] - H[:, -1]) * 0.5
            Gy_un = np.empty_like(H)
            Gy_un[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5
            Gy_un[-1, :] = (H[-2, :] - H[-1, :]) * 0.5
            G_un = np.sqrt(Gx_un**2 + Gy_un**2)
        else:
            e = np.minimum(np.arange(w) + 1, w - 1)
            s = np.minimum(np.arange(h) + 1, h - 1)
            Gx_un = (H[:, e] - H)
            Gy_un = (H[s, :] - H)
            G_un = np.sqrt(Gx_un**2 + Gy_un**2)
        utils.save_scanline(G_un, scanline_row, "4_original_log_domain_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
        utils.save_scanline(Gx_un, scanline_row, "4_original_log_domain_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 기울기 감쇠
    if fftsolver:
        Gx = np.empty_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (attenuation_map[:, 1:] + attenuation_map[:, :-1])
        Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (attenuation_map[:, -2] + attenuation_map[:, -1])
        
        Gy = np.empty_like(H)
        Gy[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5 * (attenuation_map[1:, :] + attenuation_map[:-1, :])
        Gy[-1, :] = (H[-2, :] - H[-1, :]) * 0.5 * (attenuation_map[-2, :] + attenuation_map[-1, :])
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * attenuation_map
        Gy = (H[s, :] - H) * attenuation_map

    if scanline_row is not None:
        G_att = np.sqrt(Gx**2 + Gy**2)
        utils.save_scanline(G_att, scanline_row, "5_attenuated_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
        utils.save_scanline(Gx, scanline_row, "5_attenuated_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]
    utils.print_elapsed("     [tmo] 다이버전스(발산) 계산 완료")

    # PDE 풀이
    utils.print_elapsed("     [tmo] PDE 풀이 시작")
    if fftsolver:
        U = pde_fft.solve_pde_fft(DivG, hpf_sigma=hpf_sigma)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo] PDE 풀이 완료")

    if scanline_row is not None:
        utils.save_scanline(U, scanline_row, "6_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 지수 공간으로 복원
    gamma = 1.0
    L = np.exp(gamma * U)

    # 백분위수 기반 정규화
    cut_min = 0.01 * 0.1
    cut_max = 1.0 - 0.01 * 0.5
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_min = executor.submit(np.percentile, L, cut_min * 100)
        future_max = executor.submit(np.percentile, L, cut_max * 100)
        min_val = future_min.result()
        max_val = future_max.result()

    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1)

    if scanline_row is not None:
        utils.save_scanline(L, scanline_row, "7_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir)

    utils.print_elapsed("     [tmo] 히스토그램 평활화 및 정규화 완료")
    
    return L


def pfstmo_fattal02(img, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007, pre_hpf_sigma=0.007, xp_ratio=0.05, y0=6.0):
    utils.print_elapsed("   [pfstmo] 시작 (RGB to Y 변환)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    if scanline_row is not None:
        utils.save_scanline(img, scanline_row, "0_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 1. Original 이미지에 High-Pass Filter 먼저 적용
    img_filtered = apply_high_pass_filter(img, pre_hpf_sigma=pre_hpf_sigma)

    L = tmo_fattal02(
        img_filtered, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level,
        scanline_row=scanline_row, highlight_ranges=highlight_ranges, save_dir=save_dir,
        hpf_sigma=hpf_sigma, xp_ratio=xp_ratio, y0=y0
    )
    utils.print_elapsed("   [pfstmo] tmo_fattal02 연산 완료")

    return L
