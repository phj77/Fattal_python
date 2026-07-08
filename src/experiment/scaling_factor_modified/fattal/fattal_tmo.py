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


#same
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

#same; Gaussian pyramid 만들때 avg pooling, blur를 둘 다 사용할 필요 있나?
def downSample(A):
    # Original in LuminanceHDR
    h, w = A.shape
    nh, nw = h // 2, w // 2
    B = (A[0:2*nh:2, 0:2*nw:2] + A[1:2*nh:2, 0:2*nw:2] + 
         A[0:2*nh:2, 1:2*nw:2] + A[1:2*nh:2, 1:2*nw:2]) * 0.25

    # downsample by sampling
    # B = A[::2, ::2]
    return B

#new!
def createGaussianPyramids(H, n_pyramid_levels):
    """C++의 createGaussianPyramids를 정확히 재현"""
    pyramids = [H]
    L = gaussianBlur(H)  # 먼저 블러

    for k in range(1, n_pyramid_levels):
        down = downSample(L)          # 블러된 이미지를 다운샘플
        pyramids.append(down)
        if k < n_pyramid_levels - 1:
            L = gaussianBlur(down)    # 다음 레벨을 위해 블러
        # 마지막 레벨은 추가 blur 불필요 (다음 단계 없음)

    return pyramids

#same
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

def calculate_scaling_factor(gradient,alfa,beta,noise):
    avgGrad = np.mean(gradient)
    a = alfa * avgGrad
    # [바꾸기 전 버전]
    # grad_safe = np.maximum(gradient, 1e-4)
    # scaling_factor = ((grad_safe + noise) / a) ** (beta - 1.0)

    mask = gradient > 0
    safe_grad = np.where(mask, gradient, 1.0)  # 0인 곳은 더미값 1.0 (0^음수 power 연산 오류 방지)
    scaling_factor = np.where(mask, ((safe_grad + noise) / a) ** (beta - 1.0), 0.0)
    return scaling_factor

#new=============================================
# def calculate_piecewise_scaling_factor(gradient, alfa, beta, v0=0.01, xp_ratio=0.1, vp=9):
#     """
#     조각 정의 기반 스케일링 팩터 계산 함수.
    
#     Args:
#         gradient (np.ndarray): 그래디언트 크기 배열
#         alfa (float): 평균 그래디언트 대비 기준점 비율
#         beta (float): 원본 스케일링 압축 파라미터 
#         v0 (float): 그래디언트가 0일 때의 스케일링 값 (노이즈 억제력 제어)
#         xp_ratio (float): 기준점(a) 대비 피크 위치의 비율 (0.0 < xp_ratio < 1.0)
#         vp (float): 피크에서의 최대 스케일링 팩터 값
#     """ 
#     avgGrad = np.mean(gradient)
#     a = alfa * avgGrad
    
#     # 피크 위치 도출
#     xp = xp_ratio * a
    
#     mask = gradient > 0
#     # 0 이하 방지용 더미값 할당 (런타임 경고 방지)
#     x = np.where(mask, gradient, 1.0)
#     scaling_factor = np.zeros_like(x)
    
#     # 구간 1: x >= a
#     mask1 = x >= a
#     scaling_factor[mask1] = (x[mask1] / a) ** (beta - 1.0)
    
#     # 구간 2: 0 < x < xp
#     mask2 = (x < xp) & mask
#     scaling_factor[mask2] = vp - ((vp - v0) / (xp**2)) * ((x[mask2] - xp)**2)
    
#     # 구간 3: xp <= x < a
#     mask3 = (x >= xp) & (x < a) & mask
#     x3 = x[mask3]
#     t = (x3 - xp) / (a - xp)
#     m = (beta - 1.0) * (a - xp) / a
#     scaling_factor[mask3] = (2*vp - 2 + m) * (t**3) + (-3*vp + 3 - m) * (t**2) + vp
    
#     return np.where(mask, scaling_factor, 0.0)

import numpy as np

def calculate_piecewise_scaling_factor(gradient, alfa, beta, v0=0.01, xp_ratio=0.08, vp=6.0, w=1):
    """
    단일 피크 조건과 폭 조절(w) 기능이 포함된 스케일링 팩터 계산 함수.
    
    Args:
        gradient (np.ndarray): 그래디언트 크기 배열
        alfa (float): 평균 그래디언트 대비 기준점 비율
        beta (float): 원본 스케일링 압축 파라미터 
        v0 (float): 그래디언트가 0일 때의 스케일링 값 (노이즈 억제)
        xp_ratio (float): 기준점(a) 대비 피크 위치의 비율 (0.0 < xp_ratio < 1.0)
        vp (float): 피크에서의 최대 스케일링 팩터 값
        w (float): 좌측 피크 폭 조절 파라미터 (w > 1.0, 기본값 2.0)
    """
    avgGrad = np.mean(gradient)
    a = alfa * avgGrad
    xp = xp_ratio * a
    
    mask = gradient > 0
    # 0 이하 방지용 더미값 할당 (런타임 경고 방지)
    x = np.where(mask, gradient, 1.0)
    scaling_factor = np.zeros_like(x)
    
    # 구간 1: x >= a
    mask1 = x >= a
    scaling_factor[mask1] = (x[mask1] / a) ** (beta - 1.0)
    
    # 구간 2: 0 < x < xp (폭 조절 파라미터 w 적용)
    mask2 = (x < xp) & mask
    # 1.0 - x/xp는 양수이므로 안전하게 w 거듭제곱 연산 가능
    scaling_factor[mask2] = vp - (vp - v0) * ((1.0 - x[mask2] / xp) ** w)
    
    # 구간 3: xp <= x < a
    mask3 = (x >= xp) & (x < a) & mask
    x3 = x[mask3]
    t = (x3 - xp) / (a - xp)
    m = (beta - 1.0) * (a - xp) / a
    scaling_factor[mask3] = (2*vp - 2 + m) * (t**3) + (-3*vp + 3 - m) * (t**2) + vp
    
    return np.where(mask, scaling_factor, 0.0)
#new=============================================

def calculate_level_scaling_factor(H, k, alfa, beta, noise):
    G = calculate_gradient_mag(H, k)
    if k == 3:
        scaling_factor = calculate_piecewise_scaling_factor(G, alfa, beta)
    else:
        scaling_factor = calculate_scaling_factor(G, alfa, beta, noise)

    # scaling_factor = calculate_scaling_factor(G, alfa, beta, noise)

    #scaling_factor = calculate_piecewise_scaling_factor(G, alfa, beta)

    return scaling_factor

def calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal):
    """
    병렬 처리로 사전에 계산된 scaling_factor 배열을 받아
    순차적 의존성이 있는 attenuation 행렬 생성 및 업샘플링을 수행합니다.
    """
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


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007):
    utils.print_elapsed("     [tmo] 시작")
    h, w = Y.shape
    #detail_level = np.clip(detail_level, 0, 3) #detail level 이상의 피라미드 층만 감쇠 함수를 연산함.
    
    #TOP_SIZE = 2**8 if fftsolver else 32
    TOP_SIZE = 2**5 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_after_high_pass_filter_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
        # 원본 original hdr 이미지의 x방향 gradient 계산 및 저장
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
                futures.append((k, executor.submit(calculate_level_scaling_factor, pyramids[k], k, alfa, beta, noise)))
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
        # Gx 계산 (가로 방향)
        Gx = np.empty_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (attenuation_map[:, 1:] + attenuation_map[:, :-1])
        Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (attenuation_map[:, -2] + attenuation_map[:, -1])
        
        # Gy 계산 (세로 방향)
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
    DivG[1:, :] -= Gy[:-1, :] # 0 padding 후 후방차분

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

    # 백분위수 기반 정규화 (0.1% ~ 99.5%) - ThreadPoolExecutor 병렬 연산
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

def pfstmo_fattal02(img, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007, pre_hpf_sigma=0.007):
    utils.print_elapsed("   [pfstmo] 시작 (RGB to Y 변환)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # 0. Original HDR map (pre highpass filter 적용 전) scanline 저장
    if scanline_row is not None:
        utils.save_scanline(img, scanline_row, "0_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 1. Original 이미지에 High-Pass Filter 먼저 적용
    img_filtered = apply_high_pass_filter(img, pre_hpf_sigma=pre_hpf_sigma)

    L = tmo_fattal02(
        img_filtered, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level,
        scanline_row=scanline_row, highlight_ranges=highlight_ranges, save_dir=save_dir,
        hpf_sigma=hpf_sigma
    )
    utils.print_elapsed("   [pfstmo] tmo_fattal02 연산 완료")

    return L
