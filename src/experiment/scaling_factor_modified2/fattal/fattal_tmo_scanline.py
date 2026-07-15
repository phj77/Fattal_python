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
from fattal import pde_fft


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

def calculate_scaling_factor(gradient, alfa, beta, noise, opt_y_0):
    avgGrad = np.mean(gradient)
    a = alfa * avgGrad
    
    safe_grad = np.maximum(gradient, 0.0)
    mask_ge = gradient >= a
    mask_less = gradient < a
    
    # a 이상의 경우: 원래 식 적용
    scaling_factor_ge = np.where(mask_ge, ((safe_grad + noise) / a) ** (beta - 1.0), 0.0)
    
    # a 미만의 경우: 새로운 함수 적용
    exponent = (1.0 - beta) / (opt_y_0 - 1.0)
    # y_0 - (y_0 - 1) * (x/a)^((1-beta)/(y_0-1))
    scaling_factor_less = np.where(mask_less, opt_y_0 - (opt_y_0 - 1.0) * ((safe_grad / a) ** exponent), 0.0)
    
    scaling_factor = scaling_factor_ge + scaling_factor_less
    return scaling_factor

def calculate_level_scaling_factor(H, k, alfa, beta, noise, opt_y_0):
    G = calculate_gradient_mag(H, k)
    scaling_factor = calculate_scaling_factor(G, alfa, beta, noise, opt_y_0)

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


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, hpf_sigma=0.007, pyramid_top_size=2**3, opt_y_0=2.0, scanline_row=None, highlight_ranges=None, save_dir=None, scanline_col=None):
    utils.print_elapsed("     [tmo] 시작")
    h, w = Y.shape
    #detail_level = np.clip(detail_level, 0, 3) #detail level 이상의 피라미드 층만 감쇠 함수를 연산함.
    
    #TOP_SIZE = 2**8 if fftsolver else 32
    TOP_SIZE = pyramid_top_size if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
        # 원본 original hdr 이미지의 x방향 gradient 계산 및 저장
        if fftsolver:
            Gx_hdr = np.empty_like(Y)
            Gx_hdr[:, :-1] = (Y[:, 1:] - Y[:, :-1]) * 0.5
            Gx_hdr[:, -1] = (Y[:, -2] - Y[:, -1]) * 0.5
        else:
            e = np.minimum(np.arange(w) + 1, w - 1)
            Gx_hdr = (Y[:, e] - Y)
        utils.save_scanline(Gx_hdr, scanline_row, "2_original_HDR_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)
         
    if scanline_col is not None:
        utils.save_vertical_scanline(Y, scanline_col, "1_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
        # 원본 original hdr 이미지의 y방향 gradient 계산 및 저장
        if fftsolver:
            Gy_hdr = np.empty_like(Y)
            Gy_hdr[:-1, :] = (Y[1:, :] - Y[:-1, :]) * 0.5
            Gy_hdr[-1, :] = (Y[-2, :] - Y[-1, :]) * 0.5
        else:
            s = np.minimum(np.arange(h) + 1, h - 1)
            Gy_hdr = (Y[s, :] - Y)
        utils.save_vertical_scanline(Gy_hdr, scanline_col, "2_original_HDR_GY", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)
    
    if scanline_row is not None:
        utils.save_scanline(H, scanline_row, "3_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)

    if scanline_col is not None:
        utils.save_vertical_scanline(H, scanline_col, "3_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)

    utils.print_elapsed("     [tmo] 로그 공간 변환 완료")

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
                futures.append((k, executor.submit(calculate_level_scaling_factor, pyramids[k], k, alfa, beta, noise, opt_y_0)))
            else:
                scaling_factor[k] = None
                
        for k, future in futures:
            scaling_factor[k] = future.result()

    # FI 행렬 계산
    attenuation_map = calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal)
    utils.print_elapsed("     [tmo] FI 행렬 및 그래디언트 계산 완료")

    if scanline_row is not None or scanline_col is not None:
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
            
        if scanline_row is not None:
            utils.save_scanline(G_un, scanline_row, "4_original_log_domain_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
            utils.save_scanline(Gx_un, scanline_row, "4_original_log_domain_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)
            
        if scanline_col is not None:
            utils.save_vertical_scanline(G_un, scanline_col, "4_original_log_domain_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
            utils.save_vertical_scanline(Gy_un, scanline_col, "4_original_log_domain_GY", highlight_ranges=highlight_ranges, save_dir=save_dir)

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

    if scanline_row is not None or scanline_col is not None:
        G_att = np.sqrt(Gx**2 + Gy**2)
        if scanline_row is not None:
            utils.save_scanline(G_att, scanline_row, "5_attenuated_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
            utils.save_scanline(Gx, scanline_row, "5_attenuated_GX", highlight_ranges=highlight_ranges, save_dir=save_dir, ylim=[-0.043, 0.043])
        if scanline_col is not None:
            utils.save_vertical_scanline(G_att, scanline_col, "5_attenuated_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
            utils.save_vertical_scanline(Gy, scanline_col, "5_attenuated_GY", highlight_ranges=highlight_ranges, save_dir=save_dir)

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
        from fattal import pde_multigrid
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo] PDE 풀이 완료")

    if scanline_row is not None:
        utils.save_scanline(U, scanline_row, "6_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)
    if scanline_col is not None:
        utils.save_vertical_scanline(U, scanline_col, "6_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)

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
        utils.save_scanline(L, scanline_row, "7_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir, ylim=[0.0, 0.95])
    if scanline_col is not None:
        utils.save_vertical_scanline(L, scanline_col, "7_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir)

    utils.print_elapsed("     [tmo] 정규화 완료")
    
    return L

def pfstmo_fattal02(img, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, hpf_sigma=0.007, pyramid_top_size=2**3, opt_y_0=2.0, scanline_row=None, highlight_ranges=None, save_dir=None, scanline_col=None):
    utils.print_elapsed("   [pfstmo] 시작 (RGB to Y 변환)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    L = tmo_fattal02(img, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, hpf_sigma=hpf_sigma, pyramid_top_size=pyramid_top_size, opt_y_0=opt_y_0, scanline_row=scanline_row, highlight_ranges=highlight_ranges, save_dir=save_dir, scanline_col=scanline_col)
    utils.print_elapsed("   [pfstmo] tmo_fattal02 연산 완료")

    return L
