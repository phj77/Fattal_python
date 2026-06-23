import numpy as np
import scipy.fft as fft
import cv2
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from fattal import pde_multigrid
from fattal import pde_fft


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

def calculate_attenuation(gradient, alfa, beta, noise, k, tmp_low=1.0, tmp_high=1.5, k_threshold=3):
    avgGrad = np.mean(gradient)
    grad_safe = np.maximum(gradient, 1e-4)
    a = alfa * avgGrad
    tmp = tmp_low if k <= k_threshold else tmp_high
    attenuation = ((grad_safe + noise) / a) ** ((beta - 1.0) * tmp)
    return attenuation

def calculate_level_attenuation(H, k, alfa, beta, noise, tmp_low=1.0, tmp_high=1.5, k_threshold=3):
    G = calculate_gradient_mag(H, k)
    attenuation = calculate_attenuation(G, alfa, beta, noise, k, tmp_low, tmp_high, k_threshold)
    return attenuation

def calculateFiMatrix(values, pyramids, nlevels, newfattal):
    h, w = pyramids[-1].shape
    fi = [None] * nlevels

    if newfattal:
        fi[-1] = np.ones((h, w), dtype=np.float32)
    else:
        fi[-1] = np.empty((h, w), dtype=np.float32) 

    for k in range(nlevels - 1, -1, -1):
        if values[k] is not None:
            if newfattal:
                fi[k] *= values[k]
            else:
                fi[k] = values[k]
                
        if k > 0:
            target_shape = pyramids[k-1].shape
            if newfattal:
                up = upSample(fi[k], target_shape)
                fi[k-1] = gaussianBlur(up)
            else:
                fi[k-1] = np.empty(target_shape, dtype=np.float32)

    return fi[0]


def tmo_fattal02_bilateral(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, HE_weight, 
                            bilateral_d=9, bilateral_sigma_color=0.1, bilateral_sigma_space=5.0,
                            tmp_low=1.0, tmp_high=1.5, k_threshold=3,
                            scanline_row=None, highlight_ranges=None, save_dir=None):
    utils.print_elapsed("     [tmo_bilateral] 시작")
    h, w = Y.shape
    detail_level = np.clip(detail_level, 0, 3)
    
    MSIZE = 8 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
         
    H = np.log(100.0 * Y / maxLum + 1e-4)
    
    if scanline_row is not None:
        utils.save_scanline(H, scanline_row, "2_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo_bilateral] 로그 공간 변환 완료")

    # 가우시안 피라미드 구성 
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0: nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    utils.print_elapsed("     [tmo_bilateral] 가우시안 피라미드 구성 완료")

    # value 행렬 병렬 계산
    attenuation = [None] * nlevels
    with ThreadPoolExecutor(max_workers=nlevels) as executor:
        futures = []
        for k in range(nlevels):
            if k >= detail_level or k == nlevels - 1 or not newfattal:
                futures.append((k, executor.submit(calculate_level_attenuation, pyramids[k], k, alfa, beta, noise, tmp_low, tmp_high, k_threshold)))
            else:
                attenuation[k] = None
                
        for k, future in futures:
            attenuation[k] = future.result()

    # k>=3 레벨들에 대해 양방향 필터(Bilateral Filter) 적용
    for lvl in range(3, len(attenuation)):
        if attenuation[lvl] is not None:
            att_f32 = attenuation[lvl].astype(np.float32)
            attenuation[lvl] = cv2.bilateralFilter(att_f32, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    utils.print_elapsed(f"     [tmo_bilateral] Level >= 3 attenuation map들에 Bilateral Filter 적용 완료 (d={bilateral_d}, sc={bilateral_sigma_color}, ss={bilateral_sigma_space})")

    # FI 행렬 계산
    FI = calculateFiMatrix(attenuation, pyramids, nlevels, newfattal)
    utils.print_elapsed("     [tmo_bilateral] FI 행렬 및 그래디언트 계산 완료")

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
        utils.save_scanline(G_un, scanline_row, "3_original_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
        utils.save_scanline(Gx_un, scanline_row, "3_original_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 기울기 감쇠
    if fftsolver:
        Gx = np.empty_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (FI[:, 1:] + FI[:, :-1])
        Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (FI[:, -2] + FI[:, -1])
        
        Gy = np.empty_like(H)
        Gy[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5 * (FI[1:, :] + FI[:-1, :])
        Gy[-1, :] = (H[-2, :] - H[-1, :]) * 0.5 * (FI[-2, :] + FI[-1, :])
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * FI
        Gy = (H[s, :] - H) * FI

    if scanline_row is not None:
        G_att = np.sqrt(Gx**2 + Gy**2)
        utils.save_scanline(G_att, scanline_row, "4_attenuated_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
        utils.save_scanline(Gx, scanline_row, "4_attenuated_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]
    utils.print_elapsed("     [tmo_bilateral] 다이버전스(발산) 계산 완료")

    # PDE 풀이
    utils.print_elapsed("     [tmo_bilateral] PDE 풀이 시작")
    if fftsolver:
        U = pde_fft.solve_pde_fft(DivG, hpf_sigma=0.007)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo_bilateral] PDE 풀이 완료")

    if scanline_row is not None:
        utils.save_scanline(U, scanline_row, "5_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 지수 공간으로 복원
    gamma = 1.0
    L = np.exp(gamma * U)

    # 백분위수 기반 정규화 (0.1% ~ 99.5%)
    cut_min = 0.01 * 0.1
    cut_max = 1.0 - 0.01 * 0.5
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_min = executor.submit(np.percentile, L, cut_min * 100)
        future_max = executor.submit(np.percentile, L, cut_max * 100)
        min_val = future_min.result()
        max_val = future_max.result()

    L = (L - min_val) / (max_val - min_val)
    L = np.clip(L, 0, 1)

    if HE_weight > 0.0:
        L = utils.exact_continuous_he(L, HE_weight)
        
    if scanline_row is not None:
        utils.save_scanline(L, scanline_row, "6_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo_bilateral] 히스토그램 평활화 및 정규화 완료")
    
    return L

def pfstmo_fattal02_bilateral(R, G, B, opt_alpha, opt_beta, opt_saturation, opt_noise, newfattal, fftsolver, detail_level, HE_weight, 
                              bilateral_d=9, bilateral_sigma_color=0.1, bilateral_sigma_space=5.0,
                              tmp_low=1.0, tmp_high=1.5, k_threshold=3,
                              scanline_row=None, highlight_ranges=None, save_dir=None):
    utils.print_elapsed("   [pfstmo_bilateral] 시작 (RGB to Y 변환)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    Yr = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    L = tmo_fattal02_bilateral(Yr, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, HE_weight, 
                               bilateral_d=bilateral_d, bilateral_sigma_color=bilateral_sigma_color, bilateral_sigma_space=bilateral_sigma_space,
                               tmp_low=tmp_low, tmp_high=tmp_high, k_threshold=k_threshold,
                               scanline_row=scanline_row, highlight_ranges=highlight_ranges, save_dir=save_dir)
    utils.print_elapsed("   [pfstmo_bilateral] tmo_fattal02_bilateral 연산 완료")

    epsilon = 1e-4
    Y_safe = np.maximum(Yr, epsilon)
    L_safe = np.maximum(L, epsilon)

    Gray_out = np.maximum(R / Y_safe, 0) * L_safe
    utils.print_elapsed("   [pfstmo_bilateral] 채도 복원 완료 및 반환")

    return Gray_out, Gray_out, Gray_out
