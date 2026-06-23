import numpy as np
import scipy.fft as fft
import cv2
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Add 'src' directory to sys.path to ensure correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils import utils
from fattal import pde_multigrid
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

#same
def downSample(A):
    h, w = A.shape
    nh, nw = h // 2, w // 2
    B = (A[0:2*nh:2, 0:2*nw:2] + A[1:2*nh:2, 0:2*nw:2] + 
         A[0:2*nh:2, 1:2*nw:2] + A[1:2*nh:2, 1:2*nw:2]) * 0.25
    return B

#new!
def createGaussianPyramids(H, nlevels):
    """C++의 createGaussianPyramids를 정확히 재현"""
    pyramids = [H]
    L = gaussianBlur(H)  # 먼저 블러

    for k in range(1, nlevels):
        down = downSample(L)          # 블러된 이미지를 다운샘플
        pyramids.append(down)
        if k < nlevels - 1:
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

def calculate_attenuation(gradient, alfa, beta, noise, k, target_k=None, tmp=1.0):
    avgGrad = np.mean(gradient)
    grad_safe = np.maximum(gradient, 1e-4)
    a = alfa * avgGrad
    if target_k is not None and k == target_k:
        # 지정된 하나의 층(target_k)에 대해서만 지수부에 tmp를 반영
        attenuation = ((grad_safe + noise) / a) ** ((beta - 1.0) * tmp)
    else:
        # 나머지 층은 기존 수식을 그대로 유지
        attenuation = ((grad_safe + noise) / a) ** (beta - 1.0)
    return attenuation

def calculate_level_attenuation(H, k, alfa, beta, noise, target_k=None, tmp=1.0):
    G = calculate_gradient_mag(H, k)
    attenuation = calculate_attenuation(G, alfa, beta, noise, k, target_k, tmp)
    return attenuation

def calculateFiMatrix(values, pyramids, nlevels, newfattal):
    """
    병렬 처리로 사전에 계산된 values 배열을 받아
    순차적 의존성이 있는 FI 행렬 생성 및 업샘플링을 수행합니다.
    """
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


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, HE_weight, scanline_row=None, highlight_ranges=None, save_dir=None, target_k=None, tmp=1.0):
    utils.print_elapsed("     [tmo] Start")
    h, w = Y.shape
    detail_level = np.clip(detail_level, 0, 3)
    
    MSIZE = 8 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
         
    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)
    
    if scanline_row is not None:
        utils.save_scanline(H, scanline_row, "2_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo] Log-space transform completed")

    # 가우시안 피라미드 구성 
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0: nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    utils.print_elapsed("     [tmo] Gaussian pyramids created")

    # value 행렬 병렬 계산
    attenuation = [None] * nlevels
    with ThreadPoolExecutor(max_workers=nlevels) as executor:
        futures = []
        for k in range(nlevels):
            if k >= detail_level or k == nlevels - 1 or not newfattal:
                futures.append((k, executor.submit(calculate_level_attenuation, pyramids[k], k, alfa, beta, noise, target_k, tmp)))
            else:
                attenuation[k] = None
                
        for k, future in futures:
            attenuation[k] = future.result()

    # FI 행렬 계산
    FI = calculateFiMatrix(attenuation, pyramids, nlevels, newfattal)
    utils.print_elapsed("     [tmo] FI matrix & gradient calculated")

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
        # Gx 계산 (가로 방향)
        Gx = np.empty_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (FI[:, 1:] + FI[:, :-1])
        Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (FI[:, -2] + FI[:, -1])
        
        # Gy 계산 (세로 방향)
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
    DivG[1:, :] -= Gy[:-1, :] # 0 padding 후 후방차분

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]
    utils.print_elapsed("     [tmo] Divergence calculated")

    # PDE 풀이
    U = np.zeros_like(DivG)
    utils.print_elapsed("     [tmo] PDE solving started")
    if fftsolver:
        U = pde_fft.solve_pde_fft(DivG, hpf_sigma = 0.007) #0.007
    else:
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo] PDE solved")

    if scanline_row is not None:
        utils.save_scanline(U, scanline_row, "5_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)

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

    # exact_continuous_he 내부 혹은 호출부 수정 제안
    if HE_weight > 0.0:
        L = utils.exact_continuous_he(L, HE_weight)
        
    if scanline_row is not None:
        utils.save_scanline(L, scanline_row, "6_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo] HE & Normalization completed")
    
    return L

def pfstmo_fattal02(R, G, B, opt_alpha, opt_beta, opt_saturation, opt_noise, newfattal, fftsolver, detail_level, HE_weight, scanline_row=None, highlight_ranges=None, save_dir=None, target_k=None, tmp=1.0):
    utils.print_elapsed("   [pfstmo] Start (RGB to Y)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # RGB to Y 변환 (Rec. 709 휘도 계수 사용)
    Yr = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    L = tmo_fattal02(Yr, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, HE_weight, scanline_row, highlight_ranges, save_dir=save_dir, target_k=target_k, tmp=tmp)
    utils.print_elapsed("   [pfstmo] tmo_fattal02 completed")

    epsilon = 1e-4
    Y_safe = np.maximum(Yr, epsilon)
    L_safe = np.maximum(L, epsilon)

    # Gray scale image and opt_saturation = 1
    Gray_out = np.maximum(R / Y_safe,0) * L_safe
    utils.print_elapsed("   [pfstmo] Saturation restored and return")

    return Gray_out, Gray_out, Gray_out
