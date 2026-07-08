import numpy as np
import scipy.fft as fft
import cv2
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import utils
from fattal import pde_multigrid

# [DCT-II 변경 사항] experiment/DCT2/fattal 내의 pde_fft 모듈을 참조
from experiment.DCT2.fattal import pde_fft


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

def calculate_attenuation(gradient,alfa,beta,noise,k):
    avgGrad = np.mean(gradient)
    grad_safe = np.maximum(gradient, 1e-4)
    a = alfa * avgGrad
    attenuation = ((grad_safe + noise) / a) ** (beta - 1.0)
    return attenuation

def calculate_level_attenuation(H, k, alfa, beta, noise):
    G = calculate_gradient_mag(H, k)
    attenuation = calculate_attenuation(G, alfa, beta, noise,k)
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


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007):
    utils.print_elapsed("     [tmo DCT-II] 시작")
    h, w = Y.shape
    detail_level = np.clip(detail_level, 0, 3)
    
    MSIZE = 8 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    if scanline_row is not None:
        utils.save_scanline(Y, scanline_row, "1_original_HDR_Y", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
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
    
    if scanline_row is not None:
        utils.save_scanline(H, scanline_row, "3_log_space_H", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo DCT-II] 로그 공간 변환 완료")

    # 가우시안 피라미드 구성 
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0: nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    utils.print_elapsed("     [tmo DCT-II] 가우시안 피라미드 구성 완료")

    # value 행렬 병렬 계산
    attenuation = [None] * nlevels
    with ThreadPoolExecutor(max_workers=nlevels) as executor:
        futures = []
        for k in range(nlevels):
            if k >= detail_level or k == nlevels - 1 or not newfattal:
                futures.append((k, executor.submit(calculate_level_attenuation, pyramids[k], k, alfa, beta, noise)))
            else:
                attenuation[k] = None
                
        for k, future in futures:
            attenuation[k] = future.result()

    # FI 행렬 계산
    FI = calculateFiMatrix(attenuation, pyramids, nlevels, newfattal)
    utils.print_elapsed("     [tmo DCT-II] FI 행렬 및 그래디언트 계산 완료")

    if scanline_row is not None:
        if fftsolver:
            # [DCT-II 변경 사항] 반 칸 대칭 노이만 BC: 경계 밖 flux = 0
            # Gx_un[:, -1] = 0, Gy_un[-1, :] = 0 (경계 외부 반사 대칭 → 기울기 = 0)
            Gx_un = np.zeros_like(H)
            Gx_un[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5
            Gy_un = np.zeros_like(H)
            Gy_un[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5
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
        # [DCT-II 변경 사항] 반 칸 대칭 노이만 BC: 경계 외부 반사 패딩 H[N]=H[N-1] 적용 시
        # G_x[N-1] = H[N] - H[N-1] = 0 → 경계 flux는 정확히 0
        # np.zeros_like로 초기화하여 경계(Gx[:,-1], Gy[-1,:])는 0 유지
        Gx = np.zeros_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (FI[:, 1:] + FI[:, :-1])
        # Gx[:, -1] = 0  ← DCT-II 노이만 BC: 오른쪽 경계 flux = 0
        
        Gy = np.zeros_like(H)
        Gy[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5 * (FI[1:, :] + FI[:-1, :])
        # Gy[-1, :] = 0  ← DCT-II 노이만 BC: 아래쪽 경계 flux = 0
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * FI
        Gy = (H[s, :] - H) * FI

    if scanline_row is not None:
        G_att = np.sqrt(Gx**2 + Gy**2)
        utils.save_scanline(G_att, scanline_row, "5_attenuated_gradient_magnitude", highlight_ranges=highlight_ranges, save_dir=save_dir)
        utils.save_scanline(Gx, scanline_row, "5_attenuated_GX", highlight_ranges=highlight_ranges, save_dir=save_dir)

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]
    # [DCT-II 변경 사항] 기존 DCT-I 코드에는 아래 보정이 있었으나 DCT-II에서는 불필요:
    #   DivG[:, 0] += Gx[:, 0]
    #   DivG[0, :] += Gy[0, :]
    # DCT-II에서 왼쪽/위쪽 경계 ghost flux = 0 이므로,
    # DivG[i,0] = Gx[i,0] - 0 = Gx[i,0] 이 이미 위 연산에서 올바르게 계산됨.
    # DCT-I에서는 홀수 대칭 확장으로 ghost flux = -Gx[i,0] 이어서 2배 보정이 필요했음.
    utils.print_elapsed("     [tmo DCT-II] 다이버전스(발산) 계산 완료")

    # PDE 풀이
    utils.print_elapsed("     [tmo DCT-II] PDE 풀이 시작")
    if fftsolver:
        # [DCT-II 변경 사항] 반 칸 경계 중심 대칭 DCT-II 커스텀 PDE 솔버 호출
        U = pde_fft.solve_pde_fft(DivG, hpf_sigma=hpf_sigma)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo DCT-II] PDE 풀이 완료")

    if scanline_row is not None:
        utils.save_scanline(U, scanline_row, "6_before_exponential_U", highlight_ranges=highlight_ranges, save_dir=save_dir)

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

        
    if scanline_row is not None:
        utils.save_scanline(L, scanline_row, "7_final_LDR_L", highlight_ranges=highlight_ranges, save_dir=save_dir)
        
    utils.print_elapsed("     [tmo DCT-II] 히스토그램 평활화 및 정규화 완료")
    
    return L

def pfstmo_fattal02(R, G, B, opt_alpha, opt_beta, opt_saturation, opt_noise, newfattal, fftsolver, detail_level, scanline_row=None, highlight_ranges=None, save_dir=None, hpf_sigma=0.007):
    utils.print_elapsed("   [pfstmo DCT-II] 시작 (RGB to Y 변환)")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # Y (Luminance) 연산
    Y = 0.25 * R + 0.65 * G + 0.1 * B
    L = tmo_fattal02(Y, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, scanline_row, highlight_ranges, save_dir=save_dir, hpf_sigma=hpf_sigma)
    
    # 색상 채도 복원
    s = opt_saturation
    with np.errstate(divide='ignore', invalid='ignore'):
        R_out = np.where(Y > 1e-4, ((R / Y) ** s) * L, L)
        G_out = np.where(Y > 1e-4, ((G / Y) ** s) * L, L)
        B_out = np.where(Y > 1e-4, ((B / Y) ** s) * L, L)

    utils.print_elapsed("   [pfstmo DCT-II] tmo_fattal02 및 채도 복원 연산 완료")
    return R_out, G_out, B_out
