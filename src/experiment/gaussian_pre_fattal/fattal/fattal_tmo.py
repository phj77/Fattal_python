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


def apply_gaussian_blur(img: np.ndarray, pre_gaussian_sigma: float = 0.007) -> np.ndarray:
    """
    Original 이미지에 Gaussian Blur를 먼저 적용합니다.
    
    Args:
        img (np.ndarray): 입력 2D 이미지 (단일 채널)
        pre_gaussian_sigma (float): Gaussian Blur 표준편차/강도 (0 이하일 경우 필터 미적용)
        
    Returns:
        np.ndarray: Gaussian Blur가 적용된 이미지
    """
    if pre_gaussian_sigma <= 0.0:
        return img.copy()

    utils.print_elapsed(f"   [Pre-Gaussian] Gaussian Blur 적용 시작 (sigma_ratio={pre_gaussian_sigma})")
    h, w = img.shape
    sigma_spatial = pre_gaussian_sigma * max(h, w)
    if sigma_spatial < 0.5:
        sigma_spatial = 0.5

    ksize = int(np.ceil(sigma_spatial * 3) * 2 + 1)
    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma_spatial, sigmaY=sigma_spatial)

    utils.print_elapsed("   [Pre-Gaussian] Gaussian Blur 적용 완료")
    return blurred


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
    grad_safe = np.maximum(gradient, 1e-4)
    a = alfa * avgGrad
    scaling_factor = ((grad_safe + noise) / a) ** (beta - 1.0)
    return scaling_factor


def calculate_level_scaling_factor(H, k, alfa, beta, noise):
    G = calculate_gradient_mag(H, k)
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


def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level, hpf_sigma=0.007):
    utils.print_elapsed("     [tmo] 시작")
    h, w = Y.shape
    
    TOP_SIZE = 2**3 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)
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
                futures.append((k, executor.submit(calculate_level_scaling_factor, pyramids[k], k, alfa, beta, noise)))
            else:
                scaling_factor[k] = None
                
        for k, future in futures:
            scaling_factor[k] = future.result()

    # FI 행렬 계산
    attenuation_map = calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal)
    utils.print_elapsed("     [tmo] FI 행렬 및 그래디언트 계산 완료")

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

    utils.print_elapsed("     [tmo] 히스토그램 평활화 및 정규화 완료")
    
    return L


def pfstmo_fattal02(img, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, hpf_sigma=0.007, pre_gaussian_sigma=0.007):
    utils.print_elapsed("   [pfstmo] 시작")
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # 1. Original 이미지에 Gaussian Blur 먼저 적용
    img_filtered = apply_gaussian_blur(img, pre_gaussian_sigma=pre_gaussian_sigma)

    # 2. Fattal TMO 알고리즘 실행
    L = tmo_fattal02(img_filtered, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level, hpf_sigma=hpf_sigma)
    utils.print_elapsed("   [pfstmo] tmo_fattal02 연산 완료")

    return L
