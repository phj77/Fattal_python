import numpy as np
import scipy.fft as fft
import pde_multigrid
import cv2
import sys

import utils.utils

def transform_ev2normal(A):
    h, w = A.shape
    A_copy = A.copy()
    
    A_copy[1:h-1, 1:w-1] *= 0.25
    A_copy[1:h-1, 0] *= 0.5
    A_copy[1:h-1, w-1] *= 0.5
    A_copy[0, 1:w-1] *= 0.5
    A_copy[h-1, 1:w-1] *= 0.5

    # FFTW_REDFT00은 Type 1 DCT에 해당합니다.
    T = fft.dctn(A_copy, type=1, norm=None)
    return T

def transform_normal2ev(A):
    h, w = A.shape
    T = fft.dctn(A, type=1, norm=None)
    
    T *= (1.0 / ((h - 1) * (w - 1)))
    T[0, :] *= 0.5
    T[h-1, :] *= 0.5
    T[:, 0] *= 0.5
    T[:, w-1] *= 0.5
    return T

def get_lambda(n):
    i = np.arange(n)
    return -4.0 * np.sin(i / (2.0 * (n - 1)) * np.pi)**2

def make_compatible_boundary(F):
    h, w = F.shape
    sum_val = np.sum(F[1:h-1, 1:w-1])
    sum_val += 0.5 * (np.sum(F[1:h-1, 0]) + np.sum(F[1:h-1, w-1]))
    sum_val += 0.5 * (np.sum(F[0, 1:w-1]) + np.sum(F[h-1, 1:w-1]))
    sum_val += 0.25 * (F[0, 0] + F[0, w-1] + F[h-1, 0] + F[h-1, w-1])

    add = -sum_val / (h + w - 3)
    
    F_copy = F.copy()
    F_copy[0, :] += add
    F_copy[h-1, :] += add
    F_copy[1:h-1, 0] += add
    F_copy[1:h-1, w-1] += add
    return F_copy

def solve_pde_fft(F):
    h, w = F.shape
    F_compat = make_compatible_boundary(F)
    F_tr = transform_normal2ev(F_compat)

    l1 = get_lambda(h).reshape(-1, 1)
    l2 = get_lambda(w).reshape(1, -1)
    
    denom = l1 + l2
    denom[0, 0] = 1.0  # 0으로 나누기 방지
    F_tr = F_tr / denom
    F_tr[0, 0] = 0.0
    
    U = transform_ev2normal(F_tr)
    U -= np.max(U)
    return U

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

#same
def calculateGradients(H, k):
    h, w = H.shape
    divider = 2.0 ** (k + 1)
    
    w_idx = np.maximum(np.arange(w) - 1, 0)
    e_idx = np.minimum(np.arange(w) + 1, w - 1)
    n_idx = np.maximum(np.arange(h) - 1, 0)
    s_idx = np.minimum(np.arange(h) + 1, h - 1)

    gx = (H[:, w_idx] - H[:, e_idx]) / divider
    gy = (H[s_idx, :] - H[n_idx, :]) / divider

    G = np.sqrt(gx**2 + gy**2)
    avgGrad = np.mean(G)
    return G, avgGrad

def calculateFiMatrix(gradients, avgGrads, nlevels, detail_level, alfa, beta, noise, newfattal):
    """
    C++의 분기 구조를 유지하면서 Numpy의 벡터화 연산을 통해 성능을 최적화한 구현체입니다.
    외부에서 할당된 배열을 수정하는 대신, 계산이 완료된 최상위 배열을 반환합니다.
    """
    h, w = gradients[-1].shape
    fi = [None] * nlevels

    # C++: fi[nlevels - 1] = new pfs::Array2Df(width, height);
    # C++: if (newfattal) { ... = 1.0f; }
    # 효율성 최적화: 조건에 따라 메모리 할당과 초기화를 분리합니다.
    if newfattal:
        fi[-1] = np.ones((h, w), dtype=np.float32)
    else:
        # newfattal == false인 경우 다음 루프에서 전체가 덮어씌워지므로
        # np.zeros가 아닌 np.empty를 사용하여 할당 오버헤드를 제거합니다.
        fi[-1] = np.empty((h, w), dtype=np.float32) 

    # C++: for (int k = nlevels - 1; k >= 0; k--)
    for k in range(nlevels - 1, -1, -1):
        
        # C++: if (k >= detail_level || k == nlevels - 1 || newfattal == false)
        if k >= detail_level or k == nlevels - 1 or not newfattal:
            
            # C++의 이중 for 루프 내 픽셀 단위 연산을 Numpy 벡터 연산으로 완전히 대체합니다.
            grad = gradients[k] 
            
            # C++: float grad = ((*gradients[k])(x, y) < 1e-4f) ? 1e-4 : (*gradients[k])(x, y);
            # 효율성 최적화: 조건부 생성(np.where) 대신 고도로 최적화된 내부 함수(np.maximum)를 사용합니다.
            grad_safe = np.maximum(grad, 1e-4)
            
            # C++: float value = powf((grad + noise) / a, beta - 1.0f);
            a = alfa * avgGrads[k]
            value = ((grad_safe + noise) / a) ** (beta - 1.0)
            
            # C++: if (newfattal) (*fi[k])(x, y) *= value;
            # C++: else           (*fi[k])(x, y) = value;
            if newfattal:
                fi[k] *= value  # 메모리 복사를 피하기 위한 In-place 곱셈 연산
            else:
                fi[k] = value   # 새로운 배열 참조 할당
                
        # C++: if (k > 1) { fi[k - 1] = new ... } else { fi[0] = &FI; }
        # C++: if (k > 0 && newfattal) { upSample... gaussianBlur... }
        # 파이썬의 특성에 맞게 두 분기를 병합하여 레벨 전이(Level Transition) 과정을 단순화합니다.
        if k > 0:
            target_shape = gradients[k-1].shape
            
            if newfattal:
                # 하위 레벨로의 업샘플링 및 블러 처리
                up = upSample(fi[k], target_shape)
                fi[k-1] = gaussianBlur(up)
            else:
                # newfattal이 거짓일 경우, 다음 루프에서 fi[k] = value 구문을 통해 
                # 배열 전체가 새롭게 덮어씌워지게 됩니다.
                # 따라서 np.zeros 대신 할당 속도가 가장 빠른 np.empty를 사용합니다.
                fi[k-1] = np.empty(target_shape, dtype=np.float32)

    # C++에서는 FI 변수의 포인터를 교체하여 결과를 반환하지만,
    # 파이썬에서는 로컬에서 완성된 최상위 피라미드 계층을 직접 반환하는 것이 가장 안전하고 빠릅니다.
    return fi[0]


# fixed! -> DivG part
def tmo_fattal02(Y, alfa, beta, noise, newfattal, fftsolver, detail_level):
    h, w = Y.shape
    detail_level = np.clip(detail_level, 0, 3)
    
    MSIZE = 8 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)

    # 가우시안 피라미드 구성 
    ###===========different========================
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0: nlevels = 1

    # pyramids = [H]
    # for k in range(1, nlevels):
    #     down = downSample(pyramids[-1])
    #     if k < nlevels - 1:
    #         down = gaussianBlur(down)
    #     pyramids.append(down)

    pyramids = createGaussianPyramids(H, nlevels)
    ###============================================

    gradients = []
    avgGrads = []
    for k in range(nlevels):
        G, avg = calculateGradients(pyramids[k], k)
        gradients.append(G)
        avgGrads.append(avg)

    # FI 행렬 계산
    FI = calculateFiMatrix(gradients, avgGrads, nlevels, detail_level, alfa, beta, noise, newfattal)

    # 기울기 감쇠
    if fftsolver:
        # y축 경계 인덱스 처리
        yp1 = np.arange(1, h + 1)
        yp1[-1] = h - 2
        
        # x축 경계 인덱스 처리
        xp1 = np.arange(1, w + 1)
        xp1[-1] = w - 2
        
        Gx = (H[:, xp1] - H) * 0.5 * (FI[:, xp1] + FI)
        Gy = (H[yp1, :] - H) * 0.5 * (FI[yp1, :] + FI)
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * FI
        Gy = (H[s, :] - H) * FI

    # gradient magnitude map의 hisgoram visualization======================================
    G_map = cv2.magnitude(Gx, Gy)

    G_cut_min = 0.01 * 0.01
    G_cut_max = 1.0 - 0.005
    G_min_val = np.percentile(G_map, G_cut_min * 100)
    G_max_val = np.percentile(G_map, G_cut_max * 100)

    G_map = np.maximum(G_map, G_min_val)
    G_map = np.minimum(G_map, G_max_val)

    utils.utils.plot_float_array_histogram(G_map)
    sys.exit()
    #======================================================================================

    
    # # show gradient map==================================================================
    # #show
    # G_map = cv2.magnitude(Gx, Gy)

    # #============기울기 정규화
    # G_map = G_map/H    
    # #============

    # G_cut_min = 0.01 * 0.01
    # G_cut_max = 1.0 - 0.005
    # G_min_val = np.percentile(G_map, G_cut_min * 100)
    # G_max_val = np.percentile(G_map, G_cut_max * 100)

    # G_map = np.maximum(G_map, G_min_val)
    # G_map = np.minimum(G_map, G_max_val)

    # G_map_normalized = cv2.normalize(
    #     G_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    # )


    # cv2.imshow('gradient magnitude map', G_map_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()    

    # # #save
    # cv2.imwrite(f'./images/beta_images/Newfattal_Multigrid_alpha_0.3/A/beta_0.60_grad.png', G_map_normalized)
    # sys.exit() 
    #=======================================================================================

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :] # 0 padding 후 후방차분

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]


    # PDE 풀이
    if fftsolver:
        U = solve_pde_fft(DivG)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)

    # 지수 공간으로 복원
    gamma = 1.0
    L = np.exp(gamma * U)

    # 백분위수 기반 정규화 (0.1% ~ 99.5%)
    cut_min = 0.01 * 0.1
    cut_max = 1.0 - 0.01 * 0.5
    min_val = np.percentile(L, cut_min * 100)
    max_val = np.percentile(L, cut_max * 100)

    L = (L - min_val) / (max_val - min_val)
    #L = np.maximum(L, 0.0)
    L = np.clip(L, 0, 1)

    #histogram equalizaiton before quantaization
    #L = utils.utils.exact_continuous_he(L)
    
    return L

def pfstmo_fattal02(R, G, B, opt_alpha, opt_beta, opt_saturation, opt_noise, newfattal, fftsolver, detail_level):
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # RGB to Y 변환 (Rec. 709 휘도 계수 사용)
    Yr = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    L = tmo_fattal02(Yr, opt_alpha, opt_beta, opt_noise, newfattal, fftsolver, detail_level)

    epsilon = 1e-4
    Y_safe = np.maximum(Yr, epsilon)
    L_safe = np.maximum(L, epsilon)

    # RGB 채널 재결합 (채도 복원)
    R_out = np.power(np.maximum(R / Y_safe, 0.0), opt_saturation) * L_safe
    G_out = np.power(np.maximum(G / Y_safe, 0.0), opt_saturation) * L_safe
    B_out = np.power(np.maximum(B / Y_safe, 0.0), opt_saturation) * L_safe

    return R_out, G_out, B_out








#===================for attach two gradient maps==========================================================
#=========================================================================================================
#==========================================================================================================
def get_ideal_log_grad(Y, alfa, beta, noise, newfattal, fftsolver, detail_level):
    h, w = Y.shape
    detail_level = np.clip(detail_level, 0, 3)
    
    MSIZE = 8 if fftsolver else 32

    minLum = np.min(Y)
    maxLum = np.max(Y)

    # 로그 공간 변환
    H = np.log(100.0 * Y / maxLum + 1e-4)

    # 가우시안 피라미드 구성 
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0: nlevels = 1
    pyramids = createGaussianPyramids(H, nlevels)

    gradients = []
    avgGrads = []
    for k in range(nlevels):
        G, avg = calculateGradients(pyramids[k], k)
        gradients.append(G)
        avgGrads.append(avg)

    # FI 행렬 계산
    FI = calculateFiMatrix(gradients, avgGrads, nlevels, detail_level, alfa, beta, noise, newfattal)

    # 기울기 감쇠
    if fftsolver:
        # y축 경계 인덱스 처리
        yp1 = np.arange(1, h + 1)
        yp1[-1] = h - 2
        
        # x축 경계 인덱스 처리
        xp1 = np.arange(1, w + 1)
        xp1[-1] = w - 2
        
        Gx = (H[:, xp1] - H) * 0.5 * (FI[:, xp1] + FI)
        Gy = (H[yp1, :] - H) * 0.5 * (FI[yp1, :] + FI)
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * FI
        Gy = (H[s, :] - H) * FI
    
    # normalizing gradient================================= 기울기 정규화
    #return Gx/H, Gy/H
    # =====================================================
    return Gx, Gy

def solving_pde(fftsolver, Gx, Gy):
    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :] # 0 padding 후 후방차분

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]


    # PDE 풀이
    if fftsolver:
        U = solve_pde_fft(DivG)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)

    # 지수 공간으로 복원
    gamma = 1.0
    L = np.exp(gamma * U)

    # 백분위수 기반 정규화 (0.1% ~ 99.5%)
    cut_min = 0.01 * 0.1
    cut_max = 1.0 - 0.01 * 0.5
    min_val = np.percentile(L, cut_min * 100)
    max_val = np.percentile(L, cut_max * 100)

    L = (L - min_val) / (max_val - min_val)
    L = np.maximum(L, 0.0)
    
    return L

def tmo_fusion_grad(Y, alfa, betas, noise, newfattal, fftsolver, detail_level):
    Gx_1, Gy_1 = get_ideal_log_grad(Y, alfa, betas[0], noise, newfattal, fftsolver, detail_level)
    Gx_2, Gy_2 = get_ideal_log_grad(Y, alfa, betas[1], noise, newfattal, fftsolver, detail_level)

    # gradient map의 scale을 맞춘다.==========================================================
    #gradient map의 유효 범위는 0~k 라고 가정했음. 하방선(0)이 올라가면 추가 조치 필요함.
    #cliping안하고 fusion
    G_1_top_mag = utils.utils.get_top_percentile_threshold(Gx_1, Gy_1, top_percentile=0.5)
    G_2_top_mag = utils.utils.get_top_percentile_threshold(Gx_2, Gy_2, top_percentile=0.5)
    if G_2_top_mag > G_1_top_mag:
        Gx_1, Gy_1 = utils.utils.clip_gradient_intensity(Gx_1, Gy_1, top_percentile=0.5)
        Gx_1 = Gx_1 * (G_2_top_mag / G_1_top_mag)
        Gy_1 = Gy_1 * (G_2_top_mag / G_1_top_mag)
    else:
        Gx_2, Gy_2 = utils.utils.clip_gradient_intensity(Gx_2, Gy_2, top_percentile=0.5)
        Gx_2 = Gx_2 * (G_1_top_mag / G_2_top_mag)
        Gy_2 = Gy_2 * (G_1_top_mag / G_2_top_mag)

    #둘다 cliping해서 fusion
    #========================================================================================

    #직접 조정=================================================================================
    cover_area_x =[0,2500]
    cover_area_y =[1013, 2047]
    # cover_area_x =[0,2746]
    # cover_area_y =[813, 2047]
    #=========================================================================================

    Gx_2[cover_area_y[0]:cover_area_y[1], cover_area_x[0]:cover_area_x[1]] = Gx_1[cover_area_y[0]:cover_area_y[1], cover_area_x[0]:cover_area_x[1]]
    Gy_2[cover_area_y[0]:cover_area_y[1], cover_area_x[0]:cover_area_x[1]] = Gy_1[cover_area_y[0]:cover_area_y[1], cover_area_x[0]:cover_area_x[1]]

    #show gradient map==================================================================
    # G_map = cv2.magnitude(Gx_2, Gy_2)

    # G_cut_min = 0.01 * 0.01
    # G_cut_max = 1.0 - 0.005
    # G_min_val = np.percentile(G_map, G_cut_min * 100)
    # G_max_val = np.percentile(G_map, G_cut_max * 100)

    # G_map = np.maximum(G_map, G_min_val)
    # G_map = np.minimum(G_map, G_max_val)

    # G_map_normalized = cv2.normalize(
    #     G_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    # )

    # cv2.imshow('gradient magnitude map', G_map_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit() 

    # #====================================================================================

    L = solving_pde(fftsolver, Gx_2, Gy_2)

    L = np.clip(L, 0, 1)

    # 히스토그램 시각화
    utils.utils.plot_float_array_histogram(L)
    sys.exit()

    #histogram equalizaiton before quantaization
    L = utils.utils.exact_continuous_he(L,weight=0.08)
    

    return L


def pfstmo_fattal02_fusion(R, G, B, opt_alpha, opt_betas, opt_saturation, opt_noise, newfattal, fftsolver, detail_level):
    if fftsolver:
        newfattal = True

    if opt_noise <= 0.0:
        opt_noise = opt_alpha * 0.01

    # RGB to Y 변환 (Rec. 709 휘도 계수 사용)
    Yr = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    L = tmo_fusion_grad(Yr, opt_alpha, opt_betas, opt_noise, newfattal, fftsolver, detail_level)

    epsilon = 1e-4
    Y_safe = np.maximum(Yr, epsilon)
    L_safe = np.maximum(L, epsilon)

    # RGB 채널 재결합 (채도 복원)
    R_out = np.power(np.maximum(R / Y_safe, 0.0), opt_saturation) * L_safe
    G_out = np.power(np.maximum(G / Y_safe, 0.0), opt_saturation) * L_safe
    B_out = np.power(np.maximum(B / Y_safe, 0.0), opt_saturation) * L_safe

    return R_out, G_out, B_out