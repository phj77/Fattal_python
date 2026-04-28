import numpy as np
import scipy.fft as fft
import pde_multigrid

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

# why noise added?
def calculateFiMatrix(gradients, avgGrads, nlevels, detail_level, alfa, beta, noise, newfattal):
    h, w = gradients[-1].shape
    fi = [None] * nlevels
    fi[-1] = np.ones((h, w), dtype=np.float32) if newfattal else np.zeros((h, w), dtype=np.float32)

    for k in range(nlevels - 1, -1, -1):
        G = gradients[k]
        a = alfa * avgGrads[k]
        
        G_safe = np.maximum(G, 1e-4)
        value = ((G_safe + noise) / a) ** (beta - 1.0)

        if k >= detail_level or k == nlevels - 1 or not newfattal:
            if newfattal:
                fi[k] = fi[k] * value
            else:
                fi[k] = value

        if k > 0:
            target_shape = gradients[k-1].shape
            if newfattal:
                up = upSample(fi[k], target_shape)
                fi[k-1] = gaussianBlur(up)
            else:
                fi[k-1] = np.zeros(target_shape, dtype=np.float32)

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

    # 기울기 감쇠 # 다름
    # if fftsolver:
    #     yp1 = np.minimum(np.arange(h) + 1, h - 2)
    #     xp1 = np.minimum(np.arange(w) + 1, w - 2)
    #     Gx = (H[:, xp1] - H) * 0.5 * (FI[:, xp1] + FI)
    #     Gy = (H[yp1, :] - H) * 0.5 * (FI[yp1, :] + FI)
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

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]

    # PDE 풀이
    #U = solve_pde_fft(DivG)

    # PDE 풀이
    if fftsolver:
        U = solve_pde_fft(DivG)
    else:
        U = pde_multigrid.solve_pde_multigrid(DivG)

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