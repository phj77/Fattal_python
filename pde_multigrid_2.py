import math
import numpy as np

# 다중 레벨 솔버를 위한 전역 설정 값
MODYF = 0
MINS = 16
SMOOTH_IT = 1
BCG_STEPS = 20
BCG_TOL = 1e-3
V_CYCLE = 2

# 솔루션 후처리 개선을 위한 추가 CG 반복 설정
BCG_POST_IMPROVE = False
BCG_POST_STEPS = 1000
BCG_POST_TOL = 1e-7

EPS = 1.0e-12

def restrict(in_arr, out_arr):
    """
    고해상도 그리드에서 저해상도 그리드로 데이터를 축소(Restriction)합니다.
    """
    in_rows, in_cols = in_arr.shape
    out_rows, out_cols = out_arr.shape

    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filter_size = 0.5

    sy = dy / 2.0 - 0.5
    for y in range(out_rows):
        sx = dx / 2.0 - 0.5
        for x in range(out_cols):
            pix_val = 0.0
            w = 0.0
            
            ix_start = max(0, int(math.ceil(sx - dx * filter_size)))
            ix_end = min(int(math.floor(sx + dx * filter_size)), in_cols - 1)
            iy_start = max(0, int(math.ceil(sy - dy * filter_size)))
            iy_end = min(int(math.floor(sy + dy * filter_size)), in_rows - 1)

            for ix in range(ix_start, ix_end + 1):
                for iy in range(iy_start, iy_end + 1):
                    pix_val += in_arr[iy, ix]
                    w += 1.0
            
            if w > 0:
                out_arr[y, x] = pix_val / w
            
            sx += dx
        sy += dy

def prolongate(in_arr, out_arr):
    """
    저해상도 그리드에서 고해상도 그리드로 데이터를 보간(Prolongation)합니다.
    """
    in_rows, in_cols = in_arr.shape
    out_rows, out_cols = out_arr.shape

    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filter_size = 1.0

    sy = -dy / 2.0
    for y in range(out_rows):
        sx = -dx / 2.0
        for x in range(out_cols):
            pix_val = 0.0
            weight = 0.0

            ix_start = max(0, int(math.ceil(sx - filter_size)))
            ix_end = min(int(math.floor(sx + filter_size)), in_cols - 1)
            iy_start = max(0, int(math.ceil(sy - filter_size)))
            iy_end = min(int(math.floor(sy + filter_size)), in_rows - 1)

            for ix in range(ix_start, ix_end + 1):
                for iy in range(iy_start, iy_end + 1):
                    fx = abs(sx - ix)
                    fy = abs(sy - iy)
                    fval = (1.0 - fx) * (1.0 - fy)

                    pix_val += in_arr[iy, ix] * fval
                    weight += fval

            if weight != 0:
                out_arr[y, x] = pix_val / weight
            
            sx += dx
        sy += dy

def exact_sollution(F, U):
    """
    가장 거친 그리드(Coarsest grid)에서의 정확한 해를 구합니다.
    """
    U.fill(0.0)

def smooth(U, F):
    """
    주어진 레벨에서 Biconjugate Gradient를 사용하여 해를 평활화(Smoothing)합니다.
    """
    rows, cols = U.shape
    U_flat = U.reshape(-1)
    F_flat = F.reshape(-1)
    
    # 1D 뷰를 통해 값을 업데이트하면 원본 2D 배열인 U도 함께 업데이트됩니다.
    linbcg(F_flat, U_flat, BCG_TOL, BCG_STEPS, rows, cols)

def calculate_defect(D, U, F):
    """
    현재 근사해의 결함(Defect 또는 Residual)을 계산합니다.
    """
    # 벡터화된 연산을 사용하여 C++의 노드별 Neumann 경계 조건 연산을 구현합니다.
    U_e = np.empty_like(U)
    U_e[:, :-1] = U[:, 1:]
    U_e[:, -1] = U[:, -1]

    U_w = np.empty_like(U)
    U_w[:, 1:] = U[:, :-1]
    U_w[:, 0] = U[:, 0]

    U_s = np.empty_like(U)
    U_s[:-1, :] = U[1:, :]
    U_s[-1, :] = U[-1, :]

    U_n = np.empty_like(U)
    U_n[1:, :] = U[:-1, :]
    U_n[0, :] = U[0, :]

    D[:] = F - (U_e + U_w + U_n + U_s - 4.0 * U)

def add_correction(U, C):
    """
    보간된 오차 보정값을 현재 해에 더합니다.
    """
    U += C

def asolve(b, x):
    """
    전제조건자(Preconditioner) 행렬을 푸는 함수입니다.
    """
    x[:] = -4.0 * b

def atimes(x, res, rows, cols):
    """
    희소 행렬 연산을 수행합니다 (이산 라플라시안 연산자 적용).
    """
    X2D = x.reshape((rows, cols))
    RES2D = res.reshape((rows, cols))

    # 내부 영역
    RES2D[1:-1, 1:-1] = X2D[:-2, 1:-1] + X2D[2:, 1:-1] + X2D[1:-1, :-2] + X2D[1:-1, 2:] - 4.0 * X2D[1:-1, 1:-1]

    # 모서리 (Edges)
    RES2D[1:-1, 0] = X2D[:-2, 0] + X2D[2:, 0] + X2D[1:-1, 1] - 3.0 * X2D[1:-1, 0]
    RES2D[1:-1, -1] = X2D[:-2, -1] + X2D[2:, -1] + X2D[1:-1, -2] - 3.0 * X2D[1:-1, -1]
    RES2D[0, 1:-1] = X2D[1, 1:-1] + X2D[0, :-2] + X2D[0, 2:] - 3.0 * X2D[0, 1:-1]
    RES2D[-1, 1:-1] = X2D[-2, 1:-1] + X2D[-1, :-2] + X2D[-1, 2:] - 3.0 * X2D[-1, 1:-1]

    # 꼭짓점 (Corners)
    RES2D[0, 0] = X2D[1, 0] + X2D[0, 1] - 2.0 * X2D[0, 0]
    RES2D[-1, 0] = X2D[-2, 0] + X2D[-1, 1] - 2.0 * X2D[-1, 0]
    RES2D[0, -1] = X2D[1, -1] + X2D[0, -2] - 2.0 * X2D[0, -1]
    RES2D[-1, -1] = X2D[-2, -1] + X2D[-1, -2] - 2.0 * X2D[-1, -1]

def snrm(sx):
    """
    벡터의 유클리드 노름(L2 Norm)을 반환합니다.
    """
    return np.linalg.norm(sx)

def linbcg(b, x, tol, itmax, rows, cols):
    """
    선형 방정식을 풀기 위한 쌍공액 기울기법(Biconjugate Gradient Method)입니다.
    """
    n = len(b)
    p = np.zeros(n, dtype=np.float32)
    pp = np.zeros(n, dtype=np.float32)
    r = np.zeros(n, dtype=np.float32)
    rr = np.zeros(n, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    zz = np.zeros(n, dtype=np.float32)

    iter_count = 0
    atimes(x, r, rows, cols)
    
    r[:] = b - r
    rr[:] = r[:]
    atimes(r, rr, rows, cols)
    
    bnrm = snrm(b)
    if bnrm == 0.0:
        bnrm = 1.0

    asolve(r, z)

    while iter_count <= itmax:
        iter_count += 1
        asolve(rr, zz)
        
        bknum = np.dot(z, rr)
        
        if iter_count == 1:
            p[:] = z[:]
            pp[:] = zz[:]
        else:
            bk = bknum / bkden
            p[:] = z + bk * p
            pp[:] = zz + bk * pp
            
        bkden = bknum
        atimes(p, z, rows, cols)
        
        akden = np.dot(z, pp)
        ak = bknum / akden if akden != 0 else 0.0
        
        atimes(pp, zz, rows, cols)
        x[:] = x + ak * p
        r[:] = r - ak * z
        rr[:] = rr - ak * zz
        
        asolve(r, z)
        
        err = snrm(r) / bnrm
        if err <= tol:
            break
            
    return iter_count, err

def solve_pde_multigrid(F, U, progress_callback=None):
    """
    다중 그리드 알고리즘(Full Multigrid Algorithm)을 사용하여 편미분 방정식을 풉니다.
    """
    ymax, xmax = F.shape

    levels = 0
    mins = min(xmax, ymax)
    while mins >= MINS:
        levels += 1
        mins = mins // 2 + MODYF

    RHS = [None] * (levels + 1)
    IU = [None] * (levels + 1)
    VF = [None] * (levels + 1)

    VF[0] = np.zeros_like(F)
    RHS[0] = F
    IU[0] = np.copy(U)

    #===================================================
    print("1")
    #===================================================

    sx, sy = xmax, ymax
    for k in range(levels):
        sx = sx // 2 + MODYF
        sy = sy // 2 + MODYF
        
        RHS[k+1] = np.zeros((sy, sx), dtype=np.float32)
        IU[k+1] = np.zeros((sy, sx), dtype=np.float32)
        VF[k+1] = np.zeros((sy, sx), dtype=np.float32)
        
        restrict(RHS[k], RHS[k+1])

    exact_sollution(RHS[levels], IU[levels])

    #===================================================
    print("2")
    #===================================================

    for k in range(levels - 1, -1, -1):
        if progress_callback:
            progress_callback(20 + 70 * (levels - k) // (levels + 1))

        prolongate(IU[k+1], IU[k])
        VF[k][:] = RHS[k][:]

        #===================================================
        print("3")
        #===================================================

        for cycle in range(V_CYCLE):
            for k2 in range(k, levels):
                if k2 != k:
                    IU[k2].fill(0.0)

                for _ in range(SMOOTH_IT):
                    smooth(IU[k2], VF[k2])

                D = np.zeros_like(IU[k2])
                calculate_defect(D, IU[k2], VF[k2])
                restrict(D, VF[k2+1])

            exact_sollution(VF[levels], IU[levels])

            for k2 in range(levels - 1, k - 1, -1):
                C = np.zeros_like(IU[k2])
                prolongate(IU[k2+1], C)
                add_correction(IU[k2], C)

                for _ in range(SMOOTH_IT):
                    smooth(IU[k2], VF[k2])
            
            #===================================================
            print("4")
            #===================================================

    U[:] = IU[0][:]

    if BCG_POST_IMPROVE:
        U_flat = U.reshape(-1)
        F_flat = F.reshape(-1)
        linbcg(F_flat, U_flat, BCG_POST_TOL, BCG_POST_STEPS, ymax, xmax)

    if progress_callback:
        progress_callback(90)