import math
import numpy as np
try:
    from numba import njit
except ImportError:
    # Numba가 설치되지 않은 환경을 대비한 더미 데코레이터
    def njit(func):
        return func

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

@njit
def restrict(in_arr, out_arr):
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

@njit
def prolongate(in_arr, out_arr):
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
    U.fill(0.0)

def smooth(U, F):
    rows, cols = U.shape
    U_flat = U.reshape(-1)
    F_flat = F.reshape(-1)
    linbcg(F_flat, U_flat, BCG_TOL, BCG_STEPS, rows, cols)

def calculate_defect(D, U, F):
    """
    불필요한 4개의 임시 배열 생성을 제거하고 In-place 연산으로 대체.
    수학적으로 기존의 D[:] = F - (U_e + U_w + U_n + U_s - 4*U) 연산과 완전히 동일함.
    """
    # D = F + 4.0 * U
    np.add(F, U * 4.0, out=D)

    # - U_e
    D[:, :-1] -= U[:, 1:]
    D[:, -1] -= U[:, -1]

    # - U_w
    D[:, 1:] -= U[:, :-1]
    D[:, 0] -= U[:, 0]

    # - U_s
    D[:-1, :] -= U[1:, :]
    D[-1, :] -= U[-1, :]

    # - U_n
    D[1:, :] -= U[:-1, :]
    D[0, :] -= U[0, :]

def add_correction(U, C):
    U += C

def asolve(b, x):
    # 배열 할당 없는 In-place 연산
    np.multiply(b, -4.0, out=x)

def atimes(x, res, rows, cols):
    X2D = x.reshape((rows, cols))
    RES2D = res.reshape((rows, cols))

    # 내부 영역
    RES2D[1:-1, 1:-1] = X2D[:-2, 1:-1] + X2D[2:, 1:-1] + X2D[1:-1, :-2] + X2D[1:-1, 2:] - 4.0 * X2D[1:-1, 1:-1]

    # 모서리
    RES2D[1:-1, 0] = X2D[:-2, 0] + X2D[2:, 0] + X2D[1:-1, 1] - 3.0 * X2D[1:-1, 0]
    RES2D[1:-1, -1] = X2D[:-2, -1] + X2D[2:, -1] + X2D[1:-1, -2] - 3.0 * X2D[1:-1, -1]
    RES2D[0, 1:-1] = X2D[1, 1:-1] + X2D[0, :-2] + X2D[0, 2:] - 3.0 * X2D[0, 1:-1]
    RES2D[-1, 1:-1] = X2D[-2, 1:-1] + X2D[-1, :-2] + X2D[-1, 2:] - 3.0 * X2D[-1, 1:-1]

    # 꼭짓점
    RES2D[0, 0] = X2D[1, 0] + X2D[0, 1] - 2.0 * X2D[0, 0]
    RES2D[-1, 0] = X2D[-2, 0] + X2D[-1, 1] - 2.0 * X2D[-1, 0]
    RES2D[0, -1] = X2D[1, -1] + X2D[0, -2] - 2.0 * X2D[0, -1]
    RES2D[-1, -1] = X2D[-2, -1] + X2D[-1, -2] - 2.0 * X2D[-1, -1]

def snrm(sx):
    return np.linalg.norm(sx)

def linbcg(b, x, tol, itmax, rows, cols):
    n = len(b)
    # np.zeros 대신 np.empty를 사용하여 할당 오버헤드 최소화
    p = np.empty(n, dtype=np.float32)
    pp = np.empty(n, dtype=np.float32)
    r = np.empty(n, dtype=np.float32)
    rr = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    zz = np.empty(n, dtype=np.float32)

    iter_count = 0
    atimes(x, r, rows, cols)
    
    # r[:] = b - r
    np.subtract(b, r, out=r)
    
    # rr 전체를 덮어쓰지만 기존 코드의 흐름(rr[:] = r[:])을 정확히 반영
    np.copyto(rr, r)
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
            np.copyto(p, z)
            np.copyto(pp, zz)
        else:
            bk = bknum / bkden
            # In-place 갱신
            p *= bk
            p += z
            pp *= bk
            pp += zz
            
        bkden = bknum
        atimes(p, z, rows, cols)
        
        akden = np.dot(z, pp)
        ak = bknum / akden if akden != 0 else 0.0
        
        atimes(pp, zz, rows, cols)
        
        # In-place 감가산으로 메모리 재할당 방지
        x += ak * p
        r -= ak * z
        rr -= ak * zz
        
        asolve(r, z)
        
        err = snrm(r) / bnrm
        if err <= tol:
            break
            
    return iter_count, err

def solve_pde_multigrid(F, U, progress_callback=None):
    ymax, xmax = F.shape

    levels = 0
    mins = min(xmax, ymax)
    while mins >= MINS:
        levels += 1
        mins = mins // 2 + MODYF

    RHS = [None] * (levels + 1)
    IU = [None] * (levels + 1)
    VF = [None] * (levels + 1)
    
    # V-사이클 내 반복적인 메모리 할당 방지를 위한 배열 풀(Pool) 생성
    D_pool = [None] * (levels + 1)
    C_pool = [None] * (levels + 1)

    VF[0] = np.zeros_like(F)
    RHS[0] = F
    IU[0] = np.copy(U)

    print("1")

    sx, sy = xmax, ymax
    for k in range(levels):
        sx = sx // 2 + MODYF
        sy = sy // 2 + MODYF
        
        RHS[k+1] = np.zeros((sy, sx), dtype=np.float32)
        IU[k+1] = np.zeros((sy, sx), dtype=np.float32)
        VF[k+1] = np.zeros((sy, sx), dtype=np.float32)
        
        # 각 레벨에 맞는 D, C 버퍼 미리 할당
        D_pool[k] = np.zeros((RHS[k].shape[0], RHS[k].shape[1]), dtype=np.float32)
        C_pool[k] = np.zeros((RHS[k].shape[0], RHS[k].shape[1]), dtype=np.float32)
        
        restrict(RHS[k], RHS[k+1])

    D_pool[levels] = np.zeros((sy, sx), dtype=np.float32)
    C_pool[levels] = np.zeros((sy, sx), dtype=np.float32)

    exact_sollution(RHS[levels], IU[levels])

    print("2")

    for k in range(levels - 1, -1, -1):
        if progress_callback:
            progress_callback(20 + 70 * (levels - k) // (levels + 1))

        prolongate(IU[k+1], IU[k])
        VF[k][:] = RHS[k][:]

        print("3")

        for cycle in range(V_CYCLE):
            for k2 in range(k, levels):
                if k2 != k:
                    IU[k2].fill(0.0)

                for _ in range(SMOOTH_IT):
                    smooth(IU[k2], VF[k2])

                # 루프 내 np.zeros_like 생성을 제거하고 사전 할당된 버퍼 사용
                D = D_pool[k2]
                calculate_defect(D, IU[k2], VF[k2])
                restrict(D, VF[k2+1])

            exact_sollution(VF[levels], IU[levels])

            for k2 in range(levels - 1, k - 1, -1):
                C = C_pool[k2]
                prolongate(IU[k2+1], C)
                add_correction(IU[k2], C)

                for _ in range(SMOOTH_IT):
                    smooth(IU[k2], VF[k2])
            
            print("4")

    U[:] = IU[0][:]

    if BCG_POST_IMPROVE:
        U_flat = U.reshape(-1)
        F_flat = F.reshape(-1)
        linbcg(F_flat, U_flat, BCG_POST_TOL, BCG_POST_STEPS, ymax, xmax)

    if progress_callback:
        progress_callback(90)

    return U