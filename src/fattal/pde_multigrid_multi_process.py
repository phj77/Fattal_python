import math
import numpy as np

try:
    from numba import njit, prange
except ImportError:
    # Numba가 설치되지 않은 환경을 대비한 더미 데코레이터
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

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

@njit(parallel=True, fastmath=True)
def restrict(in_arr, out_arr):
    in_rows, in_cols = in_arr.shape
    out_rows, out_cols = out_arr.shape

    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filter_size = 0.5

    for y in prange(out_rows):
        sy = dy / 2.0 - 0.5 + y * dy
        for x in range(out_cols):
            sx = dx / 2.0 - 0.5 + x * dx
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

@njit(parallel=True, fastmath=True)
def prolongate(in_arr, out_arr):
    in_rows, in_cols = in_arr.shape
    out_rows, out_cols = out_arr.shape

    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filter_size = 1.0

    for y in prange(out_rows):
        sy = -dy / 2.0 + y * dy
        for x in range(out_cols):
            sx = -dx / 2.0 + x * dx
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

@njit(fastmath=True)
def exact_sollution(F, U):
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i, j] = 0.0

def smooth(U, F):
    rows, cols = U.shape
    U_flat = U.reshape(-1)
    F_flat = F.reshape(-1)
    linbcg(F_flat, U_flat, BCG_TOL, BCG_STEPS, rows, cols)

@njit(parallel=True, fastmath=True)
def calculate_defect(D, U, F):
    rows, cols = U.shape
    for i in prange(rows):
        for j in range(cols):
            val = F[i, j] + 4.0 * U[i, j]
            
            if j < cols - 1: val -= U[i, j+1]
            else: val -= U[i, cols-1]
            
            if j > 0: val -= U[i, j-1]
            else: val -= U[i, 0]
            
            if i < rows - 1: val -= U[i+1, j]
            else: val -= U[rows-1, j]
            
            if i > 0: val -= U[i-1, j]
            else: val -= U[0, j]
            
            D[i, j] = val

@njit(fastmath=True)
def add_correction(U, C):
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i, j] += C[i, j]

@njit(parallel=True, fastmath=True)
def asolve(b, x):
    for i in prange(len(b)):
        x[i] = b[i] * -4.0

@njit(parallel=True, fastmath=True)
def atimes(x, res, rows, cols):
    for i in prange(rows):
        for j in range(cols):
            idx = i * cols + j
            val = 0.0
            
            if j < cols - 1: val += x[idx + 1]
            else: val += x[idx]
            
            if j > 0: val += x[idx - 1]
            else: val += x[idx]
            
            if i < rows - 1: val += x[(i + 1) * cols + j]
            else: val += x[idx]
            
            if i > 0: val += x[(i - 1) * cols + j]
            else: val += x[idx]
            
            res[idx] = val - 4.0 * x[idx]

@njit(fastmath=True)
def snrm(sx):
    s = 0.0
    for i in range(len(sx)):
        s += sx[i] * sx[i]
    return math.sqrt(s)

@njit(fastmath=True)
def linbcg(b, x, tol, itmax, rows, cols):
    n = len(b)
    p = np.empty(n, dtype=np.float32)
    pp = np.empty(n, dtype=np.float32)
    r = np.empty(n, dtype=np.float32)
    rr = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    zz = np.empty(n, dtype=np.float32)

    iter_count = 0
    atimes(x, r, rows, cols)
    
    for i in range(n):
        r[i] = b[i] - r[i]
        rr[i] = r[i]
        
    atimes(r, rr, rows, cols)
    
    bnrm = snrm(b)
    if bnrm == 0.0:
        bnrm = 1.0

    asolve(r, z)
    
    bkden = 0.0
    while iter_count <= itmax:
        iter_count += 1
        asolve(rr, zz)
        
        bknum = 0.0
        for i in range(n):
            bknum += z[i] * rr[i]
        
        if iter_count == 1:
            for i in range(n):
                p[i] = z[i]
                pp[i] = zz[i]
        else:
            bk = bknum / bkden
            for i in range(n):
                p[i] = bk * p[i] + z[i]
                pp[i] = bk * pp[i] + zz[i]
            
        bkden = bknum
        atimes(p, z, rows, cols)
        
        akden = 0.0
        for i in range(n):
            akden += z[i] * pp[i]
            
        ak = bknum / akden if akden != 0 else 0.0
        
        atimes(pp, zz, rows, cols)
        
        for i in range(n):
            x[i] += ak * p[i]
            r[i] -= ak * z[i]
            rr[i] -= ak * zz[i]
        
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
    
    D_pool = [None] * (levels + 1)
    C_pool = [None] * (levels + 1)

    VF[0] = np.zeros_like(F)
    RHS[0] = F
    IU[0] = np.copy(U)

    sx, sy = xmax, ymax
    for k in range(levels):
        sx = sx // 2 + MODYF
        sy = sy // 2 + MODYF
        
        RHS[k+1] = np.zeros((sy, sx), dtype=np.float32)
        IU[k+1] = np.zeros((sy, sx), dtype=np.float32)
        VF[k+1] = np.zeros((sy, sx), dtype=np.float32)
        
        D_pool[k] = np.zeros((RHS[k].shape[0], RHS[k].shape[1]), dtype=np.float32)
        C_pool[k] = np.zeros((RHS[k].shape[0], RHS[k].shape[1]), dtype=np.float32)
        
        restrict(RHS[k], RHS[k+1])

    D_pool[levels] = np.zeros((sy, sx), dtype=np.float32)
    C_pool[levels] = np.zeros((sy, sx), dtype=np.float32)

    exact_sollution(RHS[levels], IU[levels])

    for k in range(levels - 1, -1, -1):
        if progress_callback:
            progress_callback(20 + 70 * (levels - k) // (levels + 1))

        prolongate(IU[k+1], IU[k])
        VF[k][:] = RHS[k][:]

        for cycle in range(V_CYCLE):
            for k2 in range(k, levels):
                if k2 != k:
                    IU[k2].fill(0.0)

                for _ in range(SMOOTH_IT):
                    smooth(IU[k2], VF[k2])

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


    U[:] = IU[0][:]

    if BCG_POST_IMPROVE:
        U_flat = U.reshape(-1)
        F_flat = F.reshape(-1)
        linbcg(F_flat, U_flat, BCG_POST_TOL, BCG_POST_STEPS, ymax, xmax)

    if progress_callback:
        progress_callback(90)

    return U