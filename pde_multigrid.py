import numpy as np
import math

def asolve(b):
    return -4.0 * b

def atimes(x):
    rows, cols = x.shape
    res = np.zeros_like(x)
    if rows > 2 and cols > 2:
        res[1:-1, 1:-1] = x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1]

        res[1:-1, 0] = x[:-2, 0] + x[2:, 0] + x[1:-1, 1] - 3 * x[1:-1, 0]
        res[1:-1, -1] = x[:-2, -1] + x[2:, -1] + x[1:-1, -2] - 3 * x[1:-1, -1]
        res[0, 1:-1] = x[1, 1:-1] + x[0, :-2] + x[0, 2:] - 3 * x[0, 1:-1]
        res[-1, 1:-1] = x[-2, 1:-1] + x[-1, :-2] + x[-1, 2:] - 3 * x[-1, 1:-1]

        res[0, 0] = x[1, 0] + x[0, 1] - 2 * x[0, 0]
        res[-1, 0] = x[-2, 0] + x[-1, 1] - 2 * x[-1, 0]
        res[0, -1] = x[1, -1] + x[0, -2] - 2 * x[0, -1]
        res[-1, -1] = x[-2, -1] + x[-1, -2] - 2 * x[-1, -1]
    else:
        for r in range(rows):
            for c in range(cols):
                w = c - 1 if c > 0 else 0
                n = r - 1 if r > 0 else 0
                s = r + 1 if r + 1 < rows else r
                e = c + 1 if c + 1 < cols else c
                res[r, c] = x[r, e] + x[r, w] + x[s, c] + x[n, c] - 4 * x[r, c]
    return res

def linbcg(b, x, tol, itmax):
    n = b.size
    
    iters = 0
    ax = atimes(x)
    r = b - ax
    rr = r.copy()
    
    bnrm = np.linalg.norm(b)
    if bnrm == 0.0:
        bnrm = 1.0
        
    z = asolve(r)
    
    p = np.zeros_like(b)
    pp = np.zeros_like(b)
    bkden = 1.0
    
    while iters <= itmax:
        iters += 1
        zz = asolve(rr)
        bknum = np.sum(z * rr)
        
        if iters == 1:
            p = z.copy()
            pp = zz.copy()
        else:
            bk = bknum / bkden
            p = z + bk * p
            pp = zz + bk * pp
            
        bkden = bknum
        az = atimes(p)
        akden = np.sum(az * pp)
        ak = bknum / akden if akden != 0 else 0.0
        
        azz = atimes(pp)
        x = x + ak * p
        r = r - ak * az
        rr = rr - ak * azz
        
        z = asolve(r)
        err = np.linalg.norm(r) / bnrm
        
        if err <= tol:
            break
            
    return x, iters, err

def restrict(in_arr, out_shape):
    out_rows, out_cols = out_shape
    in_rows, in_cols = in_arr.shape
    out_arr = np.zeros(out_shape, dtype=np.float32)
    
    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filterSize = 0.5
    
    sy_base = np.arange(out_rows) * dy + dy / 2 - 0.5
    sx_base = np.arange(out_cols) * dx + dx / 2 - 0.5
    
    for y in range(out_rows):
        sy = sy_base[y]
        min_iy = int(max(0, math.ceil(sy - dy * filterSize)))
        max_iy = int(min(math.floor(sy + dy * filterSize), in_rows - 1))
        
        for x in range(out_cols):
            sx = sx_base[x]
            min_ix = int(max(0, math.ceil(sx - dx * filterSize)))
            max_ix = int(min(math.floor(sx + dx * filterSize), in_cols - 1))
            
            if min_iy <= max_iy and min_ix <= max_ix:
                block = in_arr[min_iy:max_iy+1, min_ix:max_ix+1]
                out_arr[y, x] = np.sum(block) / block.size
                
    return out_arr

def prolongate(in_arr, out_shape):
    out_rows, out_cols = out_shape
    in_rows, in_cols = in_arr.shape
    out_arr = np.zeros(out_shape, dtype=np.float32)
    
    dx = in_cols / out_cols
    dy = in_rows / out_rows
    filterSize = 1.0
    
    sy_base = np.arange(out_rows) * dy - dy / 2
    sx_base = np.arange(out_cols) * dx - dx / 2
    
    for y in range(out_rows):
        sy = sy_base[y]
        min_iy = int(max(0, math.ceil(sy - filterSize)))
        max_iy = int(min(math.floor(sy + filterSize), in_rows - 1))
        
        for x in range(out_cols):
            sx = sx_base[x]
            min_ix = int(max(0, math.ceil(sx - filterSize)))
            max_ix = int(min(math.floor(sx + filterSize), in_cols - 1))
            
            pixVal = 0.0
            weight = 0.0
            
            for iy in range(min_iy, max_iy + 1):
                fy = abs(sy - iy)
                fval_y = 1.0 - fy
                for ix in range(min_ix, max_ix + 1):
                    fx = abs(sx - ix)
                    fval = (1.0 - fx) * fval_y
                    pixVal += in_arr[iy, ix] * fval
                    weight += fval
                    
            if weight != 0:
                out_arr[y, x] = pixVal / weight
                
    return out_arr

def exact_sollution(F, U):
    U.fill(0.0)

def smooth(U, F, tol, steps):
    U_new, _, _ = linbcg(F, U, tol, steps)
    return U_new

def calculate_defect(U, F):
    return F - atimes(U)

def add_correction(U, C):
    U += C

def solve_pde_multigrid(F):
    MODYF = 0
    MINS = 16
    SMOOTH_IT = 1
    BCG_STEPS = 20
    BCG_TOL = 1e-3
    V_CYCLE = 2
    BCG_POST_IMPROVE = False
    BCG_POST_STEPS = 1000
    BCG_POST_TOL = 1e-7

    U = np.zeros_like(F, dtype=np.float32)
    ymax, xmax = F.shape
    
    levels = 0
    mins = min(xmax, ymax)
    while mins >= MINS:
        levels += 1
        mins = mins // 2 + MODYF

    RHS = [None] * (levels + 1)
    IU = [None] * (levels + 1)
    VF = [None] * (levels + 1)

    VF[0] = np.zeros((ymax, xmax), dtype=np.float32)
    RHS[0] = F.copy()
    IU[0] = U.copy()

    sx, sy = xmax, ymax
    for k in range(levels):
        sx = sx // 2 + MODYF
        sy = sy // 2 + MODYF
        
        RHS[k+1] = np.zeros((sy, sx), dtype=np.float32)
        IU[k+1] = np.zeros((sy, sx), dtype=np.float32)
        VF[k+1] = np.zeros((sy, sx), dtype=np.float32)
        
        RHS[k+1] = restrict(RHS[k], (sy, sx))

    exact_sollution(RHS[levels], IU[levels])

    for k in range(levels - 1, -1, -1):
        IU[k] = prolongate(IU[k+1], (IU[k].shape[0], IU[k].shape[1]))
        VF[k] = RHS[k].copy()

        for cycle in range(V_CYCLE):
            for k2 in range(k, levels):
                if k2 != k:
                    IU[k2].fill(0.0)

                for _ in range(SMOOTH_IT):
                    IU[k2] = smooth(IU[k2], VF[k2], BCG_TOL, BCG_STEPS)

                D = calculate_defect(IU[k2], VF[k2])
                VF[k2+1] = restrict(D, (VF[k2+1].shape[0], VF[k2+1].shape[1]))

            exact_sollution(VF[levels], IU[levels])

            for k2 in range(levels - 1, k - 1, -1):
                C = prolongate(IU[k2+1], (IU[k2].shape[0], IU[k2].shape[1]))
                add_correction(IU[k2], C)

                for _ in range(SMOOTH_IT):
                    IU[k2] = smooth(IU[k2], VF[k2], BCG_TOL, BCG_STEPS)

    np.copyto(U, IU[0])

    if BCG_POST_IMPROVE:
        U, _, _ = linbcg(F, U, BCG_POST_TOL, BCG_POST_STEPS)

    return U