import numpy as np
import pyfftw
import multiprocessing

# 반복적인 FFT 연산 수행 시 플랜 생성 오버헤드를 줄이기 위해 캐싱 활성화
pyfftw.interfaces.cache.enable()

def get_lambda(n: int) -> np.ndarray:
    """1D 노이만 경계조건 라플라스 연산자의 고유값 계산."""
    i = np.arange(n, dtype=np.float32)
    val = -4.0 * np.sin(i * np.pi / (2.0 * n)) ** 2
    return val.astype(np.float32)

def solve_pde_fftw(F: np.ndarray, hpf_sigma: float = 0.0) -> np.ndarray:
    """
    FFTW를 활용한 2D 푸아송 방정식 직접 해법.
    
    Args:
        F: 발산(Divergence) 행렬
        hpf_sigma: 저주파 억제를 위한 High-Pass Filter의 강도 (0.0 이면 필터 미적용)
    """
    H, W = F.shape
    threads = multiprocessing.cpu_count()

    F_aligned = pyfftw.empty_aligned((H, W), dtype=np.float32)
    F_aligned[:] = F.astype(np.float32)

    # 1. 고유벡터 공간으로의 직교 변환 (DCT-II)
    F_tr = pyfftw.interfaces.scipy_fft.dctn(
        F_aligned, type=2, norm='ortho', workers=threads
    ).astype(np.float32)

    # 2. 고유값을 이용한 해 도출
    ly = get_lambda(H)
    lx = get_lambda(W)
    denom = ly[:, np.newaxis] + lx[np.newaxis, :]

    denom[0, 0] = 1.0  # ZeroDivision 방지
    F_tr /= denom
    F_tr[0, 0] = 0.0   # DC 성분 = 0으로 고정
    
    # 3. Gaussian High-Pass Filter 적용 (저주파 억제)
    if hpf_sigma > 0.0:
        # 0 ~ 1 사이로 정규화된 주파수 공간 좌표 생성
        ky = np.arange(H, dtype=np.float32) / H
        kx = np.arange(W, dtype=np.float32) / W
        
        # 중심(DC 성분)으로부터의 거리 제곱 행렬 계산
        D2 = ky[:, np.newaxis]**2 + kx[np.newaxis, :]**2
        
        # 가우시안 하이패스 필터 생성 (저주파는 0에 수렴, 고주파는 1에 수렴)
        H_filter = 1.0 - np.exp(-D2 / (2.0 * hpf_sigma**2))
        
        # 주파수 행렬에 필터 곱 연산 수행
        F_tr *= H_filter

    # 4. 고유벡터 공간에서 일반 공간으로 역변환 (DCT-III)
    U = pyfftw.interfaces.scipy_fft.idctn(
        F_tr, type=2, norm='ortho', workers=threads
    ).astype(np.float32)

    return U

def residual_pde(U: np.ndarray, F: np.ndarray) -> float:
    """해의 잔차(Residual) 오차 계산."""
    R = np.empty_like(U)

    # 내부
    R[1:-1, 1:-1] = (U[:-2, 1:-1] + U[2:, 1:-1] +
                     U[1:-1, :-2] + U[1:-1, 2:] - 4.0 * U[1:-1, 1:-1])

    # 좌/우 모서리
    R[1:-1,  0] = U[:-2,  0] + U[2:,  0] + U[1:-1,  1] - 3.0 * U[1:-1,  0]
    R[1:-1, -1] = U[:-2, -1] + U[2:, -1] + U[1:-1, -2] - 3.0 * U[1:-1, -1]

    # 상/하 모서리
    R[ 0, 1:-1] = U[ 1, 1:-1] + U[ 0, :-2] + U[ 0, 2:] - 3.0 * U[ 0, 1:-1]
    R[-1, 1:-1] = U[-2, 1:-1] + U[-1, :-2] + U[-1, 2:] - 3.0 * U[-1, 1:-1]

    # 꼭짓점
    R[ 0,  0] = U[ 1,  0] + U[ 0,  1] - 2.0 * U[ 0,  0]
    R[-1,  0] = U[-2,  0] + U[-1,  1] - 2.0 * U[-1,  0]
    R[ 0, -1] = U[ 1, -1] + U[ 0, -2] - 2.0 * U[ 0, -1]
    R[-1, -1] = U[-2, -1] + U[-1, -2] - 2.0 * U[-1, -1]

    return float(np.sqrt(np.sum((R - F) ** 2)))