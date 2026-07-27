import numpy as np
import pyfftw
import pyfftw.interfaces.scipy_fft as fftw_fft
import multiprocessing
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import utils

# 연산 속도 최적화를 위해 PyFFTW 캐시를 활성화하고, 가용한 최대 스레드를 할당합니다.
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
used_thread = 16

def transform_ev2normal(A: np.ndarray) -> np.ndarray:
    """고유 벡터 공간에서 원래 공간으로 변환합니다."""
    height, width = A.shape
    
    # Numpy 배열 연산을 통한 빠른 스케일링 적용
    S = np.ones((height, width), dtype=np.float32)
    S[1:-1, 1:-1] = 0.25
    S[0, 1:-1] = 0.5
    S[-1, 1:-1] = 0.5
    S[1:-1, 0] = 0.5
    S[1:-1, -1] = 0.5
    
    A_scaled = A * S
    
    # FFTW_REDFT00에 대응하는 DCT-I (type=1) 실행
    T = fftw_fft.dctn(A_scaled, type=1, norm=None, workers=used_thread).astype(np.float32)
    return T

def transform_normal2ev(A: np.ndarray) -> np.ndarray:
    """원래 공간에서 고유 벡터 공간으로 변환합니다."""
    height, width = A.shape
    
    # 2D DCT-I 실행
    T = fftw_fft.dctn(A, type=1, norm=None, workers=used_thread).astype(np.float32)
    
    # 출력 매트릭스 스케일링
    S = np.ones((height, width), dtype=np.float32)
    S[0, :] *= 0.5
    S[-1, :] *= 0.5
    S[:, 0] *= 0.5
    S[:, -1] *= 0.5
    S *= 1.0 / ((height - 1) * (width - 1))
    
    return T * S

def get_lambda(n: int) -> np.ndarray:
    """1D 라플라스 연산자의 고유값을 반환합니다."""
    i = np.arange(n, dtype=np.float64)
    return -4.0 * np.sin(i * np.pi / (2.0 * (n - 1)))**2

def make_compatible_boundary(F: np.ndarray) -> np.ndarray:
    """해가 존재할 수 있도록 경계 조건을 호환되게 조정합니다."""
    height, width = F.shape
    F_adj = np.copy(F)
    
    # 복잡한 합산 로직을 가중치 행렬 곱셈으로 단일화하여 최적화
    W = np.ones((height, width), dtype=np.float64)
    W[0, :] *= 0.5
    W[-1, :] *= 0.5
    W[:, 0] *= 0.5
    W[:, -1] *= 0.5
    
    total_sum = np.sum(F_adj * W)
    add = -total_sum / (height + width - 3)
    
    # 경계값 갱신 (모서리가 중복 적용되지 않도록 슬라이싱 주의)
    F_adj[0, :] += add
    F_adj[-1, :] += add
    F_adj[1:-1, 0] += add
    F_adj[1:-1, -1] += add
    
    return F_adj

def solve_pde_fft(F: np.ndarray, adjust_bound: bool = True, hpf_sigma: float = 0.0) -> np.ndarray:
    """
    노이만 경계 조건을 사용하여 2D Poisson 방정식을 계산합니다.
    """
    utils.print_elapsed("       [fft_solve] 시작")
    height, width = F.shape
    
    if adjust_bound:
        F = make_compatible_boundary(F)
        utils.print_elapsed("       [fft_solve] 경계 조건 호환성 조정 완료")
        
    F_tr = transform_normal2ev(F)
    utils.print_elapsed("       [fft_solve] normal2ev 변환 완료")
    
    l1 = get_lambda(height)
    l2 = get_lambda(width)
    
    # 브로드캐스팅을 사용한 고유값 매트릭스 생성
    denom = l1[:, None] + l2[None, :]
    
    # 0으로 나누는 경고를 방지하고 분모가 0인 곳을 0으로 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        F_tr = np.where(denom == 0, 0, F_tr / denom)
        
    F_tr[0, 0] = 0.0 # 상수를 결정하는 임의의 값 설정
    
    # --- 가우시안 High-Pass Filter 적용 ---
    if hpf_sigma > 0.0:
        ky = np.arange(height, dtype=np.float32) / height
        kx = np.arange(width, dtype=np.float32) / width
        
        D2 = ky[:, np.newaxis]**2 + kx[np.newaxis, :]**2
        H_filter = 1.0 - np.exp(-D2 / (2.0 * hpf_sigma**2))
        
        F_tr *= H_filter
    # --------------------------------------
    utils.print_elapsed("       [fft_solve] 고유값 및 HPF 필터 연산 완료")
        
    U = transform_ev2normal(F_tr)
    
    # 최대값이 0이 되도록 정규화
    U -= np.max(U)
    utils.print_elapsed("       [fft_solve] ev2normal 역변환 및 U 정규화 완료")
    
    return U

def residual_pde(U: np.ndarray, F: np.ndarray) -> float:
    """내부 포인트들에 대한 잔차(오차)의 L2 norm을 계산합니다."""
    laplace = (-4.0 * U[1:-1, 1:-1] + 
               U[:-2, 1:-1] + U[2:, 1:-1] + 
               U[1:-1, :-2] + U[1:-1, 2:])
               
    res = np.sum((laplace - F[1:-1, 1:-1])**2)
    return float(np.sqrt(res))
