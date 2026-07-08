import numpy as np
import pyfftw
import pyfftw.interfaces.scipy_fft as fftw_fft
import multiprocessing
from utils import utils

# 연산 속도 최적화를 위해 PyFFTW 캐시를 활성화하고, 가용한 최대 스레드를 할당합니다.
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

def transform_ev2normal(A: np.ndarray) -> np.ndarray:
    """
    고유 벡터 공간에서 원래 공간으로 변환합니다.
    
    [DCT-II 변경 사항]
    - 기존 (DCT-I): 타입 1 DCT (fftw_fft.dctn type=1)와 경계/코너 샘플 스케일링 S 행렬 곱셈을 사용.
    - 변경 (DCT-II): 경계 중심 대칭이 샘플과 샘플 사이(반 칸 밖)에 형성되는 IDCT-II / DCT-III 변환 적용.
      norm='ortho' 옵션을 사용하여 전방/역방향 직교성을 보장하며, 복잡한 경계/코너 스케일링 S가 필요 없음.
    """
    # Inverse DCT-II (DCT-III) 사용 (norm='ortho'로 단위 정규화)
    T = fftw_fft.idctn(A, type=2, norm='ortho', workers=10).astype(np.float32)
    return T

def transform_normal2ev(A: np.ndarray) -> np.ndarray:
    """
    원래 공간에서 고유 벡터 공간으로 변환합니다.
    
    [DCT-II 변경 사항]
    - 기존 (DCT-I): 타입 1 DCT 및 1 / ((height-1)*(width-1)) 및 경계 0.5 스케일링 행렬 S 곱셈 사용.
    - 변경 (DCT-II): 타입 2 DCT (fftw_fft.dctn type=2, norm='ortho') 적용.
      직교 규격화(norm='ortho')를 사용하므로 수동 스케일링이 필요 없음.
    """
    # 2D DCT-II 실행 (norm='ortho')
    T = fftw_fft.dctn(A, type=2, norm='ortho', workers=10).astype(np.float32)
    return T

def get_lambda(n: int) -> np.ndarray:
    """
    1D 라플라스 연산자의 고유값을 반환합니다.
    
    [DCT-II 변경 사항]
    - 기존 (DCT-I):  -4.0 * sin^2( i * pi / (2 * (n - 1)) )
    - 변경 (DCT-II): -4.0 * sin^2( i * pi / (2 * n) )
      경계 대칭축이 -0.5, n-0.5 (반 칸 밖)에 위치하므로 분모가 (n-1)이 아닌 n이 됨.
    """
    i = np.arange(n, dtype=np.float64)
    return -4.0 * np.sin(i * np.pi / (2.0 * n))**2

def make_compatible_boundary(F: np.ndarray) -> np.ndarray:
    """
    해가 존재할 수 있도록 경계 조건을 호환되게 조정합니다.
    
    [DCT-II 변경 사항]
    - 기존 (DCT-I): 경계 샘플 가중치(모서리 0.25, 테두리 0.5)를 고려한 복잡한 가중치 합산 및 경계 셀 업데이트.
    - 변경 (DCT-II): 반 칸 노이만 경계 조건에서는 모든 샘플 격자가 동일한 체적/면적을 가짐.
      Poisson 방정식 해 존재 조건(Neumann compatibility): Integral of F = 0.
      단순히 전체 행렬의 평균(mean)을 차감하는 것만으로 완벽한 호환성 조건 만족.
    """
    F_adj = F - np.mean(F)
    return F_adj

def solve_pde_fft(F: np.ndarray, adjust_bound: bool = True, hpf_sigma: float = 0.0) -> np.ndarray:
    """
    샘플 사이 반 칸 떨어진 노이만 경계 조건을 사용하여 2D Poisson 방정식을 계산합니다 (DCT-II).
    
    Args:
        F: 발산(Divergence) 행렬
        adjust_bound: 경계 조건 호환성 조정 여부
        hpf_sigma: 저주파 억제를 위한 High-Pass Filter 강도 (0.0 이면 적용 안 함)
    """
    utils.print_elapsed("       [fft_solve DCT-II] 시작")
    height, width = F.shape
    
    if adjust_bound:
        F = make_compatible_boundary(F)
        utils.print_elapsed("       [fft_solve DCT-II] 경계 조건 호환성(전체 평균 제거) 조정 완료")
        
    F_tr = transform_normal2ev(F)
    utils.print_elapsed("       [fft_solve DCT-II] normal2ev (DCT-II) 변환 완료")
    
    l1 = get_lambda(height)
    l2 = get_lambda(width)
    
    # 브로드캐스팅을 사용한 고유값 매트릭스 생성
    denom = l1[:, None] + l2[None, :]
    
    # 0으로 나누는 경고를 방지하고 분모가 0인 곳을 0으로 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        F_tr = np.where(denom == 0, 0, F_tr / denom)
        
    F_tr[0, 0] = 0.0 # 상수를 결정하는 임의의 값 설정 (DC 성분 제거)
    
    # --- 가우시안 High-Pass Filter 적용 ---
    if hpf_sigma > 0.0:
        ky = np.arange(height, dtype=np.float32) / height
        kx = np.arange(width, dtype=np.float32) / width
        
        D2 = ky[:, np.newaxis]**2 + kx[np.newaxis, :]**2
        H_filter = 1.0 - np.exp(-D2 / (2.0 * hpf_sigma**2))
        
        F_tr *= H_filter
    # --------------------------------------
    utils.print_elapsed("       [fft_solve DCT-II] 고유값 및 HPF 필터 연산 완료")
        
    U = transform_ev2normal(F_tr)
    
    # 최대값이 0이 되도록 정규화
    U -= np.max(U)
    utils.print_elapsed("       [fft_solve DCT-II] ev2normal (IDCT-II) 역변환 및 U 정규화 완료")
    
    return U

def residual_pde(U: np.ndarray, F: np.ndarray) -> float:
    """내부 포인트들에 대한 잔차(오차)의 L2 norm을 계산합니다."""
    laplace = (-4.0 * U[1:-1, 1:-1] + 
               U[:-2, 1:-1] + U[2:, 1:-1] + 
               U[1:-1, :-2] + U[1:-1, 2:])
               
    res = np.sum((laplace - F[1:-1, 1:-1])**2)
    return float(np.sqrt(res))
