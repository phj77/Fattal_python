import numpy as np
import pyfftw
import multiprocessing

# 반복적인 FFT 연산 수행 시 플랜 생성 오버헤드를 줄이기 위해 캐싱 활성화
pyfftw.interfaces.cache.enable()

def get_lambda(n: int) -> np.ndarray:
    """1D 노이만 경계조건 라플라스 연산자의 고유값 계산"""
    i = np.arange(n, dtype=np.float32)
    # 정밀도 유지를 위해 64비트 연산 후 32비트로 다운캐스팅
    val = -4.0 * np.sin(i * np.pi / (2.0 * (n - 1))) ** 2
    return val.astype(np.float32)

def make_compatible_boundary(F: np.ndarray) -> None:
    """방정식 해가 존재하도록 우변 행렬의 경계 조건을 수정 (In-place 연산)"""
    H, W = F.shape

    # 부동소수점 누적 오차를 최소화하기 위해 합산 과정에만 float64 사용
    interior_sum = np.sum(F[1:-1, 1:-1], dtype=np.float64)
    edges_x = np.sum(F[0, 1:-1], dtype=np.float64) + np.sum(F[-1, 1:-1], dtype=np.float64)
    edges_y = np.sum(F[1:-1, 0], dtype=np.float64) + np.sum(F[1:-1, -1], dtype=np.float64)
    corners = np.float64(F[0, 0] + F[0, -1] + F[-1, 0] + F[-1, -1])

    total_sum = interior_sum + 0.5 * edges_x + 0.5 * edges_y + 0.25 * corners
    add_val = np.float32(-total_sum / (H + W - 3))

    F[0, :] += add_val
    F[-1, :] += add_val
    F[1:-1, 0] += add_val
    F[1:-1, -1] += add_val

def transform_normal2ev(A: np.ndarray, threads: int) -> np.ndarray:
    """일반 공간에서 고유벡터 공간으로의 변환 (FFTW_REDFT00)"""
    H, W = A.shape

    # FFTW Type-1 2D DCT 수행
    T = pyfftw.interfaces.scipy_fft.dctn(A, type=1, workers=threads).astype(np.float32)

    # 전체 배열 스케일링
    scale = np.float32(1.0 / ((H - 1) * (W - 1)))
    T *= scale

    # C++ 로직에 따른 경계 스케일링 
    # (행/열 슬라이싱이 교차하는 모서리는 0.5가 두 번 곱해져 자동 0.25 처리됨)
    T[:, 0] *= 0.5
    T[:, -1] *= 0.5
    T[0, :] *= 0.5
    T[-1, :] *= 0.5

    return T

def transform_ev2normal(A: np.ndarray, threads: int) -> np.ndarray:
    """고유벡터 공간에서 일반 공간으로의 역변환 (FFTW_REDFT00)"""
    # 입력 배열 원본 보존
    A_scaled = A.copy()

    # 내부 스케일링
    A_scaled[1:-1, 1:-1] *= 0.25

    # 엣지 스케일링 (C++ 루프 구조에 맞춰 모서리는 스케일링에서 제외)
    A_scaled[0, 1:-1] *= 0.5
    A_scaled[-1, 1:-1] *= 0.5
    A_scaled[1:-1, 0] *= 0.5
    A_scaled[1:-1, -1] *= 0.5

    # FFTW Type-1 2D DCT 수행
    T = pyfftw.interfaces.scipy_fft.dctn(A_scaled, type=1, workers=threads).astype(np.float32)
    return T

def solve_pde_fftw(F: np.ndarray, adjust_bound: bool = True) -> np.ndarray:
    """FFTW를 활용한 2D 푸아송 방정식 직접 해법"""
    H, W = F.shape
    threads = multiprocessing.cpu_count()

    # SIMD(SSE/AVX) 가속 최적화를 위해 16/32바이트 정렬된 메모리에 배열 복사
    F_aligned = pyfftw.empty_aligned((H, W), dtype=np.float32)
    F_aligned[:] = F.astype(np.float32)

    if adjust_bound:
        make_compatible_boundary(F_aligned)

    # 1. 고유벡터 공간으로의 직교 변환 (F_tr = EVy^-1 * F * (EVx^-1)^tr)
    F_tr = transform_normal2ev(F_aligned, threads)

    # 2. 고유값을 이용한 해 도출 (Broadcasting 활용)
    ly = get_lambda(H)
    lx = get_lambda(W)
    denom = ly[:, np.newaxis] + lx[np.newaxis, :]

    denom[0, 0] = 1.0  # ZeroDivision 방지 
    F_tr /= denom
    F_tr[0, 0] = 0.0   # 상수에 해당하는 해의 DC성분을 0으로 초기화

    # 3. 고유벡터 공간에서 일반 공간으로 역변환 (U = EVy * F_tr * EVx^tr)
    U = transform_ev2normal(F_tr, threads)

    # 4. 행렬 내 양수 성분이 없도록 정규화
    U -= np.max(U)

    return U

def residual_pde(U: np.ndarray, F: np.ndarray) -> float:
    """해의 잔차(Residual) 오차 계산 (C++ 검증용 함수와 동일)"""
    # 벡터화된 2D 라플라스 연산
    laplace = (-4.0 * U[1:-1, 1:-1] + 
               U[:-2, 1:-1] + U[2:, 1:-1] + 
               U[1:-1, :-2] + U[1:-1, 2:])
    
    res = np.sum((laplace - F[1:-1, 1:-1]) ** 2)
    return float(np.sqrt(res))