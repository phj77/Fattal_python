import numpy as np
import time

class Array2Df:
    """pfs::Array2Df 객체의 메모리 구조를 모사하는 클래스"""
    def __init__(self, data):
        # 원본 C++ 코드와 동일하게 32비트 부동소수점(float32)으로 내부 데이터를 관리합니다.
        self.data = np.array(data, dtype=np.float32)

    def get_rows(self) -> int:
        return self.data.shape[0]

    def get_cols(self) -> int:
        return self.data.shape[1]

class Frame:
    """pfs::Frame 객체를 모사하는 클래스"""
    def __init__(self, x_data, y_data, z_data):
        self.x_channel = Array2Df(x_data)
        self.y_channel = Array2Df(y_data)
        self.z_channel = Array2Df(z_data)

    def get_xyz_channels(self):
        return self.x_channel, self.y_channel, self.z_channel

def apply_gamma_array(array: Array2Df, exponent: float, timer_profiling: bool = False) -> None:
    """pfs::applyGamma(pfs::Array2Df *array, const float exponent) 함수를 완벽히 모사합니다."""
    if timer_profiling:
        start_time = time.perf_counter()

    data = array.data

    # 1. 값이 0.0 초과인지 확인하는 불리언(Boolean) 마스크 생성
    # 이는 C++의 if ((*array)(j, i) > 0.0f) 및 vmaskf_gt 벡터 마스킹과 동일한 역할을 합니다.
    mask = data > 0.0

    # 2. 마스크에 해당하는 (0보다 큰) 요소에만 지수 연산을 수행하여 원본 배열에 덮어씁니다.
    # NumPy의 power 연산은 내부적으로 C 기반의 SIMD 연산을 수행하므로 속도가 빠릅니다.
    data[mask] = np.power(data[mask], exponent)

    # 3. 마스크에 해당하지 않는 (0 이하인) 요소는 모두 0.0으로 강제 설정합니다.
    # 이는 C++의 else { (*array)(j, i) = 0.0f; } 분기를 처리합니다.
    data[~mask] = 0.0

    if timer_profiling:
        end_time = time.perf_counter()
        elapsed_msec = (end_time - start_time) * 1000.0
        print(f"applyGamma() = {elapsed_msec:.3f} msec")

def apply_gamma_frame(frame: Frame, gamma: float) -> None:
    """pfs::applyGamma(pfs::Frame *frame, float gamma) 함수를 완벽히 모사합니다."""
    # C++의 조기 반환 조건과 일치합니다.
    if gamma == 1.0:
        return

    x, y, z = frame.get_xyz_channels()
    
    # 지수 역수 계산
    exponent = 1.0 / gamma

    # 각 채널별로 배열 전용 gamma 연산을 적용합니다.
    apply_gamma_array(x, exponent)
    apply_gamma_array(y, exponent)
    apply_gamma_array(z, exponent)