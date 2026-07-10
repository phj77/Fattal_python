# gf_fattal_tmo.py - Guided Filter + Fattal Base Layer용 log domain TMO 함수
# 주의: 이 모듈은 exe 스크립트에서 src/ 경로가 sys.path에 추가된 후에 import됩니다.
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fattal.fattal_tmo import (
    createGaussianPyramids,
    calculate_level_scaling_factor,
    calculate_attenuation,
)
from fattal import pde_fft, pde_multigrid
from utils import utils


def tmo_fattal02_logdomain(H, alfa, beta, noise, newfattal, fftsolver, detail_level, hpf_sigma=0.007):
    """
    이미 log domain으로 변환된 H를 입력받아 Fattal TMO 핵심 연산을 수행합니다.
    - 내부 로그 변환(np.log)을 건너뜁니다.
    - 지수 복원(np.exp) 및 정규화를 수행하지 않습니다.
    - PDE 풀이 결과 U (log domain)를 그대로 반환합니다.
    
    Args:
        H: log domain 이미지 (예: np.log(100 * Y / maxLum + 1e-4))
        나머지 파라미터: tmo_fattal02와 동일
    
    Returns:
        U: PDE 풀이 결과 (log domain)
    """
    utils.print_elapsed("     [tmo_logdomain] 시작")
    h, w = H.shape

    TOP_SIZE = 2**3 if fftsolver else 32

    # 가우시안 피라미드 구성
    mins = min(w, h)
    n_pyramid_levels = 0
    temp_mins = mins
    while temp_mins >= TOP_SIZE:
        n_pyramid_levels += 1
        temp_mins //= 2
    if n_pyramid_levels == 0: n_pyramid_levels = 1

    pyramids = createGaussianPyramids(H, n_pyramid_levels)
    utils.print_elapsed("     [tmo_logdomain] 가우시안 피라미드 구성 완료")

    # value 행렬 병렬 계산
    scaling_factor = [None] * n_pyramid_levels
    with ThreadPoolExecutor(max_workers=n_pyramid_levels) as executor:
        futures = []
        for k in range(n_pyramid_levels):
            if k >= detail_level or k == n_pyramid_levels - 1 or not newfattal:
                futures.append((k, executor.submit(calculate_level_scaling_factor, pyramids[k], k, alfa, beta, noise)))
            else:
                scaling_factor[k] = None
                
        for k, future in futures:
            scaling_factor[k] = future.result()

    # FI 행렬 계산
    attenuation_map = calculate_attenuation(scaling_factor, pyramids, n_pyramid_levels, newfattal)
    utils.print_elapsed("     [tmo_logdomain] FI 행렬 및 그래디언트 계산 완료")

    # 기울기 감쇠
    if fftsolver:
        Gx = np.empty_like(H)
        Gx[:, :-1] = (H[:, 1:] - H[:, :-1]) * 0.5 * (attenuation_map[:, 1:] + attenuation_map[:, :-1])
        Gx[:, -1] = (H[:, -2] - H[:, -1]) * 0.5 * (attenuation_map[:, -2] + attenuation_map[:, -1])
        
        Gy = np.empty_like(H)
        Gy[:-1, :] = (H[1:, :] - H[:-1, :]) * 0.5 * (attenuation_map[1:, :] + attenuation_map[:-1, :])
        Gy[-1, :] = (H[-2, :] - H[-1, :]) * 0.5 * (attenuation_map[-2, :] + attenuation_map[-1, :])
    else:
        e = np.minimum(np.arange(w) + 1, w - 1)
        s = np.minimum(np.arange(h) + 1, h - 1)
        Gx = (H[:, e] - H) * attenuation_map
        Gy = (H[s, :] - H) * attenuation_map

    # 다이버전스(발산) 계산
    DivG = Gx + Gy
    DivG[:, 1:] -= Gx[:, :-1]
    DivG[1:, :] -= Gy[:-1, :]

    if fftsolver:
        DivG[:, 0] += Gx[:, 0]
        DivG[0, :] += Gy[0, :]
    utils.print_elapsed("     [tmo_logdomain] 다이버전스(발산) 계산 완료")

    # PDE 풀이
    utils.print_elapsed("     [tmo_logdomain] PDE 풀이 시작")
    if fftsolver:
        U = pde_fft.solve_pde_fft(DivG, hpf_sigma=hpf_sigma)
    else:
        U = np.zeros_like(DivG)
        U = pde_multigrid.solve_pde_multigrid(DivG, U)
    utils.print_elapsed("     [tmo_logdomain] PDE 풀이 완료")

    return U
