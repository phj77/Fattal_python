# config.py - Base/Detail Separation + Fattal 실험 전용 설정
import itertools
import numpy as np
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parents[4]# Fattal_python

data_path = project_root / "data" 
experiment_result_path = project_root / "experiment_result" / "base_detail_seperate"

# 입출력 디렉토리 설정
INPUT_DIR = str(data_path / "hard_case"/ "2")
OUTPUT_DIR = str(experiment_result_path /"hard_case"/"2")


# ─── 이미지 크롭(자르기) 범위 설정 ──────────────────────────────────────────
# 직사각형 크롭 범위: (min_pixel, max_pixel)
# None으로 설정하면 크롭하지 않고 전체 이미지를 사용합니다.
CROP_Y_RANGE = (182, 1884)  # 세로 범위 (Y축), 예: (201, 1833); dataset3
CROP_X_RANGE = (110, 3072)  # 가로 범위 (X축), 예: (311, 2982)
CROP_Y_RANGE = (256, 334)  # 세로 범위 (Y축) # dataset2
CROP_X_RANGE = (170, 3072)  # 가로 범위 (X축) # dataset2
CROP_Y_RANGE = None
CROP_X_RANGE = None
# CROP_Y_RANGE = (182, 1884)  # 세로 범위 (Y축) # dataset2
# CROP_X_RANGE = (110, 3072)  # 가로 범위 (X축) # dataset2
# ─────────────────────────────────────────────────────────────────────────────

# ─── Guided Filter 파라미터 ─────────────────────────────────────────────────
# gf_radius : Guided Filter의 윈도우 반지름 (정수)
# gf_eps    : Guided Filter의 regularization parameter (ε)
# detail_factor : detail layer에 곱할 계수 (합성 시 사용)
# ─────────────────────────────────────────────────────────────────────────────

# ─── Fattal TMO 파라미터 ────────────────────────────────────────────────────
# 기존 Fattal 파라미터와 동일
# ─────────────────────────────────────────────────────────────────────────────

alpha_range = np.round(np.arange(0.1, 1, 0.05), 2).tolist()

beta_range = np.round(np.arange(0.1, 1.00, 0.03), 2).tolist()

alpha_range_v2 = np.round(np.arange(0.2, 1, 0.2), 2).tolist()

beta_range_v2 = np.round(np.arange(0.2, 1.00, 0.1), 2).tolist()

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    # Fattal 파라미터
    'opt_alpha': [0.8],
    'opt_beta': [0.8],
    'opt_noise': [0.001],
    'newfattal': [True],
    'fftsolver': [True],
    'detail_level': [0],
    'hpf_sigma': [0.007],
    # Guided Filter 파라미터
    'gf_radius': [10],
    'gf_eps': [0.005], #[0.001, 0.005, 0.01, 0.05]
    # Detail 합성 파라미터
    'detail_factor': [7],
}

def get_parameter_combinations():
    """
    PARAM_GRID에 정의된 파라미터들의 모든 조합을 딕셔너리 리스트 형태로 반환합니다.
    """
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    # 모든 파라미터 리스트의 데카르트 곱(Cartesian product)을 구합니다.
    combinations = list(itertools.product(*values))
    
    # 생성된 조합을 다시 딕셔너리 형태로 매핑하여 반환합니다.
    return [dict(zip(keys, combo)) for combo in combinations]
