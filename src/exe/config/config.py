# config.py
import itertools
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
data_path = project_root / "data" 
test_path = project_root / "test"

# 입출력 디렉토리 설정
INPUT_DIR = str(data_path / "data_one"/"3")
#OUTPUT_DIR = str(test_path/"scaling_factor_visualization"/"detail_level_0")
# OUTPUT_DIR = str(test_path/"scanline"/"vertical")
OUTPUT_DIR = str(test_path/"tmp"/"scanline")

# ─── 파라미터 자동 생성 설정 ────────────────────────────────────────
# np.arange(시작, 끝(포함X), 간격)
# np.round(배열, 소수점_자릿수) : 파일명 오차 방지

alpha_range = np.round(np.arange(0.1, 1, 0.05), 2).tolist()

beta_range = np.round(np.arange(0.1, 1.00, 0.03), 2).tolist()

alpha_range_v2 = np.round(np.arange(0.1, 1, 0.1), 2).tolist()

beta_range_v2 = np.round(np.arange(0.1, 1.00, 0.1), 2).tolist()
# ───────────────────────────────────────────────────────────────────

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    'opt_alpha': [0.9],
    'opt_beta': [0.9],
    'opt_noise': [0.001],
    'newfattal': [True],
    'fftsolver': [True],
    'detail_level': [0],
    'hpf_sigma': [0.007],
    'pre_gamma': [1.0],
    'post_gamma': [1.0]
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