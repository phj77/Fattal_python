# config.py
import itertools
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[4] # src/experiment/scaling_factor_modified_monotonic/config/config.py -> 4 levels up
data_path = project_root / "data" 
test_path = project_root / "test"

# 입출력 디렉토리 설정
INPUT_DIR = str(data_path / "data_one" / "3")
OUTPUT_DIR = str(test_path / "scaling_factor_modified_monotonic" / "tmp_noncropped")

# ─── 실험 전용 사전 HPF (Pre-HPF) 및 크롭 범위 설정 ───────────────────────
CROP_Y_RANGE = (201, 1833)  # 세로 범위 (Y축)
CROP_X_RANGE = (311, 2982)  # 가로 범위 (X축)
CROP_Y_RANGE = None  # 세로 범위 (Y축)
CROP_X_RANGE = None  # 가로 범위 (X축)
# ─────────────────────────────────────────────────────────────────────────────

# ─── 파라미터 자동 생성 설정 ────────────────────────────────────────
alpha_range = np.round(np.arange(0.1, 1, 0.05), 2).tolist()
beta_range = np.round(np.arange(0.1, 1.00, 0.03), 2).tolist()

alpha_range_v2 = np.round(np.arange(0.3, 1, 0.3), 2).tolist()
beta_range_v2 = np.round(np.arange(0.1, 0.5, 0.1), 2).tolist()
# ───────────────────────────────────────────────────────────────────

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    'opt_alpha': alpha_range_v2,
    'opt_beta': beta_range_v2,
    'opt_noise': [0.001],
    'newfattal': [True],
    'fftsolver': [True],
    'detail_level': [0],
    'hpf_sigma': [0.007],
    'pre_hpf_sigma': [0.008],
    'xp_ratio': [0.1, 0.2, 0.3,0.4,0.5],
    'y0': [2.1, 3.0, 4.0],  # y0 > 2.0 (must)
}

def get_parameter_combinations():
    """
    PARAM_GRID에 정의된 파라미터들의 모든 조합을 딕셔너리 리스트 형태로 반환합니다.
    """
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]
