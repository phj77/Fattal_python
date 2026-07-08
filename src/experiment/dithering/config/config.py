# config.py
import itertools
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[4] # src/experiment/dithering/config/config.py -> 4 levels up to project root
data_path = project_root / "data" 
test_path = project_root / "test"

# 입출력 디렉토리 설정
INPUT_DIR = str(data_path / "data_one"/"3")
OUTPUT_DIR = str(test_path / "dithering")

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    'opt_alpha': [0.5],
    'opt_beta': [0.9],
    'opt_noise': [0.001],
    'newfattal': [True],
    'fftsolver': [True],
    'detail_level': [0],
    'hpf_sigma': [0.007],
    'pre_hpf_sigma': [0.010],
    'dither_strength': [0.01] # dithering parameter in log domain
}

def get_parameter_combinations():
    """
    PARAM_GRID에 정의된 파라미터들의 모든 조합을 딕셔너리 리스트 형태로 반환합니다.
    """
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]
