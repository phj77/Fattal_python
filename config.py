# config.py
import itertools
import numpy as np

# 입출력 디렉토리 설정
INPUT_DIR = './test/input'
OUTPUT_DIR = './test/output/single'

# ─── 파라미터 자동 생성 설정 ───────────────────────────────────────────
# np.arange(시작, 끝(포함X), 간격)
# np.round(배열, 소수점_자릿수) : 파일명 오차 방지

# 예: 0.1부터 0.5까지 0.2 간격 -> [0.1, 0.3, 0.5]
alpha_range = np.round(np.arange(0.1, 0.9001, 0.2), 2).tolist()

# 예: 0.85부터 0.98까지 0.05 간격 -> [0.85, 0.9, 0.95]
beta_range = np.round(np.arange(0.73, 1.00, 0.02), 2).tolist()

# 예: 0.0, 0.5, 1.0 -> [0.0, 0.5, 1.0]
he_range = np.round(np.arange(0.0, 0.2, 0.03), 2).tolist()
# ───────────────────────────────────────────────────────────────────

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    'opt_alpha': alpha_range,
    'opt_beta': beta_range,
    'opt_noise': [0.001],
    'newfattal': [True],
    'fftsolver': [False],
    'detail_level': [0],
    'HE_weight': he_range,
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