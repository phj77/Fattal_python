# config_fusion.py
import itertools
import numpy as np

# 입출력 디렉토리 설정
INPUT_DIR = './test/input'
OUTPUT_DIR = './test/output/fusion'

# ─── 파라미터 자동 생성 설정 ───────────────────────────────────────────
alpha_range = np.round(np.arange(0.1, 0.9001, 0.2), 2).tolist()
he_range = np.round(np.arange(0.0, 0.2, 0.03), 2).tolist()

# 퓨전 모델의 구역별 beta 값 독립 생성
# 예: 영역 1 (좌측 하단) -> [0.85, 0.9, 0.95]
beta1_range = np.round(np.arange(0.73, 1.00, 0.02), 2).tolist()

# 예: 영역 2 (그 외) -> [0.93, 0.98]
beta2_range = np.round(np.arange(0.73, 1.00, 0.02), 2).tolist()

# 두 beta 범위를 조합하여 [[0.85, 0.93], [0.85, 0.98], ...] 형태의 쌍을 생성합니다.
#beta_pairs = [list(pair) for pair in itertools.product(beta1_range, beta2_range)]

# 데카르트 곱을 통해 모든 쌍을 생성한 뒤, b1 < b2 조건이 성립하는 쌍만 리스트로 반환합니다.
beta_pairs = [
    [b1, b2] for b1, b2 in itertools.product(beta1_range, beta2_range) if b1 < b2
]
# ───────────────────────────────────────────────────────────────────

# 실험할 파라미터 값들을 정의합니다.
PARAM_GRID = {
    'opt_alpha': alpha_range,
    'opt_betas': beta_pairs, # 생성된 리스트 쌍을 대입
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
    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]