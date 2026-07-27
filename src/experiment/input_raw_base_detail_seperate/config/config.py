# config.py - Input RAW Base/Detail Separation + Fattal 실험 전용 설정
import itertools
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()

# Find project root dynamically by climbing up parents
project_root = current_file.parents[6] # poor_battery_enhancement
result_root = current_file.parents[4] # Fattal_python

data_path = project_root / "data" / "raw_image"
experiment_result_path = result_root / "experiment_result" / "input_raw_base_detail_seperate"

# 입출력 디렉토리 설정
INPUT_DIR = str(data_path / "tmp_hard_case"/"20260706135050579_F120260706135052_3072x2048_NG"/"01_3072 x 2048_130kV_0.3mA_F_pos(1)_NG.raw")
OUTPUT_DIR = str(experiment_result_path /"tmp_hard_case"/"a0.8b0.8hpf_0.007ghr10gfe0.005df7"/"20260706135050579_F120260706135052_3072x2048_NG"/"tmp_2")

# 입출력 디렉토리 설정
# INPUT_DIR = r"C:\Users\Park_HyoungJun\LAB\mission\poor_battery_enhancement\data\Full_dataset\0722_train_raw"
# OUTPUT_DIR = r"C:\Users\Park_HyoungJun\LAB\mission\poor_battery_enhancement\src\Fattal_python\0722_output"

# ─── 이미지 크롭(자르기) 범위 설정 ──────────────────────────────────────────
# 직사각형 크롭 범위: (min_pixel, max_pixel)
# None으로 설정하면 크롭하지 않고 전체 이미지를 사용합니다.
CROP_Y_RANGE = [224,1818]
CROP_X_RANGE = [240,372]
CROP_Y_RANGE = None
CROP_X_RANGE = None
# ─────────────────────────────────────────────────────────────────────────────

# ─── TOP_SIZE 설정 ───────────────────────────────────────────────────
PYRAMID_TOP_SIZE = 2**3

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
    'gf_eps': [0.005],
    # Detail 합성 파라미터
    'detail_factor': [7],
    'pyramid_top_size': [PYRAMID_TOP_SIZE]
}

def get_parameter_combinations():
    """
    PARAM_GRID에 정의된 파라미터들의 모든 조합을 딕셔너리 리스트 형태로 반환합니다.
    """
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]
