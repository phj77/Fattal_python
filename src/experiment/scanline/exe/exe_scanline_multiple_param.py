# exe_scanline_multiple_param.py
# config.py에 정의된 여러 파라미터 그리드 조합(alpha, beta 등)에 대해 개별 데이터셋의 스캔라인 프로파일 및 최종 톤 매핑 결과를 비교 분석하기 위해 폴더별로 자동 생성 및 저장해주는 다중 파라미터 배치 실행 스크립트입니다.
import cv2
import numpy as np
import os
import glob
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Setup paths to import modules from 'src'
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
project_root = current_file.parents[4]  # .../Fattal_python root
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from experiment.scanline.fattal.fattal_tmo import pfstmo_fattal02
from exe.config.config import INPUT_DIR, OUTPUT_DIR, get_parameter_combinations
import utils.utils as utils

# Dataset configs mapping scanline row and highlight ranges for default datasets (1~7)
dataset_configs = {
    1: {"row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
    2: {"row": 1661, "highlight": [[300, 530], [1868, 1965]]},
    3: {"row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
    4: {"row": 974,  "highlight": [[457, 475], [590, 607]]},
    5: {"row": 1170, "highlight": [[2073, 2188]]},
    6: {"row": 1590, "highlight": [[400, 620], [2095, 2190]]},
    7: {"row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
}

def validate_and_get_dataset_dirs(input_dir):
    """
    INPUT_DIR 및 하위 디렉토리를 검증하고, dataset_configs에 등록된 숫자 폴더 목록을 반환합니다.
    단일 숫자 폴더(예: data_one/3) 또는 상위 폴더(예: data_one) 모두 지원하며,
    dataset_configs에 없는 숫자/폴더는 예외를 발생시키고 종료합니다.
    """
    input_dir_abs = os.path.abspath(input_dir)
    norm_input_dir = os.path.normpath(input_dir_abs)
    dir_name = os.path.basename(norm_input_dir)

    valid_keys = sorted(list(dataset_configs.keys()))

    if not os.path.exists(norm_input_dir) or not os.path.isdir(norm_input_dir):
        print(f"[오류 예외 발생] 지정한 INPUT_DIR('{input_dir}') 경로가 존재하지 않거나 디렉토리가 아닙니다.")
        sys.exit(1)

    # Case 1: INPUT_DIR 자체가 단일 숫자 폴더인 경우 (예: data_one/3)
    if dir_name.isdigit():
        num_key = int(dir_name)
        if num_key in dataset_configs:
            hdr_files = glob.glob(os.path.join(norm_input_dir, '*.hdr'))
            if hdr_files:
                return [norm_input_dir]
            else:
                print(f"[오류 예외 발생] 지정된 숫자 폴더('{dir_name}') 내에 .hdr 파일이 존재하지 않습니다.")
                sys.exit(1)
        else:
            print(f"[오류 예외 발생] INPUT_DIR이 숫자 폴더('{dir_name}')이지만 dataset_configs{valid_keys}에 등록되어 있지 않습니다.")
            sys.exit(1)

    # Case 2: 상위 폴더인 경우 (예: data_one) 하위 디렉토리 탐색
    subdirs = [os.path.join(norm_input_dir, d) for d in os.listdir(norm_input_dir)
               if os.path.isdir(os.path.join(norm_input_dir, d))]

    valid_dataset_dirs = []
    invalid_subdirs = []

    for sd in subdirs:
        sub_name = os.path.basename(sd)
        if sub_name.isdigit():
            num_key = int(sub_name)
            if num_key in dataset_configs:
                hdr_files = glob.glob(os.path.join(sd, '*.hdr'))
                if hdr_files:
                    valid_dataset_dirs.append((num_key, sd))
                else:
                    print(f"[경고] 숫자 폴더 '{sub_name}' 내에 .hdr 파일이 존재하지 않아 스킵됩니다.")
            else:
                invalid_subdirs.append(sub_name)
        else:
            invalid_subdirs.append(sub_name)

    if not valid_dataset_dirs:
        print(f"[오류 예외 발생] INPUT_DIR('{input_dir}') 및 하위 폴더에서 dataset_configs에 등록된 숫자 폴더{valid_keys}를 찾지 못했습니다.")
        if invalid_subdirs:
            print(f"       감지된 하위 폴더: {invalid_subdirs}")
        sys.exit(1)

    # 숫자 키 기준 오름차순 정렬
    valid_dataset_dirs.sort(key=lambda x: x[0])
    return [path for key, path in valid_dataset_dirs]

def get_scanline_config(input_dir_path, img_shape=None):
    """Determine scanline row and highlight ranges based on input directory name."""
    dir_name = os.path.basename(os.path.normpath(input_dir_path))
    if dir_name.isdigit() and int(dir_name) in dataset_configs:
        cfg = dataset_configs[int(dir_name)]
        return cfg["row"], cfg["highlight"]
    
    raise ValueError(f"dataset_configs에 정의되지 않은 폴더입니다: {dir_name}")

def main():
    utils.start_timer()
    utils.print_elapsed("다중 파라미터 스캔라인 출력 작업 시작")

    # INPUT_DIR 검증 및 dataset_configs에 부합하는 하위 숫자 디렉토리 추출
    dataset_dirs = validate_and_get_dataset_dirs(INPUT_DIR)

    # 파라미터 조합 로드
    param_combinations = get_parameter_combinations()

    # config.py에서 설정한 OUTPUT_DIR을 사용
    base_output_dir = OUTPUT_DIR
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    print(f"입력 디렉토리: {INPUT_DIR}")
    print(f"출력 디렉토리: {base_output_dir}")
    print(f"감지된 데이터셋 디렉토리 수: {len(dataset_dirs)}, 파라미터 조합 수: {len(param_combinations)}\n")

    for d_idx, ds_dir in enumerate(dataset_dirs, 1):
        dataset_name = os.path.basename(os.path.normpath(ds_dir))
        hdr_files = glob.glob(os.path.join(ds_dir, '*.hdr'))

        # 하위 데이터셋 폴더별 출력 저장 경로 설정
        if len(dataset_dirs) > 1 or dataset_name.isdigit():
            dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        else:
            dataset_output_dir = base_output_dir

        total_tasks = len(hdr_files) * len(param_combinations)
        print("=" * 60)
        print(f"[{d_idx}/{len(dataset_dirs)}] 데이터셋 처리 중: {dataset_name} ({ds_dir})")
        print(f"감지된 이미지 수: {len(hdr_files)}, 파라미터 조합 수: {len(param_combinations)}")
        print(f"총 {total_tasks}회의 스캔라인 생성 작업 진행 예정")
        print("=" * 60)

        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\n--- 이미지 로드 중: {file_name} ---")

            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"[오류] 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            # 같은 intensity를 가진 3채널 이미지에서 1채널만 추출하여 사용
            if img.ndim == 3:
                img_single = img[:, :, 0]
            else:
                img_single = img

            scanline_row, highlight_ranges = get_scanline_config(ds_dir, img.shape)
            print(f"스캔라인 설정 - Row: {scanline_row}, Highlight: {highlight_ranges}")

            for idx, p in enumerate(param_combinations, 1):
                opt_alpha = p['opt_alpha']
                opt_beta = p['opt_beta']
                opt_noise = p.get('opt_noise', 0.001)
                newfattal = p.get('newfattal', True)
                fftsolver = p.get('fftsolver', True)
                detail_level = p.get('detail_level', 0)

                # 각 파라미터 조합별로 독립적인 폴더 생성
                param_folder_name = f"a{opt_alpha}_b{opt_beta}_dl{detail_level}"
                if len(hdr_files) > 1:
                    param_save_dir = os.path.join(dataset_output_dir, param_folder_name, file_name)
                else:
                    param_save_dir = os.path.join(dataset_output_dir, param_folder_name)

                if not os.path.exists(param_save_dir):
                    os.makedirs(param_save_dir)

                print(f"  [{idx}/{len(param_combinations)}] 파라미터 스캔라인 생성 중: {param_folder_name}")

                # Tone Mapping & Scanline 출력 (save_dir로 지정된 각 파라미터 폴더에 스캔라인 이미지 저장)
                L_out = pfstmo_fattal02(
                    img_single,
                    opt_alpha, opt_beta, opt_noise,
                    newfattal, fftsolver, detail_level,
                    scanline_row=scanline_row, highlight_ranges=highlight_ranges,
                    save_dir=param_save_dir
                )

                # 포맷 변환 및 클리핑 (8bit 단일 채널 이미지)
                out_img = np.clip(L_out, 0.0, 1.0)
                out_img_8bit = (out_img * 255.0).astype(np.uint8)

                save_img_name = f"{file_name}_{param_folder_name}_result.png"
                cv2.imwrite(os.path.join(param_save_dir, save_img_name), out_img_8bit)

    utils.print_elapsed("다중 파라미터 스캔라인 출력 작업 완료")

if __name__ == "__main__":
    main()
