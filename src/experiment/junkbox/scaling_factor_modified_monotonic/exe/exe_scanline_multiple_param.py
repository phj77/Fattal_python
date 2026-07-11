# exe_scanline_multiple_param.py
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

current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# scaling_factor_modified_monotonic 모듈 가져오기
from experiment.scaling_factor_modified_monotonic.fattal.fattal_tmo import pfstmo_fattal02
from experiment.scaling_factor_modified_monotonic.config.config import (
    INPUT_DIR, OUTPUT_DIR, get_parameter_combinations,
    CROP_Y_RANGE, CROP_X_RANGE
)
import utils.utils as utils

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
    input_dir_abs = os.path.abspath(input_dir)
    norm_input_dir = os.path.normpath(input_dir_abs)

    if not os.path.exists(norm_input_dir) or not os.path.isdir(norm_input_dir):
        print(f"[오류] 지정한 INPUT_DIR('{input_dir}') 경로가 존재하지 않거나 디렉토리가 아닙니다.")
        sys.exit(1)

    direct_hdr = glob.glob(os.path.join(norm_input_dir, '*.hdr'))
    if direct_hdr:
        return [norm_input_dir]

    subdirs = [os.path.join(norm_input_dir, d) for d in os.listdir(norm_input_dir)
               if os.path.isdir(os.path.join(norm_input_dir, d))]

    valid_dataset_dirs = []
    for sd in subdirs:
        sub_hdr = glob.glob(os.path.join(sd, '*.hdr'))
        if sub_hdr:
            sub_name = os.path.basename(sd)
            key = int(sub_name) if sub_name.isdigit() else 9999
            valid_dataset_dirs.append((key, sd))

    if not valid_dataset_dirs:
        print(f"[오류] INPUT_DIR('{input_dir}') 및 하위 폴더에서 .hdr 파일을 찾지 못했습니다.")
        sys.exit(1)

    valid_dataset_dirs.sort(key=lambda x: x[0])
    return [path for key, path in valid_dataset_dirs]

def get_scanline_config(input_dir_path, img_shape):
    dir_name = os.path.basename(os.path.normpath(input_dir_path))
    if dir_name.isdigit() and int(dir_name) in dataset_configs:
        cfg = dataset_configs[int(dir_name)]
        return cfg["row"], cfg["highlight"]
    return img_shape[0] // 2, None

def main():
    utils.start_timer()
    utils.print_elapsed("scaling_factor_modified_monotonic + Fattal 다중 파라미터 스캔라인 출력 작업 시작")

    dataset_dirs = validate_and_get_dataset_dirs(INPUT_DIR)
    param_combinations = get_parameter_combinations()

    base_output_dir = OUTPUT_DIR
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    print(f"입력 디렉토리: {INPUT_DIR}")
    print(f"출력 디렉토리: {base_output_dir}")
    print(f"감지된 데이터셋 디렉토리 수: {len(dataset_dirs)}, 파라미터 조합 수: {len(param_combinations)}\n")

    for d_idx, ds_dir in enumerate(dataset_dirs, 1):
        dataset_name = os.path.basename(os.path.normpath(ds_dir))
        hdr_files = glob.glob(os.path.join(ds_dir, '*.hdr'))

        if len(dataset_dirs) > 1 or dataset_name.isdigit():
            dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        else:
            dataset_output_dir = base_output_dir

        total_tasks = len(hdr_files) * len(param_combinations)
        print("=" * 60)
        print(f"[{d_idx}/{len(dataset_dirs)}] 데이터셋 처리 중: {dataset_name} ({ds_dir})")
        print(f"감지된 이미지 수: {len(hdr_files)}, 파라미터 조합 수: {len(param_combinations)}")
        print(f"총 {total_tasks}회의 scaling_factor_modified_monotonic 스캔라인 생성 작업 진행 예정")
        print("=" * 60)

        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\n--- 이미지 로드 중: {file_name} ---")

            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"[오류] 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            if img.ndim == 3:
                img_single = img[:, :, 0]
            else:
                img_single = img

            scanline_row, highlight_ranges = get_scanline_config(ds_dir, img.shape)
            print(f"스캔라인 설정 (원본) - Row: {scanline_row}, Highlight: {highlight_ranges}")

            if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
                h_orig, w_orig = img_single.shape[:2]
                ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h_orig)
                xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w_orig)
                ymin, ymax = max(0, ymin), min(h_orig, ymax)
                xmin, xmax = max(0, xmin), min(w_orig, xmax)
                
                img_single = img_single[ymin:ymax, xmin:xmax]
                
                scanline_row = scanline_row - ymin
                if highlight_ranges is not None:
                    adjusted_highlights = []
                    for rng in highlight_ranges:
                        if len(rng) == 2:
                            r_start, r_end = rng
                            r_start_adj = max(0, r_start - xmin)
                            r_end_adj = min(xmax - xmin, r_end - xmin)
                            adjusted_highlights.append([r_start_adj, r_end_adj])
                    highlight_ranges = adjusted_highlights
                print(f"스캔라인 설정 (크롭 반영) - Row: {scanline_row}, Highlight: {highlight_ranges}")

            for idx, p in enumerate(param_combinations, 1):
                opt_alpha = p['opt_alpha']
                opt_beta = p['opt_beta']
                opt_noise = p.get('opt_noise', 0.001)
                newfattal = p.get('newfattal', True)
                fftsolver = p.get('fftsolver', True)
                detail_level = p.get('detail_level', 0)
                hpf_sigma = p.get('hpf_sigma', 0.007)
                pre_hpf_sigma = p.get('pre_hpf_sigma', 0.010)
                xp_ratio = p.get('xp_ratio', 0.05)
                y0 = p.get('y0', 6.0)

                crop_suffix = ""
                if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
                    crop_suffix = f"_cropY{CROP_Y_RANGE[0]}-{CROP_Y_RANGE[1]}_X{CROP_X_RANGE[0]}-{CROP_X_RANGE[1]}"

                param_folder_name = f"preHPF{pre_hpf_sigma}_a{opt_alpha}_b{opt_beta}_n{opt_noise}_dl{detail_level}_xpRatio{xp_ratio}_y0{y0}{crop_suffix}"
                if len(hdr_files) > 1:
                    param_save_dir = os.path.join(dataset_output_dir, param_folder_name, file_name)
                else:
                    param_save_dir = os.path.join(dataset_output_dir, param_folder_name)

                if not os.path.exists(param_save_dir):
                    os.makedirs(param_save_dir)

                print(f"  [{idx}/{len(param_combinations)}] 파라미터 스캔라인 생성 중: {param_folder_name}")

                L_out = pfstmo_fattal02(
                    img_single,
                    opt_alpha, opt_beta, opt_noise,
                    newfattal, fftsolver, detail_level,
                    scanline_row=scanline_row, highlight_ranges=highlight_ranges,
                    save_dir=param_save_dir,
                    hpf_sigma=hpf_sigma,
                    pre_hpf_sigma=pre_hpf_sigma,
                    xp_ratio=xp_ratio,
                    y0=y0
                )

                out_img = np.clip(L_out, 0.0, 1.0)
                out_img_8bit = (out_img * 255.0).astype(np.uint8)

                save_img_name = f"{file_name}_{param_folder_name}_result.png"
                cv2.imwrite(os.path.join(param_save_dir, save_img_name), out_img_8bit)

    utils.print_elapsed("scaling_factor_modified_monotonic 스캔라인 출력 작업 완료")

if __name__ == "__main__":
    main()
