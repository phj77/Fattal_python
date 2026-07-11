# exe_multiple_param.py
import cv2
import numpy as np
import os
import glob
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 사용자 정의 모듈 (환경에 맞게 존재해야 함)
from experiment.scaling_factor_modified.fattal.fattal_tmo import pfstmo_fattal02

# 파라미터 및 설정 불러오기
from experiment.scaling_factor_modified.config.config import (
    INPUT_DIR, OUTPUT_DIR, get_parameter_combinations,
    CROP_Y_RANGE, CROP_X_RANGE
)

import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("시작")
    
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 입력 폴더 내의 모든 hdr 파일 경로를 탐색합니다.
    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
        return

    # 파라미터 조합을 가져옵니다.
    param_combinations = get_parameter_combinations()
    total_tasks = len(hdr_files) * len(param_combinations)

    utils.print_elapsed("구간 1 (환경 설정 및 파일 탐색 완료)")
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    print(f"총 {total_tasks}회의 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 이미지에 대하여 반복 실행
    for img_path in hdr_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 이미지 로드
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if img is None:
            print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
            continue

        # 같은 intensity를 가진 3채널 이미지에서 1채널만 추출하여 사용
        if img.ndim == 3:
            img_single = img[:, :, 0]
        else:
            img_single = img

        # 이미지 크롭 적용
        if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
            h, w = img_single.shape
            ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h)
            xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w)
            # 안전성 경계값 클리핑
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            img_single = img_single[ymin:ymax, xmin:xmax]
            utils.print_elapsed(f"구간 2.5 (이미지 크롭 완료: Y[{ymin}:{ymax}], X[{xmin}:{xmax}])")
        else:
            utils.print_elapsed(f"구간 2 (이미지 로드 완료) - 대상: {file_name}")

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for p in param_combinations:
            # 톤 매핑 연산 (감마 보정 없이 1채널 이미지를 pfstmo_fattal02에 전달)
            L_out = pfstmo_fattal02(
                img_single,
                p['opt_alpha'], p['opt_beta'], p['opt_noise'],
                p['newfattal'], p['fftsolver'], p['detail_level'],
                hpf_sigma=p.get('hpf_sigma', 0.007),
                pre_hpf_sigma=p['pre_hpf_sigma'],
                v0=p.get('v0', 0.01),
                xp_ratio=p.get('xp_ratio', 0.1),
                vp=p.get('vp', 6.0),
                pw=p.get('w', 1)
            )

            crop_suffix = ""
            if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
                h_orig, w_orig = img.shape[:2]
                ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h_orig)
                xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w_orig)
                ymin, ymax = max(0, ymin), min(h_orig, ymax)
                xmin, xmax = max(0, xmin), min(w_orig, xmax)
                crop_suffix = f"_cropY{ymin}-{ymax}_X{xmin}-{xmax}"

            param_suffix = f"preHpf{p['pre_hpf_sigma']}_a{p['opt_alpha']}_b{p['opt_beta']}_n{p['opt_noise']}_dl{p['detail_level']}_v0{p.get('v0', 0.01)}_xp{p.get('xp_ratio', 0.1)}_vp{p.get('vp', 6.0)}_w{p.get('w', 1)}{crop_suffix}"
            utils.print_elapsed(f"구간 3 (톤 매핑 연산 완료) - 파라미터: {param_suffix}")

            # 포맷 변환 및 클리핑 (8bit 단일 채널 이미지)
            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)

            utils.print_elapsed("구간 4 (후처리 완료)")

            # 식별 가능한 파일명 생성 및 저장
            save_name = f"{file_name}_{param_suffix}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, out_img_8bit)
            print(f"완료: {save_path}")
            utils.print_elapsed("구간 5 (파일 저장 완료)")
    
    utils.print_elapsed("프로그램 전체 종료")

if __name__ == "__main__":
    main()
