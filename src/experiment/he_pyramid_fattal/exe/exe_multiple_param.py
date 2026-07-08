# exe_multiple_param.py
import cv2
import numpy as np
import os
import glob
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from experiment.he_pyramid_fattal.fattal.fattal_tmo import pfstmo_fattal02
from experiment.he_pyramid_fattal.config.config import INPUT_DIR, OUTPUT_DIR, get_parameter_combinations
import utils.utils as utils


def main():
    utils.start_timer()
    utils.print_elapsed("HE + Pyramid Levels Combination Fattal 실험 실행 시작")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 입력 폴더 내의 모든 hdr 파일 경로를 탐색합니다.
    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    total_tasks = len(hdr_files) * len(param_combinations)

    utils.print_elapsed("구간 1 (환경 설정 및 파일 탐색 완료)")
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    print(f"출력 기본 디렉토리: {OUTPUT_DIR}")
    print(f"총 {total_tasks}회의 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 이미지에 대하여 반복 실행
    for img_path in hdr_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
            continue

        if img.ndim == 3:
            img_single = img[:, :, 0]
        else:
            img_single = img

        utils.print_elapsed(f"구간 2 (이미지 로드 완료) - 대상: {file_name}")

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for idx, p in enumerate(param_combinations, 1):
            opt_alpha = p['opt_alpha']
            opt_beta = p['opt_beta']
            opt_noise = p['opt_noise']
            newfattal = p['newfattal']
            fftsolver = p['fftsolver']
            detail_level = p['detail_level']
            hpf_sigma = p.get('hpf_sigma', 0.007)
            he_weight = p.get('he_weight', 1.0)
            selected_levels = p.get('selected_levels', None)

            # alphaxbeta 하위 폴더 생성 (예: a0.9_b0.81)
            subfolder_name = f"a{opt_alpha}_b{opt_beta}"
            save_dir = os.path.join(OUTPUT_DIR, subfolder_name)
            os.makedirs(save_dir, exist_ok=True)

            # 피라미드 층 조합 문자열 (예: L0-1-2-3)
            if selected_levels is not None:
                levels_str = "L" + "-".join(map(str, selected_levels))
            else:
                levels_str = "Lall"

            # 톤 매핑 연산
            L_out = pfstmo_fattal02(
                img_single,
                opt_alpha, opt_beta, opt_noise,
                newfattal, fftsolver, detail_level,
                hpf_sigma=hpf_sigma,
                selected_levels=selected_levels,
                he_weight=he_weight
            )

            # 포맷 변환 및 클리핑 (8bit 이미지)
            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)

            # 결과 저장 (파일명 예: 3_L0-1-2-3_he1.0.png)
            save_name = f"{file_name}_{levels_str}_he{he_weight}.png"
            save_path = os.path.join(save_dir, save_name)

            cv2.imwrite(save_path, out_img_8bit)
            print(f"[{idx}/{len(param_combinations)}] 저장 완료: {save_path}")
    
    utils.print_elapsed("HE + Pyramid Levels Combination Fattal 실험 프로그램 전체 종료")


if __name__ == "__main__":
    main()
