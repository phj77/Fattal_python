# exe_gaussian_pyramid_fattal.py
# 입력 HDR 이미지의 Gaussian pyramid를 만들고 각 층(level)마다 fattal_tmo를 적용하여 결과를 저장하는 실험 스크립트

import cv2
import numpy as np
import os
import glob
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fattal.fattal_tmo import gaussianBlur, downSample, pfstmo_fattal02
from experiment.gaussian_pyramid_fattal.config.config import INPUT_DIR, OUTPUT_DIR, get_parameter_combinations
import utils.utils as utils


def build_gaussian_pyramid(img, num_levels=None, top_size=32):
    """
    입력 1채널 이미지(HDR)에 대해 가우시안 피라미드 층을 생성합니다.
    - level 0: 원본 이미지
    - level k: gaussianBlur 후 downSample된 이미지
    """
    h, w = img.shape
    if num_levels is None:
        mins = min(h, w)
        num_levels = 0
        temp_mins = mins
        while temp_mins >= top_size:
            num_levels += 1
            temp_mins //= 2
        if num_levels == 0:
            num_levels = 1

    pyramid = [img.copy()]
    curr = img.copy()
    for k in range(1, num_levels):
        blurred = gaussianBlur(curr)
        curr = downSample(blurred)
        pyramid.append(curr.copy())

    return pyramid


def main():
    utils.start_timer()
    utils.print_elapsed("Gaussian Pyramid + Level-wise Fattal TMO 실험 시작")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다. 경로: {INPUT_DIR}")
        return

    param_combinations = get_parameter_combinations()
    utils.print_elapsed("환경 설정 및 입력 HDR 파일 탐색 완료")
    print(f"발견된 HDR 파일 수: {len(hdr_files)}")
    print(f"출력 디렉토리: {OUTPUT_DIR}\n")

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

        # 가우시안 피라미드 구성
        pyramid = build_gaussian_pyramid(img_single)
        num_levels = len(pyramid)
        print(f"[{file_name}] 가우시안 피라미드 총 {num_levels}개 층 생성 완료.")

        for p in param_combinations:
            param_suffix = f"a{p['opt_alpha']}_b{p['opt_beta']}_dl{p['detail_level']}_hpf_sigma{p['hpf_sigma']}"

            # 각 피라미드 레벨별로 Fattal TMO 적용 및 저장
            for level, level_img in enumerate(pyramid):
                utils.print_elapsed(f"  -> [{file_name}] Level {level}/{num_levels-1} ({level_img.shape[1]}x{level_img.shape[0]}) TMO 연산 시작")

                L_out = pfstmo_fattal02(
                    level_img,
                    p['opt_alpha'], p['opt_beta'], p['opt_noise'],
                    p['newfattal'], p['fftsolver'], p['detail_level'],
                    hpf_sigma=p.get('hpf_sigma', 0.007)
                )

                out_img = np.clip(L_out, 0.0, 1.0)
                out_img_8bit = (out_img * 255.0).astype(np.uint8)

                # 파라미터별 하위 저장 폴더 생성 (레벨별 폴더 생성을 하지 않고 파일명에 레벨 명시)
                save_dir = os.path.join(OUTPUT_DIR, param_suffix)
                os.makedirs(save_dir, exist_ok=True)

                save_name = f"{file_name}_level_{level}_{param_suffix}.png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, out_img_8bit)
                print(f"  [저장 완료] {save_path}")

    utils.print_elapsed("Gaussian Pyramid Level-wise Fattal TMO 실험 완료")


if __name__ == "__main__":
    main()
