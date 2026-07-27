# exe_input_raw_reverse.py
import cv2
import numpy as np
import os
import glob
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from experiment.input_raw_reverse.fattal.fattal_tmo import pfstmo_fattal02
from experiment.input_raw_reverse.config.config import (
    INPUT_DIR, OUTPUT_DIR, CROP_Y_RANGE, CROP_X_RANGE, get_parameter_combinations
)
import utils.utils as utils


def load_raw_image(img_path: str) -> np.ndarray:
    """
    raw 데이터 파일을 읽어 numpy float32 배열로 변환합니다.
    - 파일 크기가 24MiB (25,165,824 bytes) 이면: reshape((2048, 3072))
    - 파일 크기가 36MiB (37,748,736 bytes) 이면: reshape((3072, 3072))
    """
    file_size = os.path.getsize(img_path)
    size_24mib = 24 * 1024 * 1024  # 25,165,824 bytes
    size_36mib = 36 * 1024 * 1024  # 37,748,736 bytes

    if file_size == size_24mib:
        shape = (2048, 3072)
    elif file_size == size_36mib:
        shape = (3072, 3072)
    else:
        # float32 개수로 분기 예외 처리
        num_floats = file_size // 4
        if num_floats == 2048 * 3072:
            shape = (2048, 3072)
        elif num_floats == 3072 * 3072:
            shape = (3072, 3072)
        else:
            raise ValueError(
                f"지원하지 않는 raw 파일 크기입니다: {file_size} bytes ({file_size / (1024 * 1024):.2f} MiB). "
                f"24MiB (2048x3072) 또는 36MiB (3072x3072) 이어야 합니다."
            )

    img = np.fromfile(img_path, dtype=np.float32).reshape(shape)
    return img


def main():
    utils.start_timer()
    utils.print_elapsed("RAW Input Reverse Fattal TMO 시작")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 입력 경로가 단일 파일인지 폴더인지 구분하여 .raw 파일 목록 생성
    raw_files = []
    if os.path.isfile(INPUT_DIR):
        if INPUT_DIR.lower().endswith('.raw'):
            raw_files.append(INPUT_DIR)
        base_input_dir = os.path.dirname(INPUT_DIR)
    elif os.path.isdir(INPUT_DIR):
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith('.raw'):
                    raw_files.append(os.path.join(root, file))
        base_input_dir = INPUT_DIR
    else:
        print(f"오류: '{INPUT_DIR}' 경로가 존재하지 않습니다.")
        return

    if not raw_files:
        print(f"경고: '{INPUT_DIR}'에서 .raw 파일을 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    total_tasks = len(raw_files) * len(param_combinations)

    utils.print_elapsed("구간 1 (환경 설정 및 파일 탐색 완료)")
    print(f"총 {len(raw_files)}개의 RAW 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        print(f"크롭 범위 - Y축: {CROP_Y_RANGE}, X축: {CROP_X_RANGE}")
    print(f"총 {total_tasks}회의 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 RAW 이미지에 대하여 반복 실행
    for img_path in raw_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]

        # 원본 INPUT_DIR 기준 상대 경로 계산 (폴더 구조 유지)
        rel_path = os.path.relpath(img_path, base_input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_out_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(target_out_dir, exist_ok=True)

        try:
            img_single = load_raw_image(img_path)
        except Exception as e:
            print(f"오류: RAW 이미지 읽기 실패 ({img_path}) - {e}")
            continue

        # 이미지 크롭 적용
        crop_suffix = ""
        if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
            h, w = img_single.shape
            ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h)
            xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w)
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            img_single = img_single[ymin:ymax, xmin:xmax]
            crop_suffix = f"_cropY{ymin}-{ymax}_X{xmin}-{xmax}"
            utils.print_elapsed(f"구간 2.5 (이미지 크롭 완료: Y[{ymin}:{ymax}], X[{xmin}:{xmax}])")
        else:
            h, w = img_single.shape
            utils.print_elapsed(f"구간 2 (RAW 이미지 로드 완료) - 대상: {rel_path} (해상도: {w}x{h})")

        # Original domain에서 이미지 반전 (Reverse in original domain)
        img_single = np.max(img_single) - img_single
        utils.print_elapsed("구간 2.8 (Original Domain 이미지 반전 완료: np.max - img)")

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for p in param_combinations:
            # 톤 매핑 연산
            L_out = pfstmo_fattal02(
                img_single,
                p['opt_alpha'], p['opt_beta'], p['opt_noise'],
                p['newfattal'], p['fftsolver'], p['detail_level'],
                hpf_sigma=p.get('hpf_sigma', 0.007),
                pyramid_top_size=p['pyramid_top_size']
            )

            param_suffix = f"a{p['opt_alpha']}_b{p['opt_beta']}{crop_suffix}"
            utils.print_elapsed(f"구간 3 (톤 매핑 연산 완료) - 파라미터: {param_suffix}")

            # 포맷 변환 및 클리핑 (8bit)
            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)

            # 8bit single-channel -> 8bit 3-channel RGB (OpenCV BGR 포맷 사용)
            out_img_rgb = cv2.cvtColor(out_img_8bit, cv2.COLOR_GRAY2BGR)

            utils.print_elapsed("구간 4 (8bit RGB 변환 완료)")

            if len(param_combinations) == 1 and not crop_suffix:
                save_name = f"{file_name}.jpg"
                save_path = os.path.join(target_out_dir, save_name)
            else:
                save_dir = os.path.join(target_out_dir, param_suffix)
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"{file_name}.jpg"
                save_path = os.path.join(save_dir, save_name)

            # 8bit RGB .jpg 형식으로 저장 (주의사항 1 & 3: 해상도 유지 및 .jpg 형식)
            cv2.imwrite(save_path, out_img_rgb)
            print(f"완료: {save_path} (해상도: {out_img_rgb.shape[1]}x{out_img_rgb.shape[0]})")
            utils.print_elapsed("구간 5 (파일 저장 완료)")

    utils.print_elapsed("RAW Input Reverse Fattal TMO 프로그램 전체 종료")


if __name__ == "__main__":
    main()
