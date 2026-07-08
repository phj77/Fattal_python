import cv2
import numpy as np
import os
import glob
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(src_dir)

from experiment.scanline.fattal.fattal_tmo import pfstmo_fattal02
from exe.config.config import INPUT_DIR, OUTPUT_DIR
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("세로 스캔라인 배치 생성 시작")

    # INPUT_DIR가 특정 데이터셋 번호 폴더(예: data_one/3)를 가리킬 경우 해당 폴더만 연산 대상으로 지정합니다.
    # 그렇지 않다면 INPUT_DIR 내에 있는 숫자 폴더들을 연산 대상으로 지정합니다.
    norm_input = os.path.normpath(INPUT_DIR)
    base_name = os.path.basename(norm_input)

    dataset_dirs = []
    if base_name.isdigit():
        num_key = int(base_name)
        dataset_dirs.append((num_key, norm_input))
    else:
        for k in range(1, 8):
            sub_dir = os.path.join(norm_input, str(k))
            if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
                dataset_dirs.append((k, sub_dir))

    output_dir = OUTPUT_DIR

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dataset별 설정 매핑 (세로 스캔라인 및 하이라이트 구간 포함)
    # 3, 4, 5번 이외의 데이터셋은 세로 스캔라인 추출 설정을 비워둡니다 (col: None).
    dataset_configs = {
        1: {"alpha": 0.9, "beta": 0.82, "col": None, "highlight": None},
        2: {"alpha": 0.9, "beta": 0.80, "col": None, "highlight": None},
        3: {"alpha": 0.9, "beta": 0.81, "col": 2311, "highlight": [[206, 328], [1716, 1815]]},
        4: {"alpha": 0.9, "beta": 0.84, "col": 2486, "highlight": [[275, 377], [1662, 1730]]},
        5: {"alpha": 0.3, "beta": 0.93, "col": 1348, "highlight": [[695, 848]]},
        6: {"alpha": 0.9, "beta": 0.81, "col": None, "highlight": None},
        7: {"alpha": 0.9, "beta": 0.80, "col": None, "highlight": None}
    }

    # 데이터셋 순회 실행
    for k, input_dir in dataset_dirs:
        config = dataset_configs.get(k, {"alpha": 0.9, "beta": 0.8, "col": None, "highlight": None})
        
        # 각 데이터셋별 폴더 경로 설정
        dataset_output_dir = os.path.join(output_dir, str(k))
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)
        
        search_pattern = os.path.join(input_dir, '*.hdr')
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{input_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        print(f"\n--- 데이터셋 [{k}] 세로 스캔라인 처리 시작 (이미지 개수: {len(hdr_files)}) ---")
        
        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"이미지 로딩 중: {file_name}")
            
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue
                
            # 단일 채널 이미지로 추출하여 사용
            if img.ndim == 3:
                img_single = img[:, :, 0]
            else:
                img_single = img
            
            # 파라미터 매핑
            opt_alpha = config["alpha"]
            opt_beta = config["beta"]
            scanline_col = config["col"]
            highlight_ranges = config["highlight"]
            
            opt_noise = 0.001
            newfattal = True
            fftsolver = True
            detail_level = 0
            
            # 톤 매핑 및 세로 스캔라인 저장 실행
            L_out = pfstmo_fattal02(
                img_single,
                opt_alpha, opt_beta, opt_noise,
                newfattal, fftsolver, detail_level,
                scanline_row=None, highlight_ranges=highlight_ranges,
                save_dir=dataset_output_dir,
                scanline_col=scanline_col
            )
            
            # 포맷 변환 및 클리핑 (8bit 단일 채널 이미지)
            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)
                
            save_name = f"{file_name}_k{k}_a{opt_alpha}_b{opt_beta}.png"
            save_path = os.path.join(dataset_output_dir, save_name)
            cv2.imwrite(save_path, out_img_8bit)
            print(f"결과 이미지 저장 완료: {save_path}")

    utils.print_elapsed("모든 작업 종료")

if __name__ == "__main__":
    main()
