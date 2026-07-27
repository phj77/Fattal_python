# exe_best_params.py
import cv2
import numpy as np
import os
import glob
import sys
import time
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 사용자 정의 모듈 임포트
from experiment.input_hdr_fattal.fattal.fattal_tmo import pfstmo_fattal02
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("시작")
    
    # 프로젝트 경로 및 출력 디렉토리 설정
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    output_dir = project_root / "test" / "best_param_output_toppyramid8"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 생성됨: {output_dir}")

    # 데이터셋 번호와 이에 상응하는 alpha, beta 파라미터 설정
    dataset_params = {
        1: {'alpha': 0.9, 'beta': 0.82},
        2: {'alpha': 0.9, 'beta': 0.80},
        3: {'alpha': 0.9, 'beta': 0.81},
        4: {'alpha': 0.9, 'beta': 0.84},
        5: {'alpha': 0.3, 'beta': 0.93},
        6: {'alpha': 0.9, 'beta': 0.81},
        7: {'alpha': 0.9, 'beta': 0.80}
    }
    
    # 기본 파라미터들
    opt_noise = 0.001
    newfattal = True
    fftsolver = True
    detail_level = 0
    hpf_sigma = 0.007
    pyramid_top_size = 2**3

    print(f"출력 디렉토리: {output_dir}\n")

    # 1부터 7까지의 데이터셋 폴더를 순회하며 톤 매핑 실행
    for ds_num in sorted(dataset_params.keys()):
        p = dataset_params[ds_num]
        ds_dir = project_root / "data" / "data_one" / str(ds_num)
        
        search_pattern = str(ds_dir / "*.hdr")
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{ds_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        for img_path in hdr_files:
            file_name = Path(img_path).stem
            print(f"\n--- 데이터셋 [{ds_num}] 처리 중 ---")
            print(f"이미지: {file_name}")
            print(f"파라미터 - alpha: {p['alpha']}, beta: {p['beta']}")
            
            # 이미지 로드
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            if img.ndim == 3:
                img = img[:,:,0]
            else:
                print("\nPlease input 3-channel grayscale image.\n")
                sys.exit(1)

            # 톤 매핑 연산 실행
            L_out = pfstmo_fattal02(
                img,
                p['alpha'], p['beta'], opt_noise,
                newfattal, fftsolver, detail_level,
                hpf_sigma=hpf_sigma,
                pyramid_top_size=pyramid_top_size
            )

            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)

            # 결과물 파일 저장 (데이터셋 번호 및 파라미터 정보를 명시)
            save_name = f"{ds_num}_{file_name}_a{p['alpha']}_b{p['beta']}.png"
            save_path = output_dir / save_name

            cv2.imwrite(str(save_path), out_img_8bit)
            print(f"저장 완료: {save_path}")
            
    utils.print_elapsed("프로그램 전체 종료")

if __name__ == "__main__":
    main()
