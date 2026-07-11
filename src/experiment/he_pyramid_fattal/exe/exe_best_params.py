# exe_best_params.py (HE + Pyramid Levels Combination Fattal)
import cv2
import numpy as np
import os
import glob
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 프로젝트 src 경로를 sys.path에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]  # Fattal_python/
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# he_pyramid_fattal 실험 모듈 임포트
from experiment.he_pyramid_fattal.fattal.fattal_tmo import pfstmo_fattal02
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("HE + Pyramid Fattal best_params 실행 시작")
    
    # 출력 디렉토리 설정
    output_dir = project_root / "test" / "he_pyramid_fattal" / "best_param_output"
    
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
    
    # he_pyramid_fattal 실험용 기본 파라미터 설정
    opt_noise = 0.001
    newfattal = True
    fftsolver = True
    detail_level = 0
    hpf_sigma = 0.007
    
    # HE + Pyramid 특화 파라미터
    he_weights = [0.0, 0.1, 0.4,0.8]  # 히스토그램 평활화(HE) 반영 가중치 리스트 (0.0: 미반영 ~ 1.0: 100% 반영)
    selected_levels = None        # None이면 전체 피라미드 층 사용. 혹은 특정 레벨 리스트 지정 (예: [0, 1, 2, 3])

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
            
            # 이미지 로드
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            if img.ndim == 3:
                img_single = img[:,:,0]
            else:
                img_single = img

            # 피라미드 층 표시용 문자열 생성
            if selected_levels is not None:
                levels_str = "L" + "-".join(map(str, selected_levels))
            else:
                levels_str = "Lall"

            for he_weight in he_weights:
                print(f"\n--- 데이터셋 [{ds_num}] 처리 중 (HE + Pyramid) ---")
                print(f"이미지: {file_name}")
                print(f"파라미터 - alpha: {p['alpha']}, beta: {p['beta']}, he_weight: {he_weight}")

                # HE + Pyramid 기반 톤 매핑 연산 실행
                L_out = pfstmo_fattal02(
                    img_single,
                    p['alpha'], p['beta'], opt_noise,
                    newfattal, fftsolver, detail_level,
                    hpf_sigma=hpf_sigma,
                    selected_levels=selected_levels,
                    he_weight=he_weight
                )

                out_img = np.clip(L_out, 0.0, 1.0)
                out_img_8bit = (out_img * 255.0).astype(np.uint8)

                # 결과물 파일 저장 (데이터셋 번호 및 파라미터, HE 가중치, 선택된 레벨 정보 명시)
                save_name = f"{ds_num}_{file_name}_he_pyramid_a{p['alpha']}_b{p['beta']}_{levels_str}_he{he_weight}.png"
                save_path = output_dir / save_name

                cv2.imwrite(str(save_path), out_img_8bit)
                print(f"저장 완료: {save_path}")
            
    utils.print_elapsed("HE + Pyramid Fattal best_params 프로그램 전체 종료")

if __name__ == "__main__":
    main()
