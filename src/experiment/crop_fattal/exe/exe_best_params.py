# exe_best_params.py (Crop + Fattal Baseline Best Parameters)
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

# baseline Fattal TMO 모듈 임포트
from fattal.fattal_tmo import pfstmo_fattal02
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("Crop Fattal best_params 실행 시작")
    
    # 출력 디렉토리 설정
    output_dir = project_root / "test" / "crop_fattal" / "best_param_output"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 생성됨: {output_dir}")

    # 각 데이터셋별 이미지 크롭 범위 설정
    CROP_RANGES = {
        1: {'Y': (459, 1577), 'X': (121, 3072)},
        2: {'Y': (248, 1821), 'X': (124, 3072)},
        3: {'Y': (201, 1833), 'X': (311, 2982)},
        4: {'Y': (278, 1728), 'X': (273, 3072)},
        5: {'Y': (700, 2048), 'X': (0, 2945)},
        6: {'Y': (324, 1746), 'X': (278, 3072)},
        7: {'Y': (307, 1746), 'X': (100, 3012)}
    }

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

    print(f"출력 디렉토리: {output_dir}\n")

    # 1부터 7까지의 데이터셋 폴더를 순회하며 톤 매핑 실행
    for ds_num in sorted(dataset_params.keys()):
        p = dataset_params[ds_num]
        crop = CROP_RANGES[ds_num]
        ds_dir = project_root / "data" / "data_one" / str(ds_num)
        
        search_pattern = str(ds_dir / "*.hdr")
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{ds_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        for img_path in hdr_files:
            file_name = Path(img_path).stem
            print(f"\n--- 데이터셋 [{ds_num}] 크롭 및 Fattal TMO 처리 중 ---")
            print(f"이미지: {file_name}")
            print(f"파라미터 - alpha: {p['alpha']}, beta: {p['beta']}")
            print(f"크롭 영역 - Y: {crop['Y']}, X: {crop['X']}")
            
            # 이미지 로드
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            if img.ndim == 3:
                img_single = img[:,:,0]
            else:
                img_single = img

            # 크롭 적용
            h, w = img_single.shape
            ymin, ymax = crop['Y']
            xmin, xmax = crop['X']
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            img_cropped = img_single[ymin:ymax, xmin:xmax]
            
            print(f"크롭 완료: 원본 ({w}x{h}) -> 크롭본 ({img_cropped.shape[1]}x{img_cropped.shape[0]})")

            # 톤 매핑 연산 실행
            L_out = pfstmo_fattal02(
                img_cropped,
                p['alpha'], p['beta'], opt_noise,
                newfattal, fftsolver, detail_level,
                hpf_sigma=hpf_sigma
            )

            out_img = np.clip(L_out, 0.0, 1.0)
            out_img_8bit = (out_img * 255.0).astype(np.uint8)

            # 결과물 파일 저장 (데이터셋 번호 및 파라미터, 크롭 정보 명시)
            save_name = f"{ds_num}_{file_name}_cropY{ymin}-{ymax}_X{xmin}-{xmax}_a{p['alpha']}_b{p['beta']}.png"
            save_path = output_dir / save_name

            cv2.imwrite(str(save_path), out_img_8bit)
            print(f"저장 완료: {save_path}")
            
    utils.print_elapsed("Crop Fattal best_params 프로그램 전체 종료")

if __name__ == "__main__":
    main()
