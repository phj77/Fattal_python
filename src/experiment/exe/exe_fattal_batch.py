# exe_fattal_batch.py
import cv2
import numpy as np
import os
import glob
import sys
import time
from pathlib import Path

# 프로젝트 루트 및 라이브러리 경로 추가
current_file = Path(__file__).resolve()
src_dir = current_file.parents[2]      # src/ 폴더
project_root = current_file.parents[3] # Fattal_python/ 루트 폴더
sys.path.append(str(src_dir))

from fattal.fattal_tmo import pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# 데이터셋별 파라미터 매핑 (alpha, beta)
DATASET_PARAMS = {
    "1": {"alpha": 0.9, "beta": 0.82},
    "2": {"alpha": 0.9, "beta": 0.80},
    "3": {"alpha": 0.9, "beta": 0.81},
    "4": {"alpha": 0.9, "beta": 0.84},
    "5": {"alpha": 0.3, "beta": 0.93},
    "6": {"alpha": 0.9, "beta": 0.81},
    "7": {"alpha": 0.9, "beta": 0.80}
}

def main():
    utils.start_timer()
    utils.print_elapsed("Fattal 톤 매핑 배치 작업 시작")

    data_dir = project_root / "data" / "data_one"
    output_dir = project_root / "test" / "fattal_tmo"

    print(f"Project Root: {project_root}")
    print(f"Input Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")

    # 데이터셋 1부터 7까지 순회
    for dataset_num in sorted(DATASET_PARAMS.keys()):
        params = DATASET_PARAMS[dataset_num]
        opt_alpha = params["alpha"]
        opt_beta = params["beta"]
        
        dataset_path = data_dir / dataset_num
        if not dataset_path.exists():
            print(f"경고: 데이터셋 폴더 {dataset_path}가 존재하지 않습니다. 건너뜁니다.")
            continue
            
        # .hdr 파일 검색
        hdr_files = list(dataset_path.glob("*.hdr"))
        if not hdr_files:
            print(f"정보: {dataset_path}에 .hdr 파일이 없습니다.")
            continue
            
        # 각 데이터셋별 출력 폴더 생성
        dataset_output_dir = output_dir / dataset_num
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[데이터셋 {dataset_num}] 시작 (alpha={opt_alpha}, beta={opt_beta}, 이미지 개수: {len(hdr_files)})")
        
        for hdr_path in hdr_files:
            img_name = hdr_path.stem  # 파일명 (확장자 제외)
            print(f" - 처리 중: {hdr_path.name}")
            
            # 이미지 로드
            img = cv2.imread(str(hdr_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다: {hdr_path}")
                continue
                
            # 그레이스케일 여부 감지 및 3채널 복제
            is_grayscale = (img.ndim == 2)
            if is_grayscale:
                img = np.stack([img, img, img], axis=-1)

            # BGR → RGB 변환 및 채널 분리
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            R = img_rgb[:, :, 0]
            G = img_rgb[:, :, 1]
            B = img_rgb[:, :, 2]

            # 고정 파라미터 정의
            opt_noise = 0.001
            newfattal = True
            fftsolver = True
            detail_level = 0
            he_weight = 0.0
            pre_gamma = 1.0
            post_gamma = 1.0
            
            opt_saturation = 1.0 if is_grayscale else 0.8

            # 전처리 감마 보정
            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, pre_gamma)
            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            # 톤 매핑 실행
            R_out, G_out, B_out = pfstmo_fattal02(
                R_pre, G_pre, B_pre,
                opt_alpha, opt_beta, opt_saturation, opt_noise,
                newfattal, fftsolver, detail_level, he_weight
            )

            # 후처리 감마 보정
            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, post_gamma)
            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data

            # 채널 병합 및 8비트 포맷 변환
            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

            # 원본이 그레이스케일이면 단채널로 변환
            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]

            # 저장 경로 설정 및 저장
            save_path = dataset_output_dir / f"{img_name}.png"
            cv2.imwrite(str(save_path), out_img_bgr)
            print(f"   => 저장 완료: {save_path.name}")
            
    utils.print_elapsed("모든 작업 완료")

if __name__ == "__main__":
    main()
