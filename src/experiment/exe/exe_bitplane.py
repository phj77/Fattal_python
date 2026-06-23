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
    data_dir = project_root / "data" / "data_one"
    output_dir = project_root / "test" / "bitplane_HP"

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
            
        print(f"\n[데이터셋 {dataset_num}] 톤맵핑 시작 (alpha={opt_alpha}, beta={opt_beta})")
        
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

            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]

            # 출력 폴더 생성: test/bitplane/[dataset_num]/[image_name]/
            save_dir = output_dir / dataset_num / img_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # 톤맵핑 이미지 저장
            tonemapped_path = save_dir / "tonemapped.png"
            cv2.imwrite(str(tonemapped_path), out_img_bgr)
            
            # 비트 플레인 추출 및 저장용 그레이스케일 이미지 획득
            if out_img_bgr.ndim == 3:
                gray_img = cv2.cvtColor(out_img_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = out_img_bgr

            # 비트 플레인 0~7 저장 (Bit 0이 LSB, Bit 7이 MSB)
            for i in range(8):
                bit_plane = ((gray_img >> i) & 1) * 255
                bit_plane_path = save_dir / f"bit_{i}.png"
                cv2.imwrite(str(bit_plane_path), bit_plane.astype(np.uint8))
                
            print(f" - 완료: {save_dir}")

    print("\n모든 작업이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()
