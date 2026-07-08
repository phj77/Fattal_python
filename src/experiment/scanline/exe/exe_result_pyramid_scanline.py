# exe_result_pyramid_scanline.py
# Fattal 알고리즘으로 톤 매핑(TMO)을 완료한 최종 결과 LDR 이미지의 가우시안 피라미드 각 레벨별 이미지를 시각화하고, 특정 행(Row)의 스캔라인을 추출하여 저장하는 실행 스크립트입니다.
# Output: test/scanline/result_pyramid_scanline/<dataset_id>/<img_name>/

import os
import cv2
import numpy as np
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 패스 설정 및 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fattal.fattal_tmo import pfstmo_fattal02, createGaussianPyramids
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

# --- 경로 설정 ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Fattal_python root
DATA_DIR = project_root / "data" / "data_one"
OUTPUT_DIR = project_root / "test" / "scanline" / "result_pyramid_scanline"

# --- 데이터셋별 매핑 설정 ---
dataset_configs = {
    1: {"alpha": 0.9, "beta": 0.82, "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
    2: {"alpha": 0.9, "beta": 0.8,  "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
    3: {"alpha": 0.9, "beta": 0.81, "row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
    4: {"alpha": 0.9, "beta": 0.84, "row": 974,  "highlight": [[457, 475], [590, 607]]},
    5: {"alpha": 0.3, "beta": 0.93, "row": 1170, "highlight": [[2073, 2188]]},
    6: {"alpha": 0.9, "beta": 0.81, "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
    7: {"alpha": 0.9, "beta": 0.8,  "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
}

# --- 기본 고정 파라미터 ---
opt_noise = 0.001
newfattal = True
fftsolver = True
detail_level = 0
pre_gamma = 1.0
post_gamma = 1.0
MSIZE = 8 if fftsolver else 32

def main():
    utils.start_timer()
    utils.print_elapsed("결과 이미지 가우시안 피라미드 및 스캔라인 추출 시작")

    print(f"입력 경로: {DATA_DIR}")
    print(f"출력 경로: {OUTPUT_DIR}\n")

    for k in sorted(dataset_configs.keys()):
        config = dataset_configs[k]
        opt_alpha = config["alpha"]
        opt_beta = config["beta"]
        scanline_row = config["row"]
        highlight_ranges = config["highlight"]

        input_dir = DATA_DIR / str(k)
        if not input_dir.exists():
            print(f"  [SKIP] 데이터셋 폴더 {input_dir}가 존재하지 않습니다.")
            continue

        hdr_files = list(input_dir.glob("*.hdr"))
        if not hdr_files:
            print(f"  [SKIP] {input_dir}에 .hdr 파일이 없습니다.")
            continue

        print(f"\n--- 데이터셋 [{k}] 처리 (alpha={opt_alpha}, beta={opt_beta}, row={scanline_row}) ---")
        
        for img_path in hdr_files:
            file_name = img_path.stem
            print(f"  이미지 처리 중: {file_name}")

            # HDR 이미지 로드
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"    [ERROR] 이미지를 읽을 수 없습니다: {img_path}")
                continue

            is_grayscale = (img.ndim == 2)
            if is_grayscale:
                img = np.stack([img, img, img], axis=-1)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

            # 전처리 감마 적용
            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, pre_gamma)
            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            opt_saturation = 1.0 if is_grayscale else 0.8

            # Fattal 톤 매핑 실행 (중간 스캔라인 생성 방지를 위해 scanline_row=None 전달)
            R_out, G_out, B_out = pfstmo_fattal02(
                R_pre, G_pre, B_pre,
                opt_alpha, opt_beta, opt_saturation, opt_noise,
                newfattal, fftsolver, detail_level,
                scanline_row=None, highlight_ranges=None, save_dir=None
            )

            # 후처리 감마 적용
            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, post_gamma)
            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data

            # 최종 톤 매핑 결과 이미지
            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            
            # 최종 결과 이미지(LDR)의 단일 채널(Intensity)을 사용하여 가우시안 피라미드 생성
            # RGB가 그레이스케일이므로 R_final을 그대로 사용
            result_img = R_final

            # 가우시안 피라미드 레벨 수 결정
            h, w = result_img.shape
            mins = min(w, h)
            nlevels = 0
            temp_mins = mins
            while temp_mins >= MSIZE:
                nlevels += 1
                temp_mins //= 2
            if nlevels == 0:
                nlevels = 1

            pyramids = createGaussianPyramids(result_img, nlevels)
            print(f"    결과 가우시안 피라미드 생성 완료: {nlevels} 레벨 (원본 크기: {w}x{h})")

            # 출력 폴더 설정: test/scanline/result_pyramid_scanline/<dataset_id>/<img_name>/
            img_output_dir = OUTPUT_DIR / str(k) / file_name
            img_output_dir.mkdir(parents=True, exist_ok=True)

            # 최종 결과 이미지 저장 (비교용)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)
            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]
            
            result_save_name = f"{file_name}_k{k}_a{opt_alpha}_b{opt_beta}.png"
            cv2.imwrite(str(img_output_dir / result_save_name), out_img_bgr)
            print(f"    최종 결과 이미지 저장: {result_save_name}")

            # 가우시안 피라미드 레벨별로 스캔라인 및 이미지 저장
            for level in range(nlevels):
                level_img = pyramids[level]
                lh, lw = level_img.shape

                # 2^level에 따른 좌표 리스케일링
                scale_factor = 2.0 ** level
                row_k = int(round(scanline_row / scale_factor))
                row_k = np.clip(row_k, 0, lh - 1)

                # 하이라이트 구간 스케일링
                highlight_ranges_k = []
                if highlight_ranges is not None:
                    for rng in highlight_ranges:
                        if len(rng) == 2:
                            start_k = int(round(rng[0] / scale_factor))
                            end_k = int(round(rng[1] / scale_factor))
                            start_k = np.clip(start_k, 0, lw - 1)
                            end_k = np.clip(end_k, 0, lw - 1)
                            highlight_ranges_k.append([start_k, end_k])

                # 1. 피라미드 이미지 자체 시각화 저장
                level_img_8bit = (np.clip(level_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                pyramid_img_name = f"result_pyramid_level_{level:02d}_{lw}x{lh}.png"
                cv2.imwrite(str(img_output_dir / pyramid_img_name), level_img_8bit)

                # 2. 피라미드 이미지 스캔라인 그래프 및 npy 데이터 저장
                stage_name = f"result_pyramid_level_{level:02d}"
                utils.save_scanline(
                    level_img, 
                    row_index=row_k, 
                    stage_name=stage_name, 
                    highlight_ranges=highlight_ranges_k, 
                    save_dir=str(img_output_dir)
                )
                print(f"      레벨 {level:2d}: 이미지 및 row {row_k} 스캔라인 저장 완료 (해상도: {lw}x{lh})")

    utils.print_elapsed("결과 이미지 가우시안 피라미드 및 스캔라인 추출 종료")
    print(f"\n모든 출력물이 다음에 성공적으로 저장되었습니다: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
