# exe.py
import cv2
import numpy as np
import os
import glob

# 사용자 정의 모듈 (환경에 맞게 존재해야 함)
from fattal_tmo import pfstmo_fattal02
from gamma_correction import Frame, apply_gamma_frame

# 파라미터 및 설정 불러오기
from config import INPUT_DIR, OUTPUT_DIR, get_parameter_combinations

def main():
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 입력 폴더 내의 모든 hdr 파일 경로를 탐색합니다.
    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
        return

    # 파라미터 조합을 가져옵니다.
    param_combinations = get_parameter_combinations()
    total_tasks = len(hdr_files) * len(param_combinations)
    
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    print(f"총 {total_tasks}회의 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 이미지에 대하여 반복 실행
    for img_path in hdr_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 이미지 로드
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if img is None:
            print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
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

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for p in param_combinations:
            he_weight_clipped = np.clip(p['HE_weight'], 0, 1)
            
            # 그레이스케일일 경우 채도 복원 파라미터 무력화
            opt_saturation = 1.0 if is_grayscale else 0.8

            # 전처리 감마 보정
            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, p['pre_gamma'])

            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            # 톤 매핑
            R_out, G_out, B_out = pfstmo_fattal02(
                R_pre, G_pre, B_pre,
                p['opt_alpha'], p['opt_beta'], opt_saturation, p['opt_noise'],
                p['newfattal'], p['fftsolver'], p['detail_level'], he_weight_clipped
            )

            # 후처리 감마 보정
            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, p['post_gamma'])

            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data

            # 채널 병합 및 포맷 변환
            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

            # 원본이 그레이스케일이면 단채널로 변환 후 저장
            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]

            # 식별 가능한 파일명 생성 및 저장
            # 주요 파라미터를 파일명에 포함하여 덮어쓰기를 방지하고 결과를 구분합니다.
            param_suffix = f"a{p['opt_alpha']}_b{p['opt_beta']}_he{he_weight_clipped}_pre{p['pre_gamma']}_post{p['post_gamma']}"
            save_name = f"{file_name}_{param_suffix}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, out_img_bgr)
            print(f"완료: {save_path}")

    print("\n모든 작업이 종료되었습니다.")

if __name__ == "__main__":
    main()