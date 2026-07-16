import cv2
import numpy as np
import os
import glob
import sys

# 프로젝트 root 및 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(src_dir)

# DCT-II 실험용 모듈 가져오기
from experiment.DCT2.fattal.fattal_tmo import pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
from experiment.DCT2.config.config import INPUT_DIR, get_parameter_combinations
import utils.utils as utils

# 실험 전용 결과 저장 디렉토리
OUTPUT_DIR_DCT2 = os.path.join(os.path.dirname(os.path.dirname(src_dir)), "output_dct2")

def main():
    utils.start_timer()
    utils.print_elapsed("DCT-II 실험 실행 시작")
    
    if not os.path.exists(OUTPUT_DIR_DCT2):
        os.makedirs(OUTPUT_DIR_DCT2)

    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    total_tasks = len(hdr_files) * len(param_combinations)
    
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    print(f"결과 저장 디렉토리: {OUTPUT_DIR_DCT2}")

    for img_path in hdr_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if img is None:
            print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
            continue

        is_grayscale = (img.ndim == 2)
        if is_grayscale:
            img = np.stack([img, img, img], axis=-1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]

        for p in param_combinations:
            opt_saturation = 1.0 if is_grayscale else 0.8

            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, p['pre_gamma'])

            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            # DCT-II 기반 pfstmo_fattal02 실행
            R_out, G_out, B_out = pfstmo_fattal02(
                R_pre, G_pre, B_pre,
                p['opt_alpha'], p['opt_beta'], opt_saturation, p['opt_noise'],
                p['newfattal'], p['fftsolver'], p['detail_level']
            )

            param_suffix = f"DCT2_a{p['opt_alpha']}_b{p['opt_beta']}_pre{p['pre_gamma']}_post{p['post_gamma']}"

            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, p['post_gamma'])

            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data

            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2RGB if is_grayscale else cv2.COLOR_RGB2BGR)

            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]

            save_name = f"{file_name}_{param_suffix}.png"
            save_path = os.path.join(OUTPUT_DIR_DCT2, save_name)

            cv2.imwrite(save_path, out_img_bgr)
            print(f"[DCT-II 완료]: {save_path}")

    utils.print_elapsed("DCT-II 실험 프로그램 전체 종료")

if __name__ == "__main__":
    main()
