# exe_fusion.py
import cv2
import numpy as np
import os
import glob

# 사용자 정의 모듈
from fattal_tmo import pfstmo_fattal02_fusion
from gamma_correction import Frame, apply_gamma_frame

# 파라미터 및 설정 불러오기
from config_fusion import INPUT_DIR, OUTPUT_DIR, get_parameter_combinations

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    search_pattern = os.path.join(INPUT_DIR, '*.hdr')
    hdr_files = glob.glob(search_pattern)

    if not hdr_files:
        print(f"경고: '{INPUT_DIR}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    total_tasks = len(hdr_files) * len(param_combinations)
    
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    print(f"총 {total_tasks}회의 퓨전 톤 매핑 작업이 시작됩니다.\n")

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
            he_weight_clipped = np.clip(p['HE_weight'], 0, 1)
            
            opt_saturation = 1.0 if is_grayscale else 0.8

            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, p['pre_gamma'])

            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data

            # 퓨전 톤 매핑 실행
            R_out, G_out, B_out = pfstmo_fattal02_fusion(
                R_pre, G_pre, B_pre,
                p['opt_alpha'], p['opt_betas'], opt_saturation, p['opt_noise'],
                p['newfattal'], p['fftsolver'], p['detail_level'], he_weight_clipped
            )

            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, p['post_gamma'])

            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data

            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]

            # 리스트 형태의 opt_betas를 파일명에 사용할 수 있도록 문자열로 변환
            betas_str = "_".join(map(str, p['opt_betas']))
            param_suffix = f"a{p['opt_alpha']}_b{betas_str}_he{he_weight_clipped}"
            save_name = f"{file_name}_{param_suffix}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, out_img_bgr)
            print(f"완료: {save_path}")

    print("\n모든 작업이 종료되었습니다.")

if __name__ == "__main__":
    main()