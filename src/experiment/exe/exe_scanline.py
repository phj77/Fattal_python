# exe_scanline.py
import cv2
import numpy as np
import os
import glob
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fattal.fattal_tmo import pfstmo_fattal02
from processing.gamma_correction import Frame, apply_gamma_frame
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("스캔라인 배치 생성 시작")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    data_path = os.path.join(project_root, "data", "data_one")
    output_dir = os.path.join(project_root, "test", "scanline", "scan_tmp")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dataset별 설정 매핑
    dataset_configs = {
        1: {"alpha": 0.9, "beta": 0.82, "row": 1100, "highlight": [[2310, 2382], [1740, 1825]]},
        2: {"alpha": 0.9, "beta": 0.8,  "row": 1661, "highlight": [[300, 530], [1868, 1965]]},
        3: {"alpha": 0.9, "beta": 0.81, "row": 955,  "highlight": [[533, 622], [1380, 1490], [2260, 2355]]},
        4: {"alpha": 0.9, "beta": 0.84, "row": 974,  "highlight": [[457, 475], [590, 607]]},
        5: {"alpha": 0.3, "beta": 0.93, "row": 1170, "highlight": [[2073, 2188]]},
        6: {"alpha": 0.9, "beta": 0.81, "row": 1590, "highlight": [[400, 620], [2095, 2190]]},
        7: {"alpha": 0.9, "beta": 0.8,  "row": 1338, "highlight": [[1295, 1360], [2570, 2650]]}
    }

    # 1부터 7까지 순회
    for k in range(1, 8):
        config = dataset_configs[k]
        input_dir = os.path.join(data_path, str(k))
        
        # 각 데이터셋별 폴더 경로 설정
        dataset_output_dir = os.path.join(output_dir, str(k))
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)
        
        search_pattern = os.path.join(input_dir, '*.hdr')
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{input_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        print(f"\n--- 데이터셋 [{k}] 처리 시작 (이미지 개수: {len(hdr_files)}) ---")
        
        for img_path in hdr_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"이미지 로딩 중: {file_name}")
            
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
            
            # 파라미터 매핑
            opt_alpha = config["alpha"]
            opt_beta = config["beta"]
            scanline_row = config["row"]
            highlight_ranges = config["highlight"]
            
            opt_noise = 0.001
            newfattal = True
            fftsolver = True
            detail_level = 0
            HE_weight = 0.0
            pre_gamma = 1.0
            post_gamma = 1.0
            opt_saturation = 1.0 if is_grayscale else 0.8
            
            # 전처리 감마
            pre_frame = Frame(R, G, B)
            apply_gamma_frame(pre_frame, pre_gamma)
            R_pre = pre_frame.x_channel.data
            G_pre = pre_frame.y_channel.data
            B_pre = pre_frame.z_channel.data
            
            # 톤 매핑 실행 (이 과정에서 save_scanline이 수행됨)
            R_out, G_out, B_out = pfstmo_fattal02(
                R_pre, G_pre, B_pre,
                opt_alpha, opt_beta, opt_saturation, opt_noise,
                newfattal, fftsolver, detail_level, HE_weight,
                scanline_row=scanline_row, highlight_ranges=None,
                save_dir=dataset_output_dir
            )
            
            # 후처리 감마 및 이미지 저장 (결과 확인용)
            post_frame = Frame(R_out, G_out, B_out)
            apply_gamma_frame(post_frame, post_gamma)
            R_final = post_frame.x_channel.data
            G_final = post_frame.y_channel.data
            B_final = post_frame.z_channel.data
            
            out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
            out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
            out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)
            
            if is_grayscale:
                out_img_bgr = out_img_bgr[:, :, 0]
                
            save_name = f"{file_name}_k{k}_a{opt_alpha}_b{opt_beta}.png"
            save_path = os.path.join(dataset_output_dir, save_name)
            cv2.imwrite(save_path, out_img_bgr)
            print(f"결과 이미지 저장 완료: {save_path}")

    utils.print_elapsed("모든 작업 종료")

if __name__ == "__main__":
    main()
