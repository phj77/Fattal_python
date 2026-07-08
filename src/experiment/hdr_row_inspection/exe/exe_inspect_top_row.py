import os
import sys
import glob
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 65)
    print(f"{'데이터셋':<8} | {'HDR 파일명':<32} | {'최소값(Min)':<10} | {'최대값(Max)':<10} | {'평균값(Mean)':<10}")
    print("=" * 65)

    for dataset_num in range(1, 8):
        data_dir = os.path.join(project_root, "data", str(dataset_num))
        hdr_files = glob.glob(os.path.join(data_dir, "*.hdr"))

        if not hdr_files:
            print(f"{dataset_num:<8} | {'HDR 파일 없음':<32} | {'-':<10} | {'-':<10} | {'-':<10}")
            continue

        for hdr_path in hdr_files:
            file_name = os.path.basename(hdr_path)
            img = cv2.imread(hdr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if img is None:
                print(f"{dataset_num:<8} | {file_name[:30]:<32} | {'읽기 실패':<10} | {'-':<10} | {'-':<10}")
                continue

            # 맨 위 1행(row 0) 픽셀값
            top_row = img[0]

            r_min = np.min(top_row)
            r_max = np.max(top_row)
            r_mean = np.mean(top_row)

            # 파일명이 길 경우 조절
            disp_name = file_name if len(file_name) <= 32 else file_name[:29] + "..."
            print(f"{dataset_num:<8} | {disp_name:<32} | {r_min:<10.2f} | {r_max:<10.2f} | {r_mean:<10.2f}")

    print("=" * 65)

if __name__ == "__main__":
    main()
