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
    print("=" * 70)
    print(f"{'데이터셋':<8} | {'HDR 파일명':<32} | {'전체 Min':<12} | {'전체 Max':<12} | {'전체 Mean':<12}")
    print("=" * 70)

    for dataset_num in range(1, 8):
        data_dir = os.path.join(project_root, "data", str(dataset_num))
        hdr_files = glob.glob(os.path.join(data_dir, "*.hdr"))

        if not hdr_files:
            print(f"{dataset_num:<8} | {'HDR 파일 없음':<32} | {'-':<12} | {'-':<12} | {'-':<12}")
            continue

        for hdr_path in hdr_files:
            file_name = os.path.basename(hdr_path)
            img = cv2.imread(hdr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if img is None:
                print(f"{dataset_num:<8} | {file_name[:30]:<32} | {'읽기 실패':<12} | {'-':<12} | {'-':<12}")
                continue

            # 이미지 전체 픽셀값 최소, 최대, 평균
            img_min = np.min(img)
            img_max = np.max(img)
            img_mean = np.mean(img)

            disp_name = file_name if len(file_name) <= 32 else file_name[:29] + "..."
            print(f"{dataset_num:<8} | {disp_name:<32} | {img_min:<12.2f} | {img_max:<12.2f} | {img_mean:<12.2f}")

    print("=" * 70)

if __name__ == "__main__":
    main()
