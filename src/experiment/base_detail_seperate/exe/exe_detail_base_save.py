# exe_detail_base_save.py - Guided Filter + Fattal Base Layer Tone Mapping 실험 (L, Base, Detail 분리 저장)
# 파이프라인: HDR → log → Guided Filter (base/detail 분리) → Fattal on base → 합성 → exp → 8bit LDR
import cv2
import numpy as np
import os
import glob
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
# exe/ → base_detail_seperate/ → experiment/ → src/
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))

# src/ 추가 (fattal.*, utils.* 사용)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 실험 로컬 모듈 경로 추가 (config/, fattal/ 하위 모듈)
config_dir = os.path.join(exp_dir, 'config')
fattal_dir = os.path.join(exp_dir, 'fattal')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
if fattal_dir not in sys.path:
    sys.path.insert(0, fattal_dir)
# ─────────────────────────────────────────────────────────────────────────────

# 실험 전용 Fattal log domain 함수
from gf_fattal_tmo import tmo_fattal02_logdomain

# 실험 전용 설정
from config import INPUT_DIR, OUTPUT_DIR, CROP_Y_RANGE, CROP_X_RANGE, get_parameter_combinations

import utils.utils as utils

from concurrent.futures import ThreadPoolExecutor


def main():
    utils.start_timer()
    utils.print_elapsed("Guided Filter + Fattal 실험 실행 시작 (L, Base, Detail 분리 저장)")
    
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

    utils.print_elapsed("구간 1 (환경 설정 및 파일 탐색 완료)")
    print(f"총 {len(hdr_files)}개의 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        print(f"크롭 범위 - Y축: {CROP_Y_RANGE}, X축: {CROP_X_RANGE}")
    print(f"총 {total_tasks}회의 Guided Filter + Fattal 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 이미지에 대하여 반복 실행
    for img_path in hdr_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 이미지 로드
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if img is None:
            print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
            continue

        # 같은 intensity를 가진 3채널 이미지에서 1채널만 추출하여 사용
        if img.ndim == 3:
            img_single = img[:, :, 0]
        else:
            img_single = img

        # 이미지 크롭 적용
        crop_suffix = ""
        if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
            h, w = img_single.shape
            ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h)
            xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w)
            # 안전성 경계값 클리핑
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            img_single = img_single[ymin:ymax, xmin:xmax]
            crop_suffix = f"_cropY{ymin}-{ymax}_X{xmin}-{xmax}"
            utils.print_elapsed(f"구간 2.5 (이미지 크롭 완료: Y[{ymin}:{ymax}], X[{xmin}:{xmax}])")
        else:
            utils.print_elapsed(f"구간 2 (이미지 로드 완료) - 대상: {file_name}")

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for p in param_combinations:
            # fftsolver 사용 시 newfattal 강제 활성화
            newfattal = p['newfattal']
            if p['fftsolver']:
                newfattal = True
            opt_noise = p['opt_noise']
            if opt_noise <= 0.0:
                opt_noise = p['opt_alpha'] * 0.01

            # ── Step 1: Log domain 변환 ──
            Y = img_single.astype(np.float64)
            maxLum = np.max(Y)
            H = np.log(100.0 * Y / maxLum + 1e-4)
            utils.print_elapsed("구간 3.1 (Log domain 변환 완료)")

            # ── Step 2: Guided Filter로 base/detail 분리 ──
            H_float32 = H.astype(np.float32)
            base_layer = cv2.ximgproc.guidedFilter(
                guide=H_float32, 
                src=H_float32, 
                radius=p['gf_radius'], 
                eps=p['gf_eps']
            ).astype(np.float64)
            detail_layer = H - base_layer
            utils.print_elapsed(f"구간 3.2 (Guided Filter 분리 완료: r={p['gf_radius']}, eps={p['gf_eps']})")

            # ── Step 3: Base layer에 Fattal TMO 적용 (log domain) ──
            tone_mapped_base = tmo_fattal02_logdomain(
                base_layer,
                p['opt_alpha'], p['opt_beta'], opt_noise,
                newfattal, p['fftsolver'], p['detail_level'],
                hpf_sigma=p.get('hpf_sigma', 0.007)
            )
            utils.print_elapsed("구간 3.3 (Fattal TMO on base layer 완료)")

            # ── Step 4: 합성 (tone mapped base + detail_factor * detail) ──
            detail_factor = p['detail_factor']
            combined = tone_mapped_base + detail_factor * detail_layer
            utils.print_elapsed(f"구간 3.4 (합성 완료: detail_factor={detail_factor})")

            # ── Step 5: Exp & 정규화 → 8bit LDR ──
            L = np.exp(combined)
            L_base = np.exp(tone_mapped_base)
            L_detail = np.exp(detail_factor * detail_layer)

            # 백분위수 기반 정규화 (0.1% ~ 99.5%)
            cut_min = 0.01 * 0.1
            cut_max = 1.0 - 0.01 * 0.5

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_min = executor.submit(np.percentile, L, cut_min * 100)
                future_max = executor.submit(np.percentile, L, cut_max * 100)
                min_val = future_min.result()
                max_val = future_max.result()

            # L의 min_val, max_val을 사용하여 정규화
            L = (L - min_val) / (max_val - min_val)
            L = np.clip(L, 0.0, 1.0)
            out_img_8bit = (L * 255.0).astype(np.uint8)

            L_base = (L_base - min_val) / (max_val - min_val)
            L_base = np.clip(L_base, 0.0, 1.0)
            out_base_8bit = (L_base * 255.0).astype(np.uint8)

            L_detail = (L_detail - min_val) / (max_val - min_val)
            L_detail = np.clip(L_detail, 0.0, 1.0)
            out_detail_8bit = (L_detail * 255.0).astype(np.uint8)

            utils.print_elapsed("구간 3.5 (Exp, 정규화, 8bit 변환 완료)")

            # ── Step 6: 파일 저장 ──
            param_suffix = (
                f"a{p['opt_alpha']}_b{p['opt_beta']}_dl{p['detail_level']}"
                f"_gfr{p['gf_radius']}_gfe{p['gf_eps']}_df{detail_factor}"
                f"{crop_suffix}"
            )

            save_name_L = f"{file_name}_{param_suffix}_L.png"
            save_path_L = os.path.join(OUTPUT_DIR, save_name_L)
            cv2.imwrite(save_path_L, out_img_8bit)

            save_name_base = f"{file_name}_{param_suffix}_base.png"
            save_path_base = os.path.join(OUTPUT_DIR, save_name_base)
            cv2.imwrite(save_path_base, out_base_8bit)

            save_name_detail = f"{file_name}_{param_suffix}_detail.png"
            save_path_detail = os.path.join(OUTPUT_DIR, save_name_detail)
            cv2.imwrite(save_path_detail, out_detail_8bit)

            print(f"완료: {save_path_L}")
            print(f"완료: {save_path_base}")
            print(f"완료: {save_path_detail}")
            utils.print_elapsed("구간 4 (파일 저장 완료)")
    
    utils.print_elapsed("프로그램 전체 종료")

if __name__ == "__main__":
    main()
