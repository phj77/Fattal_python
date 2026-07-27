# exe_gf_fattal.py - Guided Filter + Fattal Base Layer Tone Mapping 실험 (RAW Input)
# 파이프라인: RAW Image → Guided Filter (base/detail 분리) → Fattal on base → 합성 → exp → 8bit RGB JPG
import cv2
import numpy as np
import os
import glob
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(exp_dir))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from experiment.input_raw_base_detail_seperate.fattal.gf_fattal_tmo import tmo_fattal02_logdomain
from experiment.input_raw_base_detail_seperate.config.config import (
    INPUT_DIR, OUTPUT_DIR, CROP_Y_RANGE, CROP_X_RANGE, get_parameter_combinations
)
import utils.utils as utils

from concurrent.futures import ThreadPoolExecutor


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Guided Filter 구현 함수.
    cv2.ximgproc가 존재하면 호환 함수를 사용하고, 없을 경우 cv2.boxFilter 기반으로 동작합니다.
    """
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        return cv2.ximgproc.guidedFilter(
            guide=guide.astype(np.float32),
            src=src.astype(np.float32),
            radius=radius,
            eps=eps
        ).astype(guide.dtype)

    guide_64 = guide.astype(np.float64)
    src_64 = src.astype(np.float64)

    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide_64, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(src_64, cv2.CV_64F, ksize)
    mean_Ip = cv2.boxFilter(guide_64 * src_64, cv2.CV_64F, ksize)
    mean_II = cv2.boxFilter(guide_64 * guide_64, cv2.CV_64F, ksize)

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    q = mean_a * guide_64 + mean_b
    return q.astype(guide.dtype)


def load_raw_image(img_path: str) -> np.ndarray:
    """
    raw 데이터 파일을 읽어 numpy float32 배열로 변환합니다.
    - 파일 크기가 24MiB (25,165,824 bytes) 이면: reshape((2048, 3072))
    - 파일 크기가 36MiB (37,748,736 bytes) 이면: reshape((3072, 3072))
    """
    file_size = os.path.getsize(img_path)
    size_24mib = 24 * 1024 * 1024  # 25,165,824 bytes
    size_36mib = 36 * 1024 * 1024  # 37,748,736 bytes

    if file_size == size_24mib:
        shape = (2048, 3072)
    elif file_size == size_36mib:
        shape = (3072, 3072)
    else:
        num_floats = file_size // 4
        if num_floats == 2048 * 3072:
            shape = (2048, 3072)
        elif num_floats == 3072 * 3072:
            shape = (3072, 3072)
        else:
            raise ValueError(
                f"지원하지 않는 raw 파일 크기입니다: {file_size} bytes ({file_size / (1024 * 1024):.2f} MiB). "
                f"24MiB (2048x3072) 또는 36MiB (3072x3072) 이어야 합니다."
            )

    img = np.fromfile(img_path, dtype=np.float32).reshape(shape)
    return img


def main():
    utils.start_timer()
    utils.print_elapsed("Guided Filter + Fattal RAW 실험 실행 시작")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 입력 경로가 단일 파일인지 폴더인지 구분하여 .raw 파일 목록 생성
    raw_files = []
    if os.path.isfile(INPUT_DIR):
        if INPUT_DIR.lower().endswith('.raw'):
            raw_files.append(INPUT_DIR)
        base_input_dir = os.path.dirname(INPUT_DIR)
    elif os.path.isdir(INPUT_DIR):
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith('.raw'):
                    raw_files.append(os.path.join(root, file))
        base_input_dir = INPUT_DIR
    else:
        print(f"오류: '{INPUT_DIR}' 경로가 존재하지 않습니다.")
        return

    if not raw_files:
        print(f"경고: '{INPUT_DIR}'에서 .raw 파일을 찾을 수 없습니다.")
        return

    param_combinations = get_parameter_combinations()
    total_tasks = len(raw_files) * len(param_combinations)

    utils.print_elapsed("구간 1 (환경 설정 및 파일 탐색 완료)")
    print(f"총 {len(raw_files)}개의 RAW 이미지와 {len(param_combinations)}개의 파라미터 조합이 감지되었습니다.")
    if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
        print(f"크롭 범위 - Y축: {CROP_Y_RANGE}, X축: {CROP_X_RANGE}")
    print(f"총 {total_tasks}회의 Guided Filter + Fattal 톤 매핑 작업이 시작됩니다.\n")

    # 2. 각 이미지에 대하여 반복 실행
    for img_path in raw_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 원본 INPUT_DIR 기준 상대 경로 계산 (폴더 구조 유지)
        rel_path = os.path.relpath(img_path, base_input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_out_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(target_out_dir, exist_ok=True)

        try:
            img_single = load_raw_image(img_path)
        except Exception as e:
            print(f"오류: RAW 이미지 읽기 실패 ({img_path}) - {e}")
            continue

        # 이미지 크롭 적용
        crop_suffix = ""
        if CROP_Y_RANGE is not None or CROP_X_RANGE is not None:
            h, w = img_single.shape
            ymin, ymax = CROP_Y_RANGE if CROP_Y_RANGE is not None else (0, h)
            xmin, xmax = CROP_X_RANGE if CROP_X_RANGE is not None else (0, w)
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            img_single = img_single[ymin:ymax, xmin:xmax]
            crop_suffix = f"_cropY{ymin}-{ymax}_X{xmin}-{xmax}"
            utils.print_elapsed(f"구간 2.5 (이미지 크롭 완료: Y[{ymin}:{ymax}], X[{xmin}:{xmax}])")
        else:
            h, w = img_single.shape
            utils.print_elapsed(f"구간 2 (RAW 이미지 로드 완료) - 대상: {rel_path} (해상도: {w}x{h})")

        # 3. 각 파라미터 조합에 대하여 반복 실행
        for p in param_combinations:
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
            base_layer = guided_filter(
                guide=H, 
                src=H, 
                radius=p['gf_radius'], 
                eps=p['gf_eps']
            )
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
            scaled_detail_layer = detail_factor * detail_layer
            combined = tone_mapped_base + scaled_detail_layer
            utils.print_elapsed(f"구간 3.4 (합성 완료: detail_factor={detail_factor})")

            # ── Step 5: Exp & 정규화 → 계층별 시각화 (합산/곱셈 물리 관계 반영) ──
            L = np.exp(combined)
            L_base = np.exp(tone_mapped_base)
            
            # 백분위수 기반 정규화 기준값 (min_val, max_val)은 L (= np.exp(combined)) 기준 (0.1% ~ 99.5%)
            cut_min = 0.01 * 0.1
            cut_max = 1.0 - 0.01 * 0.5

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_min = executor.submit(np.percentile, L, cut_min * 100)
                future_max = executor.submit(np.percentile, L, cut_max * 100)
                min_val = future_min.result()
                max_val = future_max.result()

            # 1) 최종 LDR 및 Base LDR (동일 L의 min_val, max_val 휘도 스케일 공유)
            def normalize_luminance(exp_layer: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
                if max_v > min_v:
                    norm = (exp_layer - min_v) / (max_v - min_v)
                else:
                    norm = np.zeros_like(exp_layer)
                norm = np.clip(norm, 0.0, 1.0)
                img_8bit = (norm * 255.0).astype(np.uint8)
                return cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

            out_img_rgb = normalize_luminance(L, min_val, max_val)
            tone_mapped_base_img_rgb = normalize_luminance(L_base, min_val, max_val)

            # 2) Detail 승수 비율 지도 (M_detail = exp(scaled_detail_layer), M=1.0/Gray=128 중심)
            # M_detail = exp(scaled_detail_layer) 에서 변화 없음(M=1.0)을 0.5 (Gray 128)에 맞춤
            M_detail = np.exp(scaled_detail_layer)
            m_dev = M_detail - 1.0
            m_max = max(abs(np.percentile(m_dev, 0.1)), abs(np.percentile(m_dev, 99.9)))
            if m_max < 1e-6:
                m_max = 1.0
            norm_m = 0.5 + 0.5 * (m_dev / m_max)
            norm_m = np.clip(norm_m, 0.0, 1.0)
            detail_multiplier_img_rgb = cv2.cvtColor((norm_m * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # 3) 실제 휘도 차이 기여도 (Delta L = out_img_rgb - tone_mapped_base_img_rgb, 정규화 없이 128 기준 직관 시각화)
            diff_ldr = out_img_rgb.astype(np.int16) - tone_mapped_base_img_rgb.astype(np.int16)
            delta_L_img_rgb = np.clip(128 + diff_ldr, 0, 255).astype(np.uint8)

            utils.print_elapsed("구간 3.5 (Exp, 물리적 계층별 분리 시각화 8bit RGB 변환 완료)")

            # ── Step 6: 파일 저장 ──
            param_suffix = (
                f"a{p['opt_alpha']}_b{p['opt_beta']}_dl{p['detail_level']}"
                f"_gfr{p['gf_radius']}_gfe{p['gf_eps']}_df{detail_factor}"
                f"{crop_suffix}"
            )

            if len(param_combinations) == 1 and not crop_suffix:
                output_folder = target_out_dir
            else:
                output_folder = os.path.join(target_out_dir, param_suffix)
                os.makedirs(output_folder, exist_ok=True)

            save_path_combined = os.path.join(output_folder, f"{file_name}.jpg")
            save_path_tone_mapped_base = os.path.join(output_folder, f"{file_name}_tone_mapped_base.jpg")
            save_path_detail_multiplier = os.path.join(output_folder, f"{file_name}_detail_multiplier.jpg")
            save_path_delta_L = os.path.join(output_folder, f"{file_name}_delta_L.jpg")

            cv2.imwrite(save_path_combined, out_img_rgb)
            cv2.imwrite(save_path_tone_mapped_base, tone_mapped_base_img_rgb)
            cv2.imwrite(save_path_detail_multiplier, detail_multiplier_img_rgb)
            cv2.imwrite(save_path_delta_L, delta_L_img_rgb)

            print(f"완료: {save_path_combined} (합성 결과 및 물리적 기여도 분리 시각화 이미지 3종 저장 완료)")
            utils.print_elapsed("구간 4 (파일 저장 완료)")
    
    utils.print_elapsed("Guided Filter + Fattal RAW 프로그램 전체 종료")

if __name__ == "__main__":
    main()
