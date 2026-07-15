# exe_best_params.py (Guided Filter + Fattal Base Layer - Best Parameters)
import cv2
import numpy as np
import os
import glob
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
# exe/ → base_detail_seperate/ → experiment/ → src/
current_file = Path(__file__).resolve()
exe_dir = current_file.parent
exp_dir = exe_dir.parent
project_root = exp_dir.parents[2]  # Fattal_python/
src_dir = project_root / "src"

# src/ 추가 (fattal.*, utils.* 사용)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 실험 로컬 모듈 경로 추가 (config/, fattal/ 하위 모듈)
config_dir = exp_dir / 'config'
fattal_dir = exp_dir / 'fattal'
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))
if str(fattal_dir) not in sys.path:
    sys.path.insert(0, str(fattal_dir))
# ─────────────────────────────────────────────────────────────────────────────

# 실험 전용 Fattal log domain 함수
from gf_fattal_tmo import tmo_fattal02_logdomain
# 실험 전용 설정 (크롭 설정을 공유)
from config import CROP_Y_RANGE, CROP_X_RANGE
import utils.utils as utils

def main():
    utils.start_timer()
    utils.print_elapsed("Base-Detail Separate Fattal best_params 실행 시작")
    
    # 출력 루트 디렉토리 설정
    output_base_dir = project_root / "experiment_result" / "base_detail_seperate" / "best_params"

    # 데이터셋 번호와 이에 상응하는 최적 파라미터 설정
    dataset_params = {
        1: {'alpha': 0.9, 'beta': 0.82, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        2: {'alpha': 0.9, 'beta': 0.80, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        3: {'alpha': 0.9, 'beta': 0.81, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        4: {'alpha': 0.9, 'beta': 0.84, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        5: {'alpha': 0.3, 'beta': 0.93, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        6: {'alpha': 0.9, 'beta': 0.81, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0},
        7: {'alpha': 0.9, 'beta': 0.80, 'gf_radius': 10, 'gf_eps': 0.01, 'detail_factor': 5.0}
    }
    
    # 기본 고정 파라미터들
    opt_noise = 0.001
    newfattal = True
    fftsolver = True
    detail_level = 0
    hpf_sigma = 0.007

    print(f"출력 베이스 디렉토리: {output_base_dir}\n")

    # 1부터 7까지의 데이터셋 폴더를 순회하며 톤 매핑 실행
    for ds_num in sorted(dataset_params.keys()):
        p = dataset_params[ds_num]
        
        # 출력 폴더: experiment_result/base_detail_seperate/best_params/{ds_num}/
        ds_output_dir = output_base_dir / str(ds_num)
        if not ds_output_dir.exists():
            ds_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"데이터셋 [{ds_num}] 출력 디렉토리 생성됨: {ds_output_dir}")
        
        # fftsolver 사용 시 newfattal 강제 활성화 및 opt_noise 보정
        ds_newfattal = newfattal
        if fftsolver:
            ds_newfattal = True
            
        ds_opt_noise = opt_noise
        if ds_opt_noise <= 0.0:
            ds_opt_noise = p['alpha'] * 0.01

        ds_dir = project_root / "data" / "data_one" / str(ds_num)
        
        search_pattern = str(ds_dir / "*.hdr")
        hdr_files = glob.glob(search_pattern)
        
        if not hdr_files:
            print(f"경고: '{ds_dir}' 디렉토리에서 .hdr 파일을 찾을 수 없습니다.")
            continue
            
        for img_path in hdr_files:
            file_name = Path(img_path).stem
            print(f"\n--- 데이터셋 [{ds_num}] 처리 중 ---")
            print(f"이미지: {file_name}")
            print(f"파라미터 - alpha: {p['alpha']}, beta: {p['beta']}, gf_radius: {p['gf_radius']}, gf_eps: {p['gf_eps']}, detail_factor: {p['detail_factor']}")
            
            # 이미지 로드
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"오류: 이미지를 읽을 수 없습니다 - {img_path}")
                continue

            if img.ndim == 3:
                img_single = img[:,:,0]
            else:
                img_single = img

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
                print(f"크롭 적용: Y[{ymin}:{ymax}], X[{xmin}:{xmax}]")

            # ── Step 1: Log domain 변환 ──
            Y = img_single.astype(np.float64)
            maxLum = np.max(Y)
            H = np.log(100.0 * Y / maxLum + 1e-4)

            # ── Step 2: Guided Filter로 base/detail 분리 ──
            H_float32 = H.astype(np.float32)
            base_layer = cv2.ximgproc.guidedFilter(
                guide=H_float32, 
                src=H_float32, 
                radius=p['gf_radius'], 
                eps=p['gf_eps']
            ).astype(np.float64)
            detail_layer = H - base_layer

            # ── Step 3: Base layer에 Fattal TMO 적용 (log domain) ──
            tone_mapped_base = tmo_fattal02_logdomain(
                base_layer,
                p['alpha'], p['beta'], ds_opt_noise,
                ds_newfattal, fftsolver, detail_level,
                hpf_sigma=hpf_sigma
            )

            # ── Step 4: 합성 (tone mapped base + detail_factor * detail) ──
            detail_factor = p['detail_factor']
            combined = tone_mapped_base + detail_factor * detail_layer

            # ── Step 5: Exp & 정규화 → 8bit LDR ──
            L = np.exp(combined)

            # 백분위수 기반 정규화 (0.1% ~ 99.5%)
            cut_min = 0.01 * 0.1
            cut_max = 1.0 - 0.01 * 0.5

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_min = executor.submit(np.percentile, L, cut_min * 100)
                future_max = executor.submit(np.percentile, L, cut_max * 100)
                min_val = future_min.result()
                max_val = future_max.result()

            L = (L - min_val) / (max_val - min_val)
            L = np.clip(L, 0.0, 1.0)

            out_img_8bit = (L * 255.0).astype(np.uint8)

            # 결과물 파일 저장 (데이터셋 번호 및 파라미터 정보를 명시)
            param_suffix = f"a{p['alpha']}_b{p['beta']}_gfr{p['gf_radius']}_gfe{p['gf_eps']}_df{p['detail_factor']}{crop_suffix}"
            save_name = f"{ds_num}_{file_name}_{param_suffix}.png"
            save_path = ds_output_dir / save_name

            cv2.imwrite(str(save_path), out_img_8bit)
            print(f"저장 완료: {save_path}")
            
    utils.print_elapsed("Base-Detail Separate Fattal best_params 프로그램 전체 종료")

if __name__ == "__main__":
    main()
