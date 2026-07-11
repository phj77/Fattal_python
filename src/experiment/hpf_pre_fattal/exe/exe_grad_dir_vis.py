# exe_grad_dir_vis.py
# Gaussian Pyramid level-wise Gradient Direction Visualization Script for Pre-HPF + Fattal TMO pipeline.

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# --- Path setup ---
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3]  # .../Fattal_python/src
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from experiment.hpf_pre_fattal.fattal.fattal_tmo import (
    apply_high_pass_filter,
    createGaussianPyramids
)
from experiment.hpf_pre_fattal.config.config import INPUT_DIR, OUTPUT_DIR, PARAM_GRID
import utils.utils as utils

# ─── 실험 전용 설정 ─────────────────────────────────────────────────────────
PRE_HPF_SIGMA = PARAM_GRID.get('pre_hpf_sigma', [0.010])[0]
# ─────────────────────────────────────────────────────────────────────────────

# --- Parameters (from config.py) ---
opt_alpha = PARAM_GRID['opt_alpha'][0]
opt_beta = PARAM_GRID['opt_beta'][0]
opt_noise = PARAM_GRID['opt_noise'][0]
fftsolver = PARAM_GRID['fftsolver'][0]

MSIZE = 8 if fftsolver else 32


def calculate_gradient_components(H: np.ndarray, k: int):
    """
    각 피라미드 레벨 H에 대해 x, y 방향 Gradient (gx, gy)를 계산합니다.
    
    Args:
        H (np.ndarray): 피라미드 레벨 2D 배열 (log luminance)
        k (int): 피라미드 레벨 인덱스
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (gx, gy, gradient magnitude)
    """
    divider = 2.0 ** (k + 1)
    
    gx = np.empty_like(H, dtype=np.float32)
    gx[:, 0] = (H[:, 1] - H[:, 0]) / divider
    gx[:, -1] = (H[:, -1] - H[:, -2]) / divider
    gx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2.0 * divider)
    
    gy = np.empty_like(H, dtype=np.float32)
    gy[0, :] = (H[1, :] - H[0, :]) / divider
    gy[-1, :] = (H[-1, :] - H[-2, :]) / divider
    gy[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2.0 * divider)

    grad_mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, grad_mag


def save_gradient_direction_color_wheel(save_path: Path):
    """
    HSV 색상과 Gradient 방향(각도)의 매핑 관계를 보여주는 2D 색상 환(Color Wheel) 폼 이미지를 생성 및 저장합니다.
    """
    size = 400
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    radius = np.sqrt(x**2 + y**2)
    max_r = size // 2 - 20
    
    mask = radius <= max_r
    angle_rad = np.arctan2(y, x)  # [-pi, pi]
    
    # Map angle to HSV
    # Hue: [0, 179]
    hue = (((angle_rad + np.pi) / (2.0 * np.pi)) * 179.0).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = np.full_like(hue, 255, dtype=np.uint8)
    
    hsv = np.stack([hue, sat, val], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[~mask] = 255  # 배경을 흰색으로 설정
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Gradient Direction Color Wheel Reference", fontsize=13, fontweight='bold', pad=15)
    
    # 주요 방향 주석 표기
    cx, cy = size // 2, size // 2
    r_text = max_r + 12
    ax.text(cx + r_text, cy, "0° (Right →)", va='center', ha='left', fontsize=10, fontweight='bold', color='red')
    ax.text(cx, cy + r_text, "90° (Down ↓)", va='top', ha='center', fontsize=10, fontweight='bold', color='goldenrod')
    ax.text(cx - r_text, cy, "±180° (Left ←)", va='center', ha='right', fontsize=10, fontweight='bold', color='teal')
    ax.text(cx, cy - r_text, "-90° (Up ↑)", va='bottom', ha='center', fontsize=10, fontweight='bold', color='blue')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_gradient_direction_map(angle_deg: np.ndarray, level: int, save_path: Path, title: str = None):
    """
    각 피라미드 레벨의 Gradient 방향 각도(-180° ~ 180°)를 HSV/Twilight 순환 색상 맵 및 틱 라벨로 시각화합니다.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 'hsv' 순환 컬러맵 적용 (각도 -180° ~ 180°)
    im = ax.imshow(angle_deg, cmap='hsv', vmin=-180, vmax=180, aspect='auto')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Gradient Direction Angle (Degrees)', fontsize=12)
    cbar.set_ticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    cbar.set_ticklabels([
        '-180° (← Left)',
        '-135° (↖ Up-Left)',
        '-90° (↑ Up)',
        '-45° (↗ Up-Right)',
        '0° (→ Right)',
        '45° (↘ Down-Right)',
        '90° (↓ Down)',
        '135° (↙ Down-Left)',
        '180° (← Left)'
    ])

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Gradient Direction Map - Pyramid Level {level}', fontsize=14, fontweight='bold')

    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)

    h, w = angle_deg.shape
    stats_text = (
        f'Size: {w}x{h}  |  '
        f'Min Angle: {angle_deg.min():.1f}°  |  '
        f'Max Angle: {angle_deg.max():.1f}°  |  '
        f'Mean Angle: {angle_deg.mean():.1f}°'
    )
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color='gray')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_gradient_quiver(H_k: np.ndarray, gx: np.ndarray, gy: np.ndarray, level: int, save_path: Path):
    """
    각 피라미드 레벨의 Gradient 방향을 화살표(Quiver Vector Field)로 서브샘플링하여 시각화합니다.
    """
    h, w = H_k.shape
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 배경으로 이미지 명암 표시 (그레이스케일)
    ax.imshow(H_k, cmap='gray', aspect='auto')
    
    # 서브샘플링 간격 설정 (약 30~40개 화살표 표시)
    step_y = max(1, h // 35)
    step_x = max(1, w // 35)
    
    y_indices = np.arange(0, h, step_y)
    x_indices = np.arange(0, w, step_x)
    xx, yy = np.meshgrid(x_indices, y_indices)
    
    sub_gx = gx[yy, xx]
    sub_gy = gy[yy, xx]
    mag = np.sqrt(sub_gx**2 + sub_gy**2) + 1e-8
    
    # 방향 표시를 위해 벡터 단위화 (크기 고정, 방향만 표현)
    u_unit = sub_gx / mag
    v_unit = sub_gy / mag
    
    # Matplotlib quiver: v_unit positive direction points downward on image axis
    ax.quiver(xx, yy, u_unit, v_unit, color='cyan', angles='xy', scale_units='xy', scale=0.08, width=0.003, headwidth=4)
    
    ax.set_title(f'Gradient Direction Vector Field (Quiver) - Level {level} ({w}x{h})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_gradient_direction_hsv_opencv(angle_rad: np.ndarray, save_path: Path):
    """
    Gradient 방향 각도를 HSV 도메인의 Hue 채널로 변환하여 OpenCV BGR 8비트 이미지로 직접 저장합니다.
    (Hue: 방향 각도 [-pi, pi] -> [0, 179], Saturation=255, Value=255)
    """
    # angle_rad: [-pi, pi] -> [0, 179]
    hue = (((angle_rad + np.pi) / (2.0 * np.pi)) * 179.0).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = np.full_like(hue, 255, dtype=np.uint8)

    hsv = np.stack([hue, sat, val], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(save_path), bgr)


def process_single_image(img_path: Path, data_folder_name: str, base_output_dir: Path):
    """
    단일 HDR 이미지에 대해 Pre-HPF 적용, 가우시안 피라미드 생성, 각 레벨별 Gradient 방향 시각화 및 저장 수행
    """
    file_name = img_path.stem
    save_dir = base_output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- [Pre-HPF {PRE_HPF_SIGMA}] Gradient Direction Vis Image: {file_name} ---")

    # 1. 이미지 읽기
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"  [오류] 이미지를 읽을 수 없습니다: {img_path}")
        return

    if img.ndim == 3:
        img_single = img[:, :, 0]
    else:
        img_single = img

    # 2. Pre-HPF 적용
    img_filtered = apply_high_pass_filter(img_single, pre_hpf_sigma=PRE_HPF_SIGMA)

    # 3. 로그 공간 변환
    maxLum = np.max(img_filtered)
    H = np.log(100.0 * img_filtered / maxLum + 1e-4)

    # 4. 가우시안 피라미드 생성
    h, w = H.shape
    mins = min(w, h)
    nlevels = 0
    temp_mins = mins
    while temp_mins >= MSIZE:
        nlevels += 1
        temp_mins //= 2
    if nlevels == 0:
        nlevels = 1

    pyramids = createGaussianPyramids(H, nlevels)
    print(f"  피라미드 레벨 수: {nlevels}  (원본 크기: {w}x{h})")

    # 4-1. 색상 레퍼런스 Color Wheel 저장
    color_wheel_path = save_dir / "gradient_direction_color_wheel.png"
    if not color_wheel_path.exists():
        save_gradient_direction_color_wheel(color_wheel_path)
        print(f"  [범례 저장] {color_wheel_path.name}")

    # 5. 각 피라미드 레벨별 Gradient 방향 계산 및 시각화
    for k in range(nlevels):
        H_k = pyramids[k]
        ph, pw = H_k.shape

        # Gradient 성분 및 방향 각도 계산
        gx, gy, grad_mag = calculate_gradient_components(H_k, k)
        angle_rad = np.arctan2(gy, gx)              # [-pi, pi]
        angle_deg = np.degrees(angle_rad)           # [-180, 180]

        # (1) Matplotlib Colormap Gradient Direction 시각화 저장
        plt_save_name = f"grad_dir_level_{k:02d}_{pw}x{ph}.png"
        plt_save_path = save_dir / plt_save_name
        title = f'Pre-HPF({PRE_HPF_SIGMA}) Gradient Direction Map - Level {k}/{nlevels-1} ({pw}x{ph})'
        visualize_gradient_direction_map(angle_deg, k, plt_save_path, title=title)

        # (2) Quiver Vector Field 화살표 방향 시각화 저장
        quiver_save_name = f"grad_vector_level_{k:02d}_{pw}x{ph}.png"
        quiver_save_path = save_dir / quiver_save_name
        visualize_gradient_quiver(H_k, gx, gy, k, quiver_save_path)

        # (3) OpenCV Direct HSV 이미지 저장
        cv_save_name = f"cv_grad_dir_level_{k:02d}_{pw}x{ph}.png"
        cv_save_path = save_dir / cv_save_name
        save_gradient_direction_hsv_opencv(angle_rad, cv_save_path)

        print(f"    Level {k:2d}: {pw:5d}x{ph:<5d} -> saved: {plt_save_path.name}, {quiver_save_path.name}, {cv_save_path.name}")

    print(f"  [완료] {file_name}의 모든 피라미드 레벨 Gradient 방향 시각화 저장 완료!")


def main():
    utils.start_timer()
    utils.print_elapsed("Pre-HPF Gaussian Pyramid Gradient 방향 시각화 시작")

    input_path = Path(INPUT_DIR)
    base_output_dir = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[오류] 입력 디렉토리가 존재하지 않습니다: {input_path}")
        return

    direct_hdr_files = list(input_path.glob('*.hdr'))
    if direct_hdr_files:
        print(f"단일 데이터 폴더 처리 중: {input_path.name}")
        for hdr_file in direct_hdr_files:
            process_single_image(hdr_file, input_path.name, base_output_dir)
    else:
        data_folders = sorted([
            d for d in input_path.iterdir() if d.is_dir()
        ], key=lambda x: x.name)

        if not data_folders:
            print(f"[오류] 입력 디렉토리에서 .hdr 파일이나 하위 디렉토리를 찾을 수 없습니다: {input_path}")
            return

        print(f"감지된 데이터 폴더 수: {len(data_folders)}\n")

        for folder in data_folders:
            hdr_files = list(folder.glob('*.hdr'))
            if not hdr_files:
                print(f"[{folder.name}] .hdr 파일이 없어 스킵합니다.")
                continue

            for hdr_file in hdr_files:
                process_single_image(hdr_file, folder.name, base_output_dir)

    utils.print_elapsed("Pre-HPF Gaussian Pyramid Gradient 방향 시각화 완료")
    print(f"\n결과 저장 디렉토리: {base_output_dir}")


if __name__ == "__main__":
    main()
