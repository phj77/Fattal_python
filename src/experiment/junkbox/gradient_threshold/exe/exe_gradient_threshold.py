import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# sys.path 설정하여 상위 모듈 참조
CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CURRENT_DIR.parent
EXPERIMENTS_ROOT = EXPERIMENT_DIR.parent
SRC_DIR = EXPERIMENTS_ROOT.parent.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.junkbox.gradient_threshold.core.gradient_mask import load_hdr_image, generate_low_gradient_mask

def visualize_gradient_threshold(img_path: Path, threshold: float = 5.0):
    """
    단일 HDR 이미지에 대해 Log domain을 사용하지 않고 원본 Luminance 상에서
    Gradient Magnitude <= threshold 인 픽셀을 하얗게(255), 나머지를 검게(0) 표시한 마스크를
    저장 없이 matplotlib 팝업 창(plt.show())으로 즉시 보여줍니다.
    """
    print(f"\n[Processing] {img_path.name}")
    
    # 1. 이미지 로드 및 Luminance 계산
    img_rgb, Y = load_hdr_image(img_path)
    
    # 2. Gradient Magnitude 및 Threshold 마스크 생성
    binary_mask, grad_mag, stats = generate_low_gradient_mask(Y, threshold=threshold)
    
    print(f"  - Image size: {Y.shape[1]} x {Y.shape[0]}")
    print(f"  - Min Grad: {stats['min_grad']:.4f}, Max Grad: {stats['max_grad']:.4f}, Mean Grad: {stats['mean_grad']:.4f}")
    print(f"  - Grad <= {threshold} pixels: {stats['low_grad_count']} / {stats['total_pixels']} ({stats['low_grad_ratio_pct']:.2f}%)")
    
    # 3. 팝업 창에 시각화 (Tone mapping 없이 톤 표현을 위해 간단한 log preview 또는 raw luminance 표시)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title(f"Gradient Threshold Visualization - {img_path.name}")
    
    # (1) 원본 Luminance (Tone preview용)
    lum_display = np.log1p(Y)  # 뷰어 확인용 단순 log 디스플레이
    axes[0].imshow(lum_display, cmap='gray')
    axes[0].set_title(f"Original Luminance (Log Preview)\n{img_path.name}", fontsize=12)
    axes[0].axis('off')
    
    # (2) Raw Gradient Magnitude Heatmap
    # 상위 99% 값으로 vmin, vmax 클리핑하여 보기 쉽게 표현
    vmax = float(np.percentile(grad_mag, 99)) if np.percentile(grad_mag, 99) > 0 else float(np.max(grad_mag))
    im = axes[1].imshow(grad_mag, cmap='jet', vmin=0, vmax=vmax)
    axes[1].set_title(f"Gradient Magnitude (Raw Y)\nMax(99%): {vmax:.2f}", fontsize=12)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # (3) Gradient <= 5.0 Threshold Binary Mask (White: <= 5, Black: > 5)
    axes[2].imshow(binary_mask, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Low Gradient Mask (Grad <= {threshold})\nWhite: <= {threshold} ({stats['low_grad_ratio_pct']:.1f}%)", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()  # 저장 없이 바로 팝업 창으로 보여줌

def main():
    # 기본 입력 디렉토리: PROJECT_ROOT / "data" / "data_one"
    data_one_dir = PROJECT_ROOT / "data" / "data_one"
    
    # data_one 내부의 모든 subfolder(1, 2, 3...)에 있는 .hdr 파일 수집
    hdr_files = sorted(list(data_one_dir.glob("**/*.hdr")))
    
    if not hdr_files:
        print(f"[경고] {data_one_dir} 디렉토리에서 .hdr 파일을 찾지 못했습니다.")
        return
        
    print(f"총 {len(hdr_files)}개의 HDR 파일을 찾았습니다.")
    print("각 이미지의 Gradient <= 5.0 마스크를 팝업 창으로 시각화합니다.\n (창을 닫으면 다음 이미지가 표시됩니다.)")
    
    threshold_val = 0.0
    for img_path in hdr_files:
        visualize_gradient_threshold(img_path, threshold=threshold_val)

if __name__ == "__main__":
    main()
