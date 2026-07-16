import numpy as np
from scipy.stats import rankdata
import cv2
import matplotlib.pyplot as plt

def exact_continuous_he(image, weight = 1):
    """
    정규화된 input이 들어온다.
    weight =1 -> 완전한 equalized 이미지
    양자화 오차 없이 실수형 [0, 1] 이미지에 대한 정확한 히스토그램 평활화를 수행합니다.
    """
    # 1. 이미지를 1차원 배열로 평탄화
    flat_image = image.flatten()
    
    # 2. 각 픽셀 값의 순위 계산 
    # method='average'를 사용하여 동일한 값을 가진 픽셀들에 대해 평균 순위를 부여함
    ranks = rankdata(flat_image, method='average')
    
    # 3. 순위를 [0, 1] 범위로 정규화하여 경험적 CDF 도출
    # 순위는 1부터 시작하므로 1을 빼서 최소값을 0으로 맞춤
    n_pixels = len(flat_image)
    normalized_ranks = (ranks - 1.0) / (n_pixels - 1.0)

    # 가중 합 적용
    weight = np.clip(weight, 0, 1)
    HE_img = weight*normalized_ranks + (1-weight)*flat_image
    
    # 4. 원본 이미지의 차원 형태로 재구성
    equalized_image = HE_img.reshape(image.shape)
    
    return equalized_image


def plot_float_array_histogram(data_array, bins=10000):
    """
    실수형 넘파이 배열을 입력받아 최솟값/최댓값 범위 내에서 히스토그램을 출력하는 함수
    
    Args:
        data_array (np.ndarray): 입력 실수 데이터 배열
        bins (int): 양자화할 구간의 개수
    """
    # 1. 입력 데이터 유효성 검사 및 1차원 평탄화
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)
    
    flat_data = data_array.flatten()
    
    # 2. 데이터 범위 계산
    d_min = np.min(flat_data)
    d_max = np.max(flat_data)
    d_mean = np.mean(flat_data)

    # 3. 시각화
    plt.figure(figsize=(10, 6))
    
    # hist 함수가 내부적으로 min/max를 기준으로 구간을 나눔
    counts, edges, patches = plt.hist(flat_data, bins=bins, color='skyblue', 
                                      edgecolor='black', alpha=0.7)
    
    # 평균선 추가
    plt.axvline(d_mean, color='red', linestyle='dashed', linewidth=1.5, 
                label=f'Mean: {d_mean:.4f}')
    
    # 통계 정보 텍스트 박스 추가
    stats_text = f'Min: {d_min:.4f}\nMax: {d_max:.4f}\nBins: {bins}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title('Histogram for Float Array (Quantized)')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.show()

def plot_gradient_map(Gx, Gy, cut_min, cut_max):
    """
    using Gx, Gy gradient map, plot gradient map.
    you can cut highest {cut_max}% intensity / lowest {cut_min}%
    """
    G_map = cv2.magnitude(Gx, Gy)
    G_min_val = np.percentile(G_map, cut_min * 100)
    G_max_val = np.percentile(G_map, cut_max * 100)

    G_map = np.maximum(G_map, G_min_val)
    G_map = np.minimum(G_map, G_max_val)

    G_map_normalized = cv2.normalize(
        G_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    cv2.imshow('gradient magnitude map', G_map_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def clip_gradient_intensity(Gx, Gy, top_percentile=0.5):
    """
    Gx, Gy 그래디언트 맵을 입력받아 intensity 기준 상위 top_percentile%를 clipping 합니다.
    
    Args:
        Gx (np.ndarray): x방향 그래디언트 배열
        Gy (np.ndarray): y방향 그래디언트 배열
        top_percentile (float): clipping 처리할 상위 퍼센트 (기본값: 0.5)
        
    Returns:
        tuple: clipping이 완료된 (Gx_clipped, Gy_clipped) 형태의 두 배열
    """
    # 1. 그래디언트 벡터의 크기(Intensity) 계산
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # 2. 상위 0.5%에 해당하는 임계값(Threshold) 계산 (하위 99.5% 백분위수)
    threshold = np.percentile(magnitude, 100.0 - top_percentile)
    
    # 3. 스케일링 비율(Scale factor) 계산
    # magnitude가 threshold를 초과하는 경우: threshold / magnitude
    # magnitude가 threshold 이하인 경우: 1.0 (원본 유지)
    # np.maximum을 사용하여 magnitude가 0인 픽셀에서 발생하는 ZeroDivision 오류 차단
    scale = np.where(
        magnitude > threshold, 
        threshold / np.maximum(magnitude, 1e-8), 
        1.0
    )
    
    # 4. 계산된 스케일링 비율을 원본에 곱하여 방향이 보존된 새로운 그래디언트 맵 생성
    Gx_clipped = Gx * scale
    Gy_clipped = Gy * scale
    
    return Gx_clipped, Gy_clipped


def get_top_percentile_threshold(Gx, Gy, top_percentile=0.5):
    """
    그래디언트 magnitude의 상위 percentile에 해당하는 임계값(Threshold)을 반환합니다.
    """
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # np.percentile은 하위 기준이므로, 상위 0.5%는 하위 99.5%로 계산합니다.
    threshold = np.percentile(magnitude, 100.0 - top_percentile)
    
    return threshold

# 글로벌 타이머 전역 변수 및 헬퍼 함수
_start_time = None

def start_timer():
    global _start_time
    import time
    _start_time = time.perf_counter()

def print_elapsed(label):
    global _start_time
    import time
    if _start_time is None:
        _start_time = time.perf_counter()
    elapsed = time.perf_counter() - _start_time
    print(f"{label} : {elapsed:.6f}s")


def save_scanline(image, row_index, stage_name, highlight_ranges=None, save_dir=None, ylim=None):
    """
    특정 행(row)의 스캔라인을 추출하여 고해상도(Fine) 그래프(PNG)와 데이터(NPY)로 저장합니다. (matplotlib 사용)
    
    Args:
        image (np.ndarray): 스캔라인을 추출할 2D 배열 이미지
        row_index (int): 스캔라인을 추출할 행의 인덱스
        stage_name (str): 파일명과 그래프 제목에 들어갈 단계 이름
        highlight_ranges (list of lists/tuples): 하이라이트할 X축 구간들의 리스트 (예: [[500, 600], [1340, 1500]])
        save_dir (str, optional): 저장할 디렉토리 경로. 지정하지 않으면 기본 경로를 사용합니다.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    import numpy as np
    
    # ──────────────────────────────────────────────────────────
    # [사용자 설정 저장 경로] - 이 주소를 직접 수정하여 저장 위치를 지정하세요.
    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Fattal_python 폴더
        save_dir = os.path.join(project_root, "test", "scanline", "scanline_GD")         # Fattal_python/test/scanline/scanline_GD 폴더
    # ──────────────────────────────────────────────────────────
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 행 인덱스가 이미지 범위를 벗어나지 않도록 클리핑
    h = image.shape[0]
    row_index = np.clip(row_index, 0, h - 1)
    
    scanline = image[row_index, :]
    
    # 캔버스 및 서브플롯 설정 (넓고 크게 설정, 고해상도 300 dpi)
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    
    # 얇고 세밀한 파란색 라인으로 플롯
    ax.plot(scanline, color='black', linewidth=0.8, alpha=1)
    
    # 구간 하이라이팅 (axvspan 사용)
    # has_highlight = False
    # if highlight_ranges is not None:
    #     for rng in highlight_ranges:
    #         if len(rng) == 2:
    #             start, end = rng
    #             start = max(0, min(start, len(scanline) - 1))
    #             end = max(0, min(end, len(scanline) - 1))
    #             ax.axvspan(start, end, color='#ffa500', alpha=0.25)
    #             has_highlight = True
                
    # 타이틀 및 라벨 설정
    ax.set_title(f"Scanline Intensity at Row {row_index} - {stage_name}", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Column Index (X)", fontsize=11, labelpad=8)
    ax.set_ylabel("Intensity (Y)", fontsize=11, labelpad=8)
    
    # 보조 눈금(Minor Ticks) 자동 생성 활성화
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 주 그리드(Major Grid)와 보조 그리드(Minor Grid)를 각각 다르게 렌더링
    #ax.grid(True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.6)
    #ax.grid(True, which='minor', color='#e5e5e5', linestyle=':', linewidth=0.4)
    
    # 통계 정보(Min, Max, Mean) 박스를 우측 상단에 표시
    s_min = np.min(scanline)
    s_max = np.max(scanline)
    s_mean = np.mean(scanline)
    stats_text = f"Min: {s_min:.6f}\nMax: {s_max:.6f}\nMean: {s_mean:.6f}"
    
    # 축 범위 여유 설정 (값의 상하 여백 5% 혹은 지정된 ylim)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        y_margin = (s_max - s_min) * 0.05 if s_max > s_min else 1.0
        ax.set_ylim(s_min - y_margin, s_max + y_margin)
    ax.set_xlim(0, len(scanline) - 1)
    
    # 통계 텍스트 상자 배치
    #ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
    #        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.8),
    #        fontsize=9, family='monospace')
            
    # 레이아웃 최적화
    plt.tight_layout()
    
    # 특수문자나 공백이 있을 수 있으니 안전한 파일명으로 처리
    safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    
    file_path_png = os.path.join(save_dir, f"scanline_row{row_index}_{safe_stage_name}.png")
    plt.savefig(file_path_png, dpi=300)
    plt.close()
    
    file_path_npy = os.path.join(save_dir, f"scanline_row{row_index}_{safe_stage_name}.npy")
    np.save(file_path_npy, scanline)
    print(f"[{stage_name}] Scanline at row {row_index} saved to {save_dir}.")


def save_vertical_scanline(image, col_index, stage_name, highlight_ranges=None, save_dir=None, ylim=None):
    """
    특정 열(column)의 세로 방향 스캔라인을 추출하여 고해상도(Fine) 그래프(PNG)와 데이터(NPY)로 저장합니다.
    
    Args:
        image (np.ndarray): 스캔라인을 추출할 2D 배열 이미지
        col_index (int): 스캔라인을 추출할 열의 인덱스
        stage_name (str): 파일명과 그래프 제목에 들어갈 단계 이름
        highlight_ranges (list of lists/tuples): 하이라이트할 Row Index 구간들의 리스트 (예: [[206, 328], [1716, 1815]])
        save_dir (str, optional): 저장할 디렉토리 경로.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    import numpy as np
    
    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Fattal_python 폴더
        save_dir = os.path.join(project_root, "test", "scanline", "scanline_vertical")
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 열 인덱스가 이미지 범위를 벗어나지 않도록 클리핑
    h, w = image.shape[:2]
    col_index = np.clip(col_index, 0, w - 1)
    
    # 세로 방향이므로 col_index를 고정하고 모든 행(row)을 선택
    scanline = image[:, col_index]
    
    # 캔버스 및 서브플롯 설정 (넓고 크게 설정, 고해상도 300 dpi)
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    
    # 검은색 라인으로 플롯
    ax.plot(scanline, color='black', linewidth=0.5, alpha=1)
    
    # 구간 하이라이팅 (axvspan 사용)
    has_highlight = False
    if highlight_ranges is not None:
        for rng in highlight_ranges:
            if len(rng) == 2:
                start, end = rng
                start = max(0, min(start, len(scanline) - 1))
                end = max(0, min(end, len(scanline) - 1))
                ax.axvspan(start, end, color='#ffa500', alpha=0.25)
                has_highlight = True
                
    # 타이틀 및 라벨 설정
    ax.set_title(f"Scanline Intensity at Column {col_index} - {stage_name}", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Row Index (Y)", fontsize=11, labelpad=8)
    ax.set_ylabel("Intensity (Y)", fontsize=11, labelpad=8)
    
    # 보조 눈금(Minor Ticks) 자동 생성 활성화
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 주 그리드(Major Grid)와 보조 그리드(Minor Grid)를 각각 다르게 렌더링
    ax.grid(True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', color='#e5e5e5', linestyle=':', linewidth=0.4)
    
    # 통계 정보(Min, Max, Mean) 박스를 우측 상단에 표시
    s_min = np.min(scanline)
    s_max = np.max(scanline)
    s_mean = np.mean(scanline)
    stats_text = f"Min: {s_min:.6f}\nMax: {s_max:.6f}\nMean: {s_mean:.6f}"
    
    # 축 범위 여유 설정
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        y_margin = (s_max - s_min) * 0.05 if s_max > s_min else 1.0
        ax.set_ylim(s_min - y_margin, s_max + y_margin)
    ax.set_xlim(0, len(scanline) - 1)
    
    # 통계 텍스트 상자 배치
    # ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.8),
    #         fontsize=9, family='monospace')
            
    # 레이아웃 최적화
    plt.tight_layout()
    
    # 특수문자나 공백이 있을 수 있으니 안전한 파일명으로 처리
    safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    
    file_path_png = os.path.join(save_dir, f"scanline_col{col_index}_{safe_stage_name}.png")
    plt.savefig(file_path_png, dpi=300)
    plt.close()
    
    file_path_npy = os.path.join(save_dir, f"scanline_col{col_index}_{safe_stage_name}.npy")
    np.save(file_path_npy, scanline)
    print(f"[{stage_name}] Scanline at column {col_index} saved to {save_dir}.")

