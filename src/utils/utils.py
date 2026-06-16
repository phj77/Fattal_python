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


def save_scanline(image, row_index, stage_name, save_dir="scanlines"):
    """
    특정 행(row)의 스캔라인을 추출하여 그래프(PNG)와 데이터(NPY)로 저장합니다. (OpenCV 사용)
    
    Args:
        image (np.ndarray): 스캔라인을 추출할 2D 배열 이미지
        row_index (int): 스캔라인을 추출할 행의 인덱스
        stage_name (str): 파일명과 그래프 제목에 들어갈 단계 이름
        save_dir (str): 파일들이 저장될 디렉토리 (기본값: "scanlines")
    """
    import os
    import cv2
    import numpy as np
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 행 인덱스가 이미지 범위를 벗어나지 않도록 클리핑
    h = image.shape[0]
    row_index = np.clip(row_index, 0, h - 1)
    
    scanline = image[row_index, :]
    
    # 데이터는 npy 파일로 우선 저장
    safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    file_path_npy = os.path.join(save_dir, f"scanline_row{row_index}_{safe_stage_name}.npy")
    np.save(file_path_npy, scanline)
    
    # OpenCV를 이용한 그래프 그리기
    # 캔버스 크기 정의 (높이 500, 너비 1200)
    canvas_h, canvas_w = 500, 1200
    # 흰색 배경의 3채널(BGR) 캔버스 생성
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 여백 설정
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 60
    
    plot_w = canvas_w - margin_left - margin_right
    plot_h = canvas_h - margin_top - margin_bottom
    
    # 최소값, 최대값 계산
    s_min = float(np.min(scanline))
    s_max = float(np.max(scanline))
    s_range = s_max - s_min
    if s_range == 0:
        s_range = 1.0
        
    # 1. 축 그리기 (검정색)
    # X축 (하단)
    cv2.line(canvas, (margin_left, canvas_h - margin_bottom), (canvas_w - margin_right, canvas_h - margin_bottom), (0, 0, 0), 2)
    # Y축 (좌측)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, canvas_h - margin_bottom), (0, 0, 0), 2)
    
    # 2. 격자선(Grid) 및 눈금 텍스트 그리기 (회색)
    num_y_div = 5
    for i in range(num_y_div + 1):
        # Y축 좌표 계산
        y_val = s_min + (s_range * i / num_y_div)
        y_pos = canvas_h - margin_bottom - int((i / num_y_div) * plot_h)
        
        # 격자선 그리기 (Y축 기준 가로선)
        if i > 0 and i < num_y_div:
            cv2.line(canvas, (margin_left, y_pos), (canvas_w - margin_right, y_pos), (220, 220, 220), 1)
            
        # 눈금 라벨 그리기 (소수점 4자리까지)
        label = f"{y_val:.4f}"
        # 라벨 텍스트의 크기 구하기
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, label, (margin_left - text_w - 8, y_pos + text_h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)
                    
    num_x_div = 10
    total_cols = len(scanline)
    for i in range(num_x_div + 1):
        col_idx = int((total_cols - 1) * i / num_x_div)
        x_pos = margin_left + int((i / num_x_div) * plot_w)
        
        # 격자선 그리기 (X축 기준 세로선)
        if i > 0 and i < num_x_div:
            cv2.line(canvas, (x_pos, margin_top), (x_pos, canvas_h - margin_bottom), (220, 220, 220), 1)
            
        # 눈금 라벨 그리기
        label = str(col_idx)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, label, (x_pos - text_w // 2, canvas_h - margin_bottom + text_h + 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)
                    
    # 3. 데이터 플롯 그리기 (파란색 실선)
    points = []
    for col_idx, val in enumerate(scanline):
        x = margin_left + int((col_idx / (total_cols - 1)) * plot_w)
        y = (canvas_h - margin_bottom) - int(((val - s_min) / s_range) * plot_h)
        points.append((x, y))
        
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], (255, 0, 0), 2, cv2.LINE_AA)
        
    # 4. 타이틀 및 통계 정보 그리기
    title_text = f"Scanline Intensity at Row {row_index} - {stage_name}"
    cv2.putText(canvas, title_text, (margin_left, margin_top - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
    info_text = f"Min: {s_min:.6f}  Max: {s_max:.6f}  Width: {total_cols}"
    (text_w, text_h), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(canvas, info_text, (canvas_w - margin_right - text_w, margin_top - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
                
    file_path_png = os.path.join(save_dir, f"scanline_row{row_index}_{safe_stage_name}.png")
    cv2.imwrite(file_path_png, canvas)
    print(f"[{stage_name}] Scanline at row {row_index} saved to {save_dir} (OpenCV version).")