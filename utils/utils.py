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