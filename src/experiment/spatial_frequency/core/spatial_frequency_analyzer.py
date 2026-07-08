# spatial_frequency_analyzer.py
# Core module for image spatial frequency analysis in frequency and spatial domains.

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Union


def load_image_luminance(
    img_path: Union[str, Path],
    log_domain: bool = False,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    이미지(HDR 또는 LDR)를 읽어와 RGB 배열 및 Rec.709 기준 휘도(Luminance, Y)를 계산합니다.

    Args:
        img_path (Union[str, Path]): 이미지 파일 경로 (.hdr, .exr, .png, .jpg 등)
        log_domain (bool): True일 경우 Log-Luminance ln(Y + eps)를 반환합니다.
        eps (float): Log 계산 시 0으로 나누기 및 log(0) 방지를 위한 미소값.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - img_rgb (np.ndarray): HxWx3 RGB 이미지 배열 (float32)
            - Y (np.ndarray): HxW 2D 휘도 또는 로그 휘도 배열 (float32)
            - meta (Dict[str, Any]): 이미지 메타정보 (파일명, 크기, 동적범위 등)
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"이미지 읽기 실패: {img_path}")

    # 채널 처리 (그레이스케일인 경우 RGB 확장)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Rec.709 표준 Luminance (Y = 0.2126 R + 0.7152 G + 0.0722 B)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    Y_raw = 0.2126 * R + 0.7152 * G + 0.0722 * B

    min_val, max_val = float(np.min(Y_raw)), float(np.max(Y_raw))

    if log_domain:
        Y_processed = np.log(np.maximum(Y_raw, 0) + eps)
    else:
        Y_processed = Y_raw.copy()

    meta = {
        "filename": img_path.name,
        "path": str(img_path),
        "height": Y_raw.shape[0],
        "width": Y_raw.shape[1],
        "min_val": min_val,
        "max_val": max_val,
        "dynamic_range_db": float(20 * np.log10((max_val + eps) / (min_val + eps))) if max_val > min_val else 0.0,
        "log_domain": log_domain
    }

    return img_rgb, Y_processed.astype(np.float32), meta


def compute_fft_2d(luminance: np.ndarray) -> Dict[str, Any]:
    """
    2D Fast Fourier Transform(FFT)을 수행하고 진폭 및 위상 스펙트럼을 산출합니다.

    Args:
        luminance (np.ndarray): HxW 2D 휘도 이미지 배열

    Returns:
        Dict[str, Any]:
            - F_shift (np.ndarray): 0Hz 주파수가 중앙에 배치된 2D 복소 FFT 배열
            - magnitude_spectrum (np.ndarray): 시각화용 Log Magnitude 스펙트럼 log(1 + |F|)
            - magnitude_db (np.ndarray): dB 스케일 Magnitude 스펙트럼 20*log10(|F| + 1e-9)
            - phase_spectrum (np.ndarray): 위상 스펙트럼 (radians)
            - power_spectrum (np.ndarray): 2D Power Spectrum |F|^2
    """
    h, w = luminance.shape
    F = np.fft.fft2(luminance)
    F_shift = np.fft.fftshift(F)

    abs_F = np.abs(F_shift)
    magnitude_spectrum = np.log1p(abs_F)
    magnitude_db = 20 * np.log10(abs_F + 1e-9)
    phase_spectrum = np.angle(F_shift)
    power_spectrum = abs_F ** 2

    return {
        "F_shift": F_shift,
        "magnitude_spectrum": magnitude_spectrum,
        "magnitude_db": magnitude_db,
        "phase_spectrum": phase_spectrum,
        "power_spectrum": power_spectrum
    }


def compute_radial_psd(
    power_spectrum: np.ndarray,
    num_bins: int = 100
) -> Dict[str, Any]:
    """
    2D Power Spectrum을 동심원 반경(Radial) 방향으로 평균화하여 1D Radial Power Spectral Density (RPSD)를 계산하고,
    자연 이미지의 주파수 쇠퇴 특성 1/f^alpha (Power-law slope)를 피팅합니다.

    Args:
        power_spectrum (np.ndarray): HxW 2D Power Spectrum 배열 (|F_shift|^2)
        num_bins (int): 반경 분할 구간(Bin) 개수

    Returns:
        Dict[str, Any]:
            - freqs (np.ndarray): 정규화된 공간 주파수 배열 [0, 0.5] (cycles/pixel)
            - psd_1d (np.ndarray): 각 주파수 대역별 평균 1D PSD 값
            - slope_alpha (float): 1/f^alpha 형태의 피팅 쇠퇴 지수 alpha (양수)
            - fit_r2 (float): 선형 피팅의 결정계수 R^2
            - fit_line (np.ndarray): 피팅된 1D PSD 추세선 (Log scale 상)
    """
    h, w = power_spectrum.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    y_indices, x_indices = np.indices((h, w))
    r_matrix = np.sqrt((y_indices - cy) ** 2 + (x_indices - cx) ** 2)

    # 주파수 반경의 최대 범위 (나이퀴스트 주파수 0.5 cycles/pixel에 해당)
    r_max = min(cy, cx)
    if r_max <= 0:
        r_max = max(cy, cx)

    bin_edges = np.linspace(0, r_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    psd_1d = np.zeros(num_bins, dtype=np.float64)

    for i in range(num_bins):
        mask = (r_matrix >= bin_edges[i]) & (r_matrix < bin_edges[i + 1])
        if np.any(mask):
            psd_1d[i] = np.mean(power_spectrum[mask])
        else:
            psd_1d[i] = 0.0

    # cycles / pixel 단위 공간 주파수 (최대 0.5)
    freqs = (bin_centers / r_max) * 0.5

    # 1/f^alpha 로그 피팅 (DC성분 극저주파 f < 0.02 및 psd=0 제외)
    valid_mask = (freqs >= 0.02) & (freqs <= 0.45) & (psd_1d > 1e-12)

    slope_alpha = 0.0
    fit_r2 = 0.0
    fit_line = np.zeros_like(psd_1d)

    if np.sum(valid_mask) > 3:
        log_f = np.log10(freqs[valid_mask])
        log_psd = np.log10(psd_1d[valid_mask])

        # log(PSD) = -alpha * log(f) + c
        poly = np.polyfit(log_f, log_psd, 1)
        slope_alpha = float(-poly[0])  # alpha값 (양수로 표현)

        # R^2 계산
        log_psd_pred = poly[0] * log_f + poly[1]
        ss_res = np.sum((log_psd - log_psd_pred) ** 2)
        ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
        fit_r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

        # 전체 구간 추세선 생성
        log_f_all = np.log10(np.maximum(freqs, 1e-6))
        fit_line = 10 ** (poly[0] * log_f_all + poly[1])

    return {
        "freqs": freqs,
        "psd_1d": psd_1d,
        "slope_alpha": slope_alpha,
        "fit_r2": fit_r2,
        "fit_line": fit_line
    }


def decompose_frequency_bands(
    luminance: np.ndarray,
    F_shift: np.ndarray = None,
    sigma_low: float = 0.05,
    sigma_high: float = 0.20
) -> Dict[str, Any]:
    """
    2D 주파수 영역에서 가우시안 대역 필터를 적용하여 이미지를 저주파(LF), 중주파(MF), 고주파(HF)로 분해합니다.

    Args:
        luminance (np.ndarray): HxW 2D 휘도 이미지
        F_shift (np.ndarray): 0Hz가 중심인 2D Complex FFT 배열 (없으면 자동 계산)
        sigma_low (float): 저주파 커트오프 비율 (0 ~ 0.5 cycles/pixel)
        sigma_high (float): 고주파 커트오프 비율 (0 ~ 0.5 cycles/pixel)

    Returns:
        Dict[str, Any]:
            - lf_image (np.ndarray): 저주파 서브 이미지
            - mf_image (np.ndarray): 중주파 서브 이미지
            - hf_image (np.ndarray): 고주파 서브 이미지
            - energy_pct (Dict[str, float]): 대역별 전력 에너지 비중 {'low': %, 'mid': %, 'high': %}
    """
    h, w = luminance.shape
    if F_shift is None:
        F_shift = np.fft.fftshift(np.fft.fft2(luminance))

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_idx, x_idx = np.indices((h, w))

    r_norm = np.sqrt(((y_idx - cy) / max(cy, 1)) ** 2 + ((x_idx - cx) / max(cx, 1)) ** 2) * 0.5

    # 가우시안 주파수 필터
    H_low = np.exp(- (r_norm ** 2) / (2 * (sigma_low ** 2)))
    H_high = 1.0 - np.exp(- (r_norm ** 2) / (2 * (sigma_high ** 2)))
    H_mid = np.clip(1.0 - H_low - H_high, 0.0, 1.0)

    # 주파수 마스킹 및 역 FFT
    F_lf = F_shift * H_low
    F_mf = F_shift * H_mid
    F_hf = F_shift * H_high

    lf_img = np.real(np.fft.ifft2(np.fft.ifftshift(F_lf)))
    mf_img = np.real(np.fft.ifft2(np.fft.ifftshift(F_mf)))
    hf_img = np.real(np.fft.ifft2(np.fft.ifftshift(F_hf)))

    # 대역별 에너지 계산 (|F|^2 총합)
    E_lf = np.sum(np.abs(F_lf) ** 2)
    E_mf = np.sum(np.abs(F_mf) ** 2)
    E_hf = np.sum(np.abs(F_hf) ** 2)
    E_total = E_lf + E_mf + E_hf + 1e-9

    energy_pct = {
        "low": float(E_lf / E_total * 100.0),
        "mid": float(E_mf / E_total * 100.0),
        "high": float(E_hf / E_total * 100.0)
    }

    return {
        "lf_image": lf_img,
        "mf_image": mf_img,
        "hf_image": hf_img,
        "energy_pct": energy_pct,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high
    }


def compute_spatial_frequency_map(luminance: np.ndarray) -> Dict[str, Any]:
    """
    공간 도메인에서 픽셀 단위 Row Frequency(RF) 및 Column Frequency(CF) 변화율을 바탕으로
    국소 공간 주파수(Spatial Frequency, SF) 히트맵과 단일 SF 평가 지표를 산출합니다.
    SF(x, y) = sqrt(RF(x, y)^2 + CF(x, y)^2)

    Args:
        luminance (np.ndarray): HxW 2D 휘도 이미지

    Returns:
        Dict[str, Any]:
            - sf_map (np.ndarray): HxW 국소 공간 주파수 히트맵
            - sf_index (float): 전체 이미지의 공간 주파수 통합 평가 지표 (RMS SF)
            - rf_mean (float): 평균 Row Frequency
            - cf_mean (float): 평균 Column Frequency
    """
    h, w = luminance.shape
    RF = np.zeros_like(luminance, dtype=np.float32)
    CF = np.zeros_like(luminance, dtype=np.float32)

    # Row Frequency: RF(i, j) = |I(i, j) - I(i, j-1)|
    RF[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1])

    # Column Frequency: CF(i, j) = |I(i, j) - I(i-1, j)|
    CF[1:, :] = np.abs(luminance[1:, :] - luminance[:-1, :])

    # Local Spatial Frequency map
    SF_map = np.sqrt(RF ** 2 + CF ** 2)

    # Overall Spatial Frequency Index (RMS metric)
    sf_index = float(np.sqrt(np.mean(RF ** 2 + CF ** 2)))
    rf_mean = float(np.mean(RF))
    cf_mean = float(np.mean(CF))

    return {
        "sf_map": SF_map,
        "sf_index": sf_index,
        "rf_mean": rf_mean,
        "cf_mean": cf_mean,
        "rf_map": RF,
        "cf_map": CF
    }


def analyze_spatial_frequency(
    img_path: Union[str, Path],
    log_domain: bool = True,
    num_bins: int = 100
) -> Dict[str, Any]:
    """
    이미지 로드부터 2D FFT, 1D Radial PSD, 3대역 주파수 분해, 공간 주파수 히트맵 산출까지
    모든 공간 주파수 분석 프로세스를 일괄 수행합니다.

    Args:
        img_path (Union[str, Path]): 이미지 경로
        log_domain (bool): HDR 로그 도메인 적용 여부
        num_bins (int): 1D Radial PSD 분할 수

    Returns:
        Dict[str, Any]: 통합 분석 결과 딕셔너리
    """
    img_rgb, Y, meta = load_image_luminance(img_path, log_domain=log_domain)
    fft_results = compute_fft_2d(Y)
    psd_results = compute_radial_psd(fft_results["power_spectrum"], num_bins=num_bins)
    band_results = decompose_frequency_bands(Y, F_shift=fft_results["F_shift"])
    sf_results = compute_spatial_frequency_map(Y)

    return {
        "img_rgb": img_rgb,
        "luminance": Y,
        "meta": meta,
        "fft": fft_results,
        "psd": psd_results,
        "bands": band_results,
        "sf": sf_results
    }
