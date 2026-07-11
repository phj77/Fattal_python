import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

def load_hdr_image(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    HDR 이미지를 불러와 RGB 및 Luminance(Y) 배열을 반환합니다.
    
    Args:
        img_path (Path): 이미지 파일 경로
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (RGB float32 이미지, Luminance 2D float32 이미지)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 읽을 수 없습니다: {img_path}")
        
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    # Rec.709 표준 Luminance 계산 (Log 도메인 변환 없이 원본 값 유지)
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    return img_rgb, Y

def compute_gradient_magnitude(Y: np.ndarray) -> np.ndarray:
    """
    Log domain 변환 없이 입력 2D Luminance 이미지에 대해 Gradient Magnitude G = sqrt(gx^2 + gy^2)를 계산합니다.
    
    Args:
        Y (np.ndarray): 2D Luminance 이미지 배열
        
    Returns:
        np.ndarray: Gradient Magnitude 배열
    """
    h, w = Y.shape
    gx = np.empty_like(Y, dtype=np.float32)
    gx[:, 0] = Y[:, 0] - Y[:, 1]
    gx[:, -1] = Y[:, -2] - Y[:, -1]
    gx[:, 1:-1] = (Y[:, :-2] - Y[:, 2:]) / 2.0
    
    gy = np.empty_like(Y, dtype=np.float32)
    gy[0, :] = Y[0, :] - Y[1, :]
    gy[-1, :] = Y[-2, :] - Y[-1, :]
    gy[1:-1, :] = (Y[:-2, :] - Y[2:, :]) / 2.0
    
    grad_mag = np.sqrt(gx**2 + gy**2)
    return grad_mag

def generate_low_gradient_mask(Y: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Luminance 이미지 Y에서 gradient magnitude를 구하고,
    threshold(기본값 5.0) 이하인 픽셀은 255(흰색), 초과인 픽셀은 0(검은색)으로 칠한 이진 마스크를 생성합니다.
    
    Args:
        Y (np.ndarray): Luminance 이미지
        threshold (float): Gradient 임계값 (기본 5.0)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - binary_mask: uint8 2진 마스크 (255: <= threshold, 0: > threshold)
            - grad_mag: 계산된 gradient magnitude 배열
            - stats: gradient 통계 정보
    """
    grad_mag = compute_gradient_magnitude(Y)
    
    # gradient 크기가 threshold 이하인 픽셀만 완전히 하얗게(255), 나머지는 검게(0)
    binary_mask = np.where(grad_mag <= threshold, 255, 0).astype(np.uint8)
    
    total_pixels = grad_mag.size
    low_grad_count = np.sum(grad_mag <= threshold)
    low_grad_ratio = (low_grad_count / total_pixels) * 100.0
    
    stats = {
        "threshold": threshold,
        "min_grad": float(np.min(grad_mag)),
        "max_grad": float(np.max(grad_mag)),
        "mean_grad": float(np.mean(grad_mag)),
        "low_grad_count": int(low_grad_count),
        "total_pixels": int(total_pixels),
        "low_grad_ratio_pct": float(low_grad_ratio)
    }
    
    return binary_mask, grad_mag, stats
