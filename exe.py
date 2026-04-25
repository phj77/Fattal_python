import cv2
import numpy as np

# 앞서 작성된 코드가 포함된 파일에서 메인 함수를 임포트합니다.
from fattal_tmo import pfstmo_fattal02 

# 1. 이미지 로드
# float32 형태의 원본 HDR 이미지를 읽어옵니다. OpenCV는 이미지를 BGR 채널 순서로 읽어옵니다.
img_path = 'input.hdr'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

# BGR 배열을 RGB 배열로 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 채널 분리
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# 3. 파라미터 설정
opt_alpha = 0.12      # 알파 값 (대비 제어)
opt_beta = 0.82        # 베타 값 (압축률)
opt_saturation = 0.8  # 채도 복원 강도
opt_noise = 0.001     # 노이즈 레벨 (0.0 이하 입력 시 자동 설정됨)
newfattal = True      # 신규 Fattal 알고리즘 적용 여부
fftsolver = True      # FFT 솔버 사용 (성능 향상을 위해 True 권장)
detail_level = 3      # 디테일 보존 수준 (0 ~ 3)

# 4. 톤 매핑 알고리즘 실행
R_out, G_out, B_out = pfstmo_fattal02(
    R, G, B, 
    opt_alpha, opt_beta, opt_saturation, opt_noise, 
    newfattal, fftsolver, detail_level
)

# 5. 채널 병합 및 출력 포맷 변환
out_img_rgb = np.stack((R_out, G_out, B_out), axis=-1)

# 결과값을 0.0 ~ 1.0 범위로 클리핑 후 8비트(0~255) 정수형으로 변환
out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)

# 이미지 저장을 위해 RGB 배열을 다시 BGR 배열로 변환
out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image2.png', out_img_bgr)