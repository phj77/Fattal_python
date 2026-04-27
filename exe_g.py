import cv2
import numpy as np

# 톤 매핑 알고리즘 모듈 임포트
from fattal_tmo import pfstmo_fattal02 

# 앞서 작성한 감마 보정 모듈 임포트
# (해당 코드가 'gamma_correction.py' 파일로 저장되어 있다고 가정합니다)
from gamma_correction import Frame, apply_gamma_frame

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
opt_beta = 0.82       # 베타 값 (압축률)
opt_saturation = 0.8  # 채도 복원 강도
opt_noise = 0.001     # 노이즈 레벨 (0.0 이하 입력 시 자동 설정됨)
newfattal = True     # 신규 Fattal 알고리즘 적용 여부
fftsolver = True      # FFT 솔버 사용 (성능 향상을 위해 True 권장)
detail_level = 3      # 디테일 보존 수준 (0 ~ 3)

# 추가된 감마 보정 파라미터
pre_gamma = 1.2     # 전처리 감마 값 (1.0일 경우 원본 유지)
post_gamma = 0.68      # 후처리 감마 값 (일반적인 sRGB 모니터 출력을 위해 2.2 사용)

# 4. 전처리: 톤 매핑 전 감마 보정 적용
# 분리된 채널 데이터를 Frame 객체로 포장합니다.
pre_frame = Frame(R, G, B)
apply_gamma_frame(pre_frame, pre_gamma)

# 감마 보정이 적용된 데이터를 다시 NumPy 배열로 추출합니다.
# gamma_correction 모듈 내부에서 데이터가 복사(copy)되므로 반드시 갱신된 데이터를 명시적으로 가져와야 합니다.
R_pre = pre_frame.x_channel.data
G_pre = pre_frame.y_channel.data
B_pre = pre_frame.z_channel.data

# 5. 톤 매핑 알고리즘 실행
# 원본 데이터(R, G, B) 대신 전처리가 완료된 데이터(R_pre, G_pre, B_pre)를 전달합니다.
R_out, G_out, B_out = pfstmo_fattal02(
    R_pre, G_pre, B_pre, 
    opt_alpha, opt_beta, opt_saturation, opt_noise, 
    newfattal, fftsolver, detail_level
)

# 6. 후처리: 톤 매핑 후 감마 보정 적용
# 톤 매핑 결과 채널 데이터를 다시 Frame 객체로 포장합니다.
post_frame = Frame(R_out, G_out, B_out)
apply_gamma_frame(post_frame, post_gamma)

# 최종 감마 보정이 적용된 데이터를 추출합니다.
R_final = post_frame.x_channel.data
G_final = post_frame.y_channel.data
B_final = post_frame.z_channel.data

# 7. 채널 병합 및 출력 포맷 변환
out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)

# 결과값을 0.0 ~ 1.0 범위로 클리핑 후 8비트(0~255) 정수형으로 변환
out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)

# 이미지 저장을 위해 RGB 배열을 다시 BGR 배열로 변환
out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image7.png', out_img_bgr)