import cv2
import numpy as np

from fattal_tmo import pfstmo_fattal02
from gamma_correction import Frame, apply_gamma_frame

# 1. 이미지 로드
img_path = 'input.hdr'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

# ─── 추가: 그레이스케일 여부 감지 ────────────────────────────────────────────
is_grayscale = (img.ndim == 2)

if is_grayscale:
    # 단일 채널을 3채널로 복제하여 이후 파이프라인이 동일하게 동작하도록 합니다.
    # cvtColor, Frame, pfstmo_fattal02 모두 3채널 입력을 요구하기 때문입니다.
    img = np.stack([img, img, img], axis=-1)  # (H, W) → (H, W, 3), BGR 형태 유지
# ─────────────────────────────────────────────────────────────────────────────

# 2. BGR → RGB 변환 및 채널 분리
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# 3. 파라미터 설정
opt_alpha = 0.3
opt_beta = 0.98
opt_noise = 0.001
newfattal = True  ##### fft_solver가 True이면 무조건 True
fftsolver = False
detail_level = 0

pre_gamma = 1
post_gamma = 1

# ─── 추가: 그레이스케일일 경우 채도 복원 파라미터 무력화 ──────────────────────
# opt_saturation은 컬러 채널 간 비율을 복원하는 파라미터입니다.
# 세 채널이 동일한 그레이스케일 복제본이므로, 1.0으로 설정해야
# 채널 간 비율이 변형되지 않습니다. (0.8 등으로 설정 시 결과가 왜곡됩니다)
opt_saturation = 1.0 if is_grayscale else 0.8
# ─────────────────────────────────────────────────────────────────────────────

# 4. 전처리 감마 보정
pre_frame = Frame(R, G, B)
apply_gamma_frame(pre_frame, pre_gamma)

R_pre = pre_frame.x_channel.data
G_pre = pre_frame.y_channel.data
B_pre = pre_frame.z_channel.data

# 5. 톤 매핑
R_out, G_out, B_out = pfstmo_fattal02(
    R_pre, G_pre, B_pre,
    opt_alpha, opt_beta, opt_saturation, opt_noise,
    newfattal, fftsolver, detail_level
)

# 6. 후처리 감마 보정
post_frame = Frame(R_out, G_out, B_out)
apply_gamma_frame(post_frame, post_gamma)

R_final = post_frame.x_channel.data
G_final = post_frame.y_channel.data
B_final = post_frame.z_channel.data

# 7. 채널 병합 및 포맷 변환
out_img_rgb = np.stack((R_final, G_final, B_final), axis=-1)
out_img_rgb = np.clip(out_img_rgb, 0.0, 1.0)
out_img_8bit = (out_img_rgb * 255.0).astype(np.uint8)
out_img_bgr = cv2.cvtColor(out_img_8bit, cv2.COLOR_RGB2BGR)

# ─── 추가: 원본이 그레이스케일이면 단채널로 변환 후 저장 ─────────────────────
# 복제된 3채널은 값이 모두 동일하므로 어느 채널을 추출해도 결과가 같습니다.
# BGR2GRAY 변환(가중 평균)을 쓰지 않고 단순 채널 추출이 더 정확합니다.
if is_grayscale:
    out_img_bgr = out_img_bgr[:, :, 0]
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# # 히스토그램 평활화 적용
# if not is_grayscale:
#     out_img_bgr = out_img_bgr[:, :, 0]
# equalized_image = cv2.equalizeHist(out_img_bgr)
# cv2.imshow('image', equalized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # cv2.imwrite('result.jpg', equalized_image)
# ─────────────────────────────────────────────────────────────────────────────



cv2.imshow('image', out_img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite(f'./images/beta_images/NewFattal_Multigrid_alpha_0.3/beta_{opt_beta}.png', out_img_bgr)