import os
import cv2
import numpy as np
import pandas as pd

# 📁 경로 설정
screenshot_path = "../game_screenshot.png"
center_path = "../center_grid.npy"
ocr_csv_path = "../ocr_result_corrected_10x17.csv"
output_dir = "../ocr_data"
debug_dir = "../debug_samples"  # 디버깅용 저장 경로

# 📂 출력 디렉토리 생성
for digit in range(10):
    os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# 📸 이미지 및 좌표 로드
image = cv2.imread(screenshot_path)
if image is None:
    raise FileNotFoundError(f"❌ 이미지 로드 실패: {screenshot_path}")
center_grid = np.load(center_path)
ocr_result = pd.read_csv(ocr_csv_path, header=None).values  # (10, 17)

# 🔍 각 사과에서 숫자 추출하여 저장
count = 0
skipped = 0
margin = 14  # crop 영역 마진

for i in range(10):
    for j in range(17):
        label = ocr_result[i][j]
        if label not in range(10):
            skipped += 1
            continue  # 잘못된 OCR 결과

        x, y = center_grid[i][j]
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + margin), min(image.shape[0], y + margin)
        roi = image[y1:y2, x1:x2]

        roi_resized = cv2.resize(roi, (32, 32))

        # 🔎 너무 어두운 경우 걸러냄 (숫자가 없을 확률 높음)
        if np.mean(roi_resized) < 30:
            skipped += 1
            continue

        save_path = os.path.join(output_dir, str(label), f"img_{i}_{j}.png")
        cv2.imwrite(save_path, roi_resized)

        # 디버깅 샘플 저장 (처음 30개만 저장)
        if count < 30:
            cv2.imwrite(os.path.join(debug_dir, f"label{label}_img_{i}_{j}.png"), roi_resized)

        count += 1

print(f"✅ 저장된 이미지 수: {count}개")
print(f"⛔ 스킵된 이미지 수 (잘못된 라벨 또는 너무 어두움): {skipped}개")
print(f"🔍 디버깅 샘플 저장 완료: {debug_dir}")
