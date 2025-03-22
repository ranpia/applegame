import cv2
import numpy as np
import pytesseract
import pandas as pd
from collections import Counter

# Tesseract 실행 경로 설정 (Windows 사용자는 수정 필요)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 📌 스크린샷 로드
image_path = "game_screenshot.png"
image = cv2.imread(image_path)
if image is None:
    print("❌ 이미지 불러오기 실패! 경로 확인:", image_path)
    exit()

# 📌 center_grid.npy 로드 (10x17 사과 중심 좌표)
try:
    center_grid = np.load("center_grid.npy")  # shape: (10, 17, 2)
except:
    print("❌ center_grid.npy 파일이 없습니다! (사과 중심 좌표 필요)")
    exit()

# 📌 HSV 기반 숫자 추출 (흰색 마스킹)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)
digits_only = cv2.bitwise_and(image, image, mask=white_mask)
digits_gray = cv2.cvtColor(digits_only, cv2.COLOR_BGR2GRAY)

# 📌 OCR 실행 및 보정
ocr_results = np.full((10, 17), -1)  # -1: OCR 실패 표시

# 🔹 OCR 최적 모드 선택 함수
def run_ocr_variants(roi):
    """ 다양한 OCR 모드 실행 후 최적값 반환 """
    ocr_attempts = []
    psm_modes = ["--psm 6 digits", "--psm 8 digits", "--psm 10 digits"]

    for mode in psm_modes:
        text = pytesseract.image_to_string(roi, config=mode).strip()
        try:
            value = int(text)
            if 0 <= value <= 9:
                ocr_attempts.append(value)
        except:
            pass

    if len(ocr_attempts) > 0:
        return Counter(ocr_attempts).most_common(1)[0][0]  # 최빈값 반환
    return -1  # OCR 실패 시 -1 유지

# 📌 개별 사과 이미지로 분할 후 OCR 적용
for i in range(10):
    for j in range(17):
        x, y = center_grid[i, j]
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(digits_gray.shape[1], x + margin), min(digits_gray.shape[0], y + margin)

        roi = digits_gray[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (roi.shape[1] * 4, roi.shape[0] * 4))

        # OCR 실행
        ocr_results[i, j] = run_ocr_variants(roi_resized)

# 📌 보정된 OCR 결과 저장
df_corrected = pd.DataFrame(ocr_results)
df_corrected.to_csv("ocr_result_corrected_10x17.csv", index=False, header=False)
print("✅ OCR 보정 완료 → ocr_result_corrected_10x17.csv 저장")