import numpy as np
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import cv2

# 📌 크롬 드라이버 경로
chrome_driver_path = "chromedriver-win64/chromedriver.exe"

# 📌 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--ignore-ssl-errors")

# 📌 드라이버 실행
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# 📌 게임 페이지 열기
driver.get("https://www.gamesaien.com/game/fruit_box_a/")

# 📌 캔버스 대기 및 위치 계산
time.sleep(2)
canvas = driver.find_element(By.ID, "canvas")
canvas_rect = driver.execute_script("""
    const rect = document.getElementById('canvas').getBoundingClientRect();
    return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
""")
canvas_left = int(canvas_rect["left"])
canvas_top = int(canvas_rect["top"])
canvas_width = int(canvas_rect["width"])
canvas_height = int(canvas_rect["height"])

# 📌 Play 버튼 클릭
play_x = canvas_left + (canvas_width // 2) - 170
play_y = canvas_top + (canvas_height // 2)
driver.execute_script(f"""
    const canvas = document.getElementById('canvas');
    const rect = canvas.getBoundingClientRect();
    const x = rect.left + {play_x - canvas_left};
    const y = rect.top + {play_y - canvas_top};
    ['mousedown', 'mouseup'].forEach(type => {{
        canvas.dispatchEvent(new MouseEvent(type, {{
            bubbles: true, cancelable: true, view: window,
            clientX: x, clientY: y
        }}));
    }});    
""")
print(f"✅ 'Play' 버튼 클릭 완료! 위치: ({play_x}, {play_y})")

# 📸 스크린샷 저장
time.sleep(2)
canvas.screenshot("data/game_screenshot.png")
print("📸 스크린샷 저장 완료: game_screenshot.png")

# 🔍 사과 중심 좌표 검출
image = cv2.imread("data/game_screenshot.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)
kernel = np.ones((3, 3), np.uint8)
clean_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circle_centers = []
for cnt in contours:
    if cv2.contourArea(cnt) < 50:
        continue
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    if 8 <= radius <= 30:
        circle_centers.append((int(x), int(y)))

if len(circle_centers) >= 170:
    sorted_centers = sorted(circle_centers, key=lambda c: (c[1], c[0]))[:170]
    center_grid = np.array(sorted_centers, dtype=np.int32).reshape((10, 17, 2))
    np.save("data/center_grid.npy", center_grid)
    print("✅ 사과 중심 좌표 저장 완료 (center_grid.npy)")
else:
    print(f"⚠️ 감지된 중심 부족: {len(circle_centers)}개")
    driver.quit()
    exit()

# 🔠 OCR 실행
os.system("python ocr_pipline.py")
ocr_result = np.loadtxt("data/ocr_result_corrected_10x17.csv", delimiter=",", dtype=int)
center_grid = np.load("data/center_grid.npy")


# 🧠 AI 기반 마스크 예측 후 드래그 실행
def find_groups_with_sum_10(grid):
    rows, cols = grid.shape
    valid_groups = []

    for height in range(1, rows + 1):
        for width in range(1, cols + 1):
            for r1 in range(rows - height + 1):
                for c1 in range(cols - width + 1):
                    r2 = r1 + height - 1
                    c2 = c1 + width - 1
                    region = grid[r1:r2 + 1, c1:c2 + 1]
                    valid_values = region[(region != -1) & (region != 0)]  # -1과 0 제외

                    # 합이 10인 그룹 찾기
                    if valid_values.size > 0 and np.sum(valid_values) == 10:
                        valid_groups.append(region)  # 그룹을 2D 배열로 저장

    return valid_groups


# 📌 합이 10인 모든 그룹 찾기
valid_groups = find_groups_with_sum_10(ocr_result)
print(f"합이 10인 그룹 개수: {len(valid_groups)}")


# 📌 각 그룹을 훈련 데이터로 저장 (하나의 CSV 파일에 저장)
def save_training_data(valid_groups):
    all_groups = []

    # 각 그룹을 1D 배열로 변환하여 리스트에 추가
    for group in valid_groups:
        all_groups.append(group.flatten())  # 각 그룹을 1D로 변환

    # 1D 배열들을 리스트로 묶어서 numpy 배열로 변환
    all_groups = np.array(all_groups, dtype=int)

    # 훈련 데이터 CSV로 저장
    np.savetxt("training_data.csv", all_groups, delimiter=",", fmt="%d")
    print(f"훈련 데이터 저장 완료: \n{all_groups}")


# 📌 훈련 데이터 저장
save_training_data(valid_groups)

# 게임 종료 후 훈련 데이터를 확인
input("게임 종료 후 Enter를 눌러 브라우저를 닫습니다...")
driver.quit()
