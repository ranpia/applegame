from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import numpy as np
import cv2
import os

# 📌 크롬 드라이버 경로
chrome_driver_path = "chromedriver-win64/chromedriver.exe"

# 📌 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--ignore-ssl-errors")
# chrome_options.add_argument("--headless=new")  # 필요 시 백그라운드 실행

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

# 📌 캔버스 스크린샷
time.sleep(2)
canvas.screenshot("game_screenshot.png")
print("📸 스크린샷 저장 완료: game_screenshot.png")

# ---------------------------
# 🔍 사과 중심 좌표 검출
# ---------------------------
image = cv2.imread("game_screenshot.png")
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
    np.save("center_grid.npy", center_grid)
    print("✅ 사과 중심 좌표 저장 완료 (center_grid.npy)")
else:
    print(f"⚠️ 감지된 중심 부족: {len(circle_centers)}개")
    driver.quit()
    exit()

# ---------------------------
# 🔠 OCR 실행
# ---------------------------
os.system("python ocr_pipline.py")

ocr_result = np.loadtxt("ocr_result_corrected_10x17.csv", delimiter=",", dtype=int)
center_grid = np.load("center_grid.npy")

def find_best_group_greedy(grid):
    rows, cols = grid.shape
    best_group = None
    max_apples = 0
    min_area = float("inf")

    for height in range(rows, 0, -1):
        for width in range(cols, 0, -1):
            area = height * width
            for r1 in range(rows - height + 1):
                for c1 in range(cols - width + 1):
                    r2 = r1 + height - 1
                    c2 = c1 + width - 1
                    region = grid[r1:r2+1, c1:c2+1]
                    valid_values = region[(region != -1) & (region != 0)]

                    if valid_values.size > 0 and np.sum(valid_values) == 10:
                        if (valid_values.size > max_apples) or \
                           (valid_values.size == max_apples and area < min_area):
                            max_apples = valid_values.size
                            min_area = area
                            best_group = (r1, c1, r2, c2)

    return best_group


# ✅ 드래그 실행 함수
def drag_apples(positions):
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    x1, y1 = min(x_coords) - 15, min(y_coords) - 15
    x2, y2 = max(x_coords) + 15, max(y_coords) + 15

    driver.execute_script(f"""
        const canvas = document.getElementById('canvas');
        const rect = canvas.getBoundingClientRect();
        const startX = rect.left + {x1};
        const startY = rect.top + {y1};
        const endX = rect.left + {x2};
        const endY = rect.top + {y2};

        const down = new MouseEvent('mousedown', {{
            bubbles: true, cancelable: true, view: window,
            clientX: startX, clientY: startY
        }});
        const move = new MouseEvent('mousemove', {{
            bubbles: true, cancelable: true, view: window,
            clientX: endX, clientY: endY
        }});
        const up = new MouseEvent('mouseup', {{
            bubbles: true, cancelable: true, view: window,
            clientX: endX, clientY: endY
        }});

        canvas.dispatchEvent(down);
        canvas.dispatchEvent(move);
        canvas.dispatchEvent(up);
    """)

# ✅ 드래그 반복 실행
drag_count = 0
while True:
    group = find_best_group_greedy(ocr_result)
    if group is None:
        break

    r1, c1, r2, c2 = group
    positions = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if ocr_result[r, c] not in (-1, 0):
                positions.append(center_grid[r, c])

    if positions:
        drag_apples(positions)
        drag_count += 1
        print(f"🟢 드래그 실행 {drag_count}회차: ({r1},{c1}) → ({r2},{c2})")

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if ocr_result[r, c] not in (-1, 0):
                    ocr_result[r, c] = 0

        time.sleep(0.03)

print(f"✅ 모든 드래그 완료! 총 {drag_count}회 실행")
input("게임 종료 후 Enter를 눌러 브라우저를 닫습니다...")
driver.quit()