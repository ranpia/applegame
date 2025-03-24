import numpy as np
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2

# 📌 크롬 드라이버 경로
chrome_driver_path = "../../chromedriver-win64/chromedriver.exe"

# 📌 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--ignore-ssl-errors")

# 🔁 반복 횟수 설정
repeat_count = 10

for attempt in range(repeat_count):
    print(f"\n🚀 [{attempt + 1}/{repeat_count}] 게임 자동 실행 시작...")

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
    canvas.screenshot("../game_screenshot.png")
    print("📸 스크린샷 저장 완료: game_screenshot.png")

    # 🔍 사과 중심 좌표 검출
    image = cv2.imread("../game_screenshot.png")
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
        np.save("../center_grid.npy", center_grid)
        print("✅ 사과 중심 좌표 저장 완료 (center_grid.npy)")
    else:
        print(f"⚠️ 감지된 중심 부족: {len(circle_centers)}개")
        driver.quit()
        continue

    # 🔠 OCR 실행
    os.system("python ocr_pipline.py")

    # 🧠 OCR 결과 확인
    if not os.path.exists("../ocr_result_corrected_10x17.csv"):
        print("❌ OCR 결과가 존재하지 않습니다. 스킵합니다.")
        driver.quit()
        continue

    # 📸 숫자 이미지 자동 저장
    os.system("python dataset_generator.py")

    print(f"✅ [{attempt + 1}] 회차 수집 완료 ✅")
    driver.quit()
    time.sleep(1)  # 다음 회차까지 잠깐 대기

print("\n🎉 전체 자동 수집 완료!")
