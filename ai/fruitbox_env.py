import numpy as np
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2

class FruitBoxEnv:
    def __init__(self, chrome_path="chromedriver-win64/chromedriver.exe"):
        self.chrome_path = chrome_path
        self.driver = None
        self.canvas = None
        self.state = None
        self.center_grid = None

    def _init_browser(self):
        chrome_options = Options()
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--ignore-ssl-errors")
        service = Service(self.chrome_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.get("https://www.gamesaien.com/game/fruit_box_a/")
        time.sleep(2)

    def reset(self):
        # 1. 크롬 새로 열기
        if self.driver:
            self.driver.quit()
        self._init_browser()

        # 2. 캔버스 찾기 및 play 클릭
        self.canvas = self.driver.find_element(By.ID, "canvas")
        rect = self.driver.execute_script("""
            const rect = document.getElementById('canvas').getBoundingClientRect();
            return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
        """)
        canvas_left = int(rect["left"])
        canvas_top = int(rect["top"])
        canvas_width = int(rect["width"])
        canvas_height = int(rect["height"])

        play_x = canvas_left + (canvas_width // 2) - 170
        play_y = canvas_top + (canvas_height // 2)
        self.driver.execute_script(f"""
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
        print("✅ 게임 시작")

        # 3. 캡처 및 OCR 실행
        time.sleep(2)
        self.canvas.screenshot("data/game_screenshot.png")
        print("📸 스크린샷 저장 완료")

        # 4. 사과 중심 계산
        self._detect_apples()

        # 5. OCR 파이프라인 실행
        os.system("python ocr_pipline.py")
        self.state = np.loadtxt("data/ocr_result_corrected_10x17.csv", delimiter=",", dtype=int)

        return self.state.copy()

    def step(self, action):
        """
        action: 마스크 배열 (10x17) 값이 1인 위치만 드래그
        """
        mask = action
        indices = np.argwhere(mask == 1)
        values = np.array([self.state[r, c] for r, c in indices if self.state[r, c] > 0])
        reward = 0

        if values.size > 0 and np.sum(values) == 10:
            from drag_logic import drag_apples
            positions = [self.center_grid[r, c] for r, c in indices]
            drag_apples(positions, self.driver)
            for r, c in indices:
                self.state[r, c] = 0
            reward = values.size
            print(f"🍎 사과 {reward}개 제거 (합 10)")

        done = not self._has_more_groups()
        return self.state.copy(), reward, done

    def _has_more_groups(self):
        for r1 in range(10):
            for c1 in range(17):
                for r2 in range(r1, 10):
                    for c2 in range(c1, 17):
                        region = self.state[r1:r2+1, c1:c2+1]
                        values = region[(region != -1) & (region != 0)]
                        if values.size > 0 and np.sum(values) == 10:
                            return True
        return False

    def _detect_apples(self):
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

        sorted_centers = sorted(circle_centers, key=lambda c: (c[1], c[0]))[:170]
        self.center_grid = np.array(sorted_centers, dtype=np.int32).reshape((10, 17, 2))
        np.save("data/center_grid.npy", self.center_grid)
        print("✅ 중심 좌표 계산 완료")
