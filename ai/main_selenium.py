import numpy as np
import time
import os
from dqn_env import FruitBoxEnv
from dqn_agent import DQNAgent
from utils.group_generator import generate_all_valid_groups
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2
import torch
import matplotlib.pyplot as plt

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
board = np.loadtxt("data/ocr_result_corrected_10x17.csv", delimiter=",", dtype=int)

# 🎯 DQN 학습 시작
from dqn_env import FruitBoxEnv
from dqn_agent import DQNAgent

EPISODES = 1
ACTION_SAMPLE_LIMIT = 100
TARGET_SCORE = 160

env = FruitBoxEnv(board)
action_list = env.get_action_list()
agent = DQNAgent(action_count=max(len(action_list), 1))

rewards_log = []
epsilon_log = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action_list = env.get_action_list()
        if len(action_list) == 0:
            break

        action_idx = agent.select_action(state)
        action_idx = min(action_idx, len(action_list) - 1)
        group = action_list[action_idx]

        next_state, reward, done = env.step(group)

        agent.remember(state, action_idx, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward
        steps += 1

        if done or steps >= ACTION_SAMPLE_LIMIT:
            break

    agent.update_target_network()
    rewards_log.append(total_reward)
    epsilon_log.append(agent.epsilon)

    print(f"[EP {episode+1}] 총 보상: {total_reward}, 단계 수: {steps}, epsilon: {agent.epsilon:.4f}")

    if total_reward >= TARGET_SCORE:
        print("🎉 목표 점수 도달! 학습 조기 종료")
        break

# ✅ 모델 저장
model_path = "model/conv_dqn_weights.pth"
torch.save(agent.model.state_dict(), model_path)
print(f"✅ 학습된 모델 저장 완료: {model_path}")

# 📊 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards_log)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(epsilon_log)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")

plt.tight_layout()
plt.savefig("data/training_progress.png")
print("📈 학습 시각화 저장 완료: training_progress.png")

input("게임 종료 후 Enter를 눌러 브라우저를 닫습니다...")
driver.quit()