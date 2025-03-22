import numpy as np
import torch
from dqn_env import FruitBoxEnv
from dqn_agent import DQNAgent
from utils.group_generator import generate_all_valid_groups
import matplotlib.pyplot as plt

# 하이퍼파라미터
EPISODES = 1000
TARGET_SCORE = 160
ACTION_SAMPLE_LIMIT = 100  # 최대 행동 수 제한

# 초기 상태 불러오기
board = np.loadtxt("data/ocr_result_corrected_10x17.csv", delimiter=",", dtype=int)
env = FruitBoxEnv(board)

# 초기 유효 그룹
action_list = env.get_action_list()
print(f"🎯 가능한 그룹 개수: {len(action_list)}")

# 에이전트 초기화
agent = DQNAgent(action_count=len(action_list))

# 기록용
rewards_log = []
epsilon_log = []

# 학습 루프
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action_list = env.get_action_list()
        if len(action_list) == 0:
            break

        # 현재 action_list를 기준으로 인덱스 선택
        action_idx = agent.select_action(state)
        action_idx = min(action_idx, len(action_list) - 1)  # out of range 방지
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

# 📈 시각화
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
plt.savefig("training_progress.png")
print("📊 학습 과정 시각화 저장 완료: training_progress.png")