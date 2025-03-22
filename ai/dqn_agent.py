import torch
import torch.nn as nn
import torch.optim as optim
import random
from model.conv_dqn import ConvDQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, action_count, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.action_count = action_count
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = ConvDQN(action_count)
        self.target_model = ConvDQN(action_count)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 1, 10, 17)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_target = q_values.clone()
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * torch.max(next_q_values[i]).item()
            q_target[i, actions[i]] = target

        loss = self.criterion(q_values, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Test run
if __name__ == '__main__':
    agent = DQNAgent(action_count=100)
    dummy_state = torch.randn(1, 10, 17).numpy()
    action = agent.select_action(dummy_state)
    print("선택된 action:", action)
