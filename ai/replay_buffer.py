import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)  # (B, 1, 10, 17)
        actions = torch.tensor(actions, dtype=torch.long)             # (B,)
        rewards = torch.tensor(rewards, dtype=torch.float32)          # (B,)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # (B, 1, 10, 17)
        dones = torch.tensor(dones, dtype=torch.float32)              # (B,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Example usage
if __name__ == '__main__':
    buffer = ReplayBuffer()
    for _ in range(5):
        buffer.push(np.random.rand(1, 10, 17), random.randint(0, 9), 1.0, np.random.rand(1, 10, 17), False)
    s, a, r, s2, d = buffer.sample(2)
    print("샘플 상태 shape:", s.shape)  # Expected: (2, 1, 10, 17)