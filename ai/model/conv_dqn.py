import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDQN(nn.Module):
    def __init__(self, action_count):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 10 * 17, 256)
        self.fc2 = nn.Linear(256, action_count)  # Output: Q-values for each action

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 16, 10, 17)
        x = F.relu(self.conv2(x))  # (B, 32, 10, 17)
        x = F.relu(self.conv3(x))  # (B, 64, 10, 17)
        x = self.flatten(x)        # (B, 64*10*17)
        x = F.relu(self.fc1(x))    # (B, 256)
        return self.fc2(x)         # (B, action_count)


def generate_all_valid_groups(board):
    """
    board: np.array of shape (10, 17) with integer values (1~9 or 0/-1)
    Returns: List of groups (each group is list of coordinates [(r, c), ...]) where the sum == 10
    """
    valid_groups = []
    rows, cols = board.shape
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    group = []
                    total = 0
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            val = board[r, c]
                            if val not in (0, -1):
                                total += val
                                group.append((r, c))
                    if total == 10 and len(group) > 0:
                        valid_groups.append(group)
    return valid_groups

# Example usage
if __name__ == "__main__":
    import numpy as np
    dummy_board = np.random.randint(1, 10, size=(10, 17))
    groups = generate_all_valid_groups(dummy_board)
    print(f"Found {len(groups)} valid groups with sum 10")
