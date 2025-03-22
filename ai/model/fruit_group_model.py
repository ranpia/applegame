import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupPredictor(nn.Module):
    def __init__(self):
        super(GroupPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 10 * 17, 512)
        self.fc2 = nn.Linear(512, 170)  # Output mask (flattened 10x17)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (B, 32, 10, 17)
        x = F.relu(self.conv2(x))       # (B, 64, 10, 17)
        x = F.relu(self.conv3(x))       # (B, 128, 10, 17)
        x = self.flatten(x)             # (B, 128 * 10 * 17)
        x = F.relu(self.fc1(x))         # (B, 512)
        x = torch.sigmoid(self.fc2(x))  # (B, 170), output mask
        return x

# For testing
if __name__ == '__main__':
    model = GroupPredictor()
    dummy_input = torch.randn(1, 1, 10, 17)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 170)
