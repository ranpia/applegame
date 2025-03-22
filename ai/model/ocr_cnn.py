import torch.nn as nn
import torch.nn.functional as F

class OCRCNN(nn.Module):
    def __init__(self):
        super(OCRCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1, 32, 32) -> (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (32, 32, 32) -> (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> (64, 8, 8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # 🔸여기 중요!
            nn.ReLU(),
            nn.Linear(128, 10)  # 숫자 0~9 분류
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
