import torch.nn as nn

class OCR_CNN(nn.Module):
    def __init__(self, use_softmax=False):
        super(OCR_CNN, self).__init__()
        self.use_softmax = use_softmax
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 1x32x32 → 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # → 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 9)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.use_softmax:
            return self.softmax(x)
        return x

