import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.ocr_cnn import OCR_CNN  # ✅ 외부 모델 불러오기

# 학습 환경
data_dir = "data/ocr_data"
save_path = "model/ocr_cnn.pth"
batch_size = 32
epochs = 30
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 로드
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 불러오기
model = OCR_CNN(use_softmax=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
print("🚀 OCR CNN 학습 시작...\n")
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print(f"[Epoch {epoch+1:02d}] Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

# 저장
torch.save(model.state_dict(), save_path)
print(f"\n✅ 모델 저장 완료: {save_path}")
