import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from model.fruit_group_model import GroupPredictor

def load_data(csv_path="data/training_from_board.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' 파일이 존재하지 않습니다.")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.shape[1] != 340:
        raise ValueError(f"CSV 열 수가 340이 아닙니다. 현재: {data.shape[1]}")

    X = data[:, :170].reshape(-1, 1, 10, 17)
    Y = data[:, 170:].reshape(-1, 1, 10, 17)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_model(epochs=20, batch_size=64):
    X, Y = load_data()
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GroupPredictor()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n🚀 모델 학습 시작...")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch:02d}] Loss: {total_loss / len(loader):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/model_weights.pth")
    print("\n✅ 모델 학습 완료 및 저장: model/model_weights.pth")

if __name__ == "__main__":
    train_model()