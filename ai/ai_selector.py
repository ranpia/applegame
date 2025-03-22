import torch
import numpy as np
from model.fruit_group_model import GroupPredictor

model = GroupPredictor()
path = "model/model_weights.pth"
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

# 🔮 현재 보드 상태 (10x17) → 마스크 (10x17) 예측
def predict_mask_ai(board_2d: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(board_2d, dtype=torch.float32).view(1, 1, 10, 17)
        out = model(x).squeeze().numpy()
        mask = (out > 0.5).astype(np.uint8)  # 확률 0.5 이상을 선택된 사과로 판단
        return mask