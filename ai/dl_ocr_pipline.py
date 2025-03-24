import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from model.ocr_cnn import OCR_CNN

# 🔧 환경 설정
model_path = "model/ocr_cnn.pth"
image_path = "data/game_screenshot.png"
center_grid_path = "data/center_grid.npy"
output_csv = "data/ocr_result_corrected_10x17.csv"

# 🧠 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OCR_CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 딥러닝 OCR 모델 로드 완료")

# 🔁 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🖼️ 스크린샷 불러오기
image = cv2.imread(image_path)
if image is None:
    print(f"❌ 이미지 로딩 실패: {image_path}")
    exit()

# 🟠 사과 중심 좌표 불러오기
try:
    center_grid = np.load(center_grid_path)  # shape (10, 17, 2)
except:
    print(f"❌ 중심 좌표 로딩 실패: {center_grid_path}")
    exit()

# 🔍 숫자 예측
ocr_result = np.full((10, 17), -1, dtype=int)

for i in range(10):
    for j in range(17):
        x, y = center_grid[i, j]
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + margin), min(image.shape[0], y + margin)

        roi = image[y1:y2, x1:x2]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # 전처리 및 예측
        input_tensor = transform(roi_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            ocr_result[i, j] = pred+1

# ✅ CSV 저장
df = pd.DataFrame(ocr_result)
df.to_csv(output_csv, index=False, header=False)
print(f"✅ OCR 결과 저장 완료 → {output_csv}")
