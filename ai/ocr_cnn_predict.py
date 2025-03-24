import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model.ocr_cnn import OCR_CNN  # 모델 정의 파일

# ======= 설정 =======
model_path = "model/ocr_cnn.pth"
image_path = "data/ocr_data/7/img_1_15.png"  # 테스트할 이미지 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= 이미지 전처리 =======
transform = transforms.Compose([
    transforms.Grayscale(),           # 흑백
    transforms.Resize((32, 32)),      # 모델 입력 크기에 맞춤
    transforms.ToTensor(),            # 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

# ======= 모델 로드 =======
model = OCR_CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 모델 로드 완료")

# ======= 예측 =======
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

predicted_class = np.argmax(probabilities)
predicted_label = predicted_class + 1  # 👉 클래스 0~8 → 라벨 1~9
confidence = probabilities[predicted_class]

print(f"🔢 예측된 숫자: {predicted_label} | 🔒 확신도: {confidence:.4f}")
print("📊 전체 확률 분포 (라벨 기준):")
for i, prob in enumerate(probabilities):
    print(f" - {i+1}: {prob:.4f}")
