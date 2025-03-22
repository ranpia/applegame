import torch
from torchvision import transforms
from PIL import Image
import os
from model.ocr_cnn import OCRCNN  # ✅ 너가 저장한 모델 정의 파일

# 🔧 환경 설정
model_path = "model/ocr_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 전처리
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🧠 모델 로드
model = OCRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 모델 로드 완료")

# 🔍 예측 함수
def predict_digit(img_path):
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return predicted

# ✅ 테스트 실행
if __name__ == "__main__":
    # 예시: 숫자 이미지 파일 하나 테스트
    test_image = "data/ocr_data/3/img_2_15.png"  # ← 여기에 테스트할 이미지 경로 넣기
    if os.path.exists(test_image):
        prediction = predict_digit(test_image)
        print(f"🔢 예측된 숫자: {prediction}")
    else:
        print(f"❌ 파일이 존재하지 않습니다: {test_image}")
