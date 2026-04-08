from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch

app = FastAPI()

# Load model directly (NOW from repo)
MODEL_PATH = "full_model_eff.pth"

model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()

class_names = [
    'Anthracnose fruit',
    'Anthracnose leaf',
    'Bacterial Canker fruit',
    'Gall_Mid leaf',
    'Healthy fruit',
    'Healthy leaf'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    predicted_idx = torch.argmax(probs).item()
    confidence = probs[predicted_idx].item()
    label = class_names[predicted_idx]

    return {
        "disease": label,
        "confidence": f"{confidence*100:.2f}%"
    }
