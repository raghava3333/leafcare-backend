from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import timm
import os
import gdown
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

app = FastAPI()

# 🔥 Model download
MODEL_PATH = "full_model_eff.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1_5eZPAan9XfL9NCroBqZJF2vty8Pve7s"
    gdown.download(url, MODEL_PATH, quiet=False)

# 🔥 Load model
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()

# Classes
class_names = [
    'Anthracnose fruit',
    'Anthracnose leaf',
    'Bacterial Canker fruit',
    'Gall_Mid leaf',
    'Healthy fruit',
    'Healthy leaf'
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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
