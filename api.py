from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import timm
import os
import requests

app = FastAPI()

MODEL_PATH = "full_model_eff.pth"

# ✅ CORRECT DIRECT DOWNLOAD LINK
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_5eZPAan9XfL9NCroBqZJF2vty8Pve7s"

# 🔥 Download model if missing
def download_model():
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    print("Download complete.")

# 🔥 Check and download
if not os.path.exists(MODEL_PATH):
    download_model()

# 🔥 Create model architecture
model = timm.create_model('efficientnet_b0', pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 6)

# 🔥 Load weights (SAFE WAY)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)

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

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    predicted_idx = torch.argmax(probs).item()

    return {
        "disease": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx])
    }
