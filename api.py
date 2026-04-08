from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import timm
import os
import requests

# 🔥 Reduce memory usage (critical for Render free tier)
torch.set_num_threads(1)

# Optional safe globals (harmless)
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

app = FastAPI()

# -------------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------------
MODEL_PATH = "full_model_eff.pth"

# 🔁 Replace this URL with YOUR own GitHub Releases direct download link
# (see instructions below on how to create it)
MODEL_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/full_model_eff.pth"

def download_model():
    """Download the model from GitHub Releases if not present."""
    print(f"Downloading model from {MODEL_URL} ...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

if not os.path.exists(MODEL_PATH):
    download_model()

# 🔥 Load the full model (weights_only=False because it's a full object)
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()

# -------------------------------------------------------------------
# CLASS NAMES & TRANSFORM
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "Mango Disease API is running"}

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
