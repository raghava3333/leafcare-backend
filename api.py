from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import os
import requests

app = FastAPI()

MODEL_PATH = "full_model_eff.pth"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1_5eZPAan9XfL9NCroBqZJF2vty8Pve7s"

# --------------------------------------------------
# ✅ CHECK FILE VALIDITY
# --------------------------------------------------
def is_valid_model(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        header = f.read(2)
        return header == b'PK' or header[0] == 0x80


# --------------------------------------------------
# ✅ DOWNLOAD MODEL
# --------------------------------------------------
def download_model():
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    print("Download complete.")


# --------------------------------------------------
# ✅ ENSURE MODEL EXISTS & VALID
# --------------------------------------------------
if not is_valid_model(MODEL_PATH):
    print("Model missing or corrupted. Downloading...")
    download_model()

    if not is_valid_model(MODEL_PATH):
        raise RuntimeError("Downloaded file is not a valid PyTorch model")


# --------------------------------------------------
# ✅ LOAD FULL MODEL
# --------------------------------------------------
print("Loading model...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()
print("Model loaded successfully.")


# --------------------------------------------------
# CLASSES
# --------------------------------------------------
class_names = [
    'Anthracnose fruit',
    'Anthracnose leaf',
    'Bacterial Canker fruit',
    'Gall_Mid leaf',
    'Healthy fruit',
    'Healthy leaf'
]

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "API running", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
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

    except Exception as e:
        return {"error": str(e)}
