from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import timm
import os
import gdown
import torch.serialization

app = FastAPI()

# Create model architecture
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)

# Load weights
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)

model.eval()
# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "full_model_eff.pth"

# ✅ ONLY FILE ID (not full link)
FILE_ID = "1_5eZPAan9XfL9NCroBqZJF2vty8Pve7s"

# -----------------------------
# ALLOW TIMM MODEL
# -----------------------------
torch.serialization.add_safe_globals(
    [timm.models.efficientnet.EfficientNet]
)

# -----------------------------
# DOWNLOAD MODEL (SAFE)
# -----------------------------
def download_model():
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete.")

# -----------------------------
# CHECK + DOWNLOAD
# -----------------------------
if not os.path.exists(MODEL_PATH):
    download_model()

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Model load failed:", e)
    raise RuntimeError("Model file is invalid or corrupted")

# -----------------------------
# CLASSES
# -----------------------------
class_names = [
    'Anthracnose fruit',
    'Anthracnose leaf',
    'Bacterial Canker fruit',
    'Gall_Mid leaf',
    'Healthy fruit',
    'Healthy leaf'
]

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"status": "API running", "model_loaded": True}

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
