from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import timm
import os
import sys
import requests

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
torch.set_num_threads(1)  # Reduce memory usage
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

app = FastAPI()

MODEL_PATH = "full_model_eff.pth"

# 🔁 REPLACE THIS URL WITH YOUR ACTUAL GITHUB RELEASES DIRECT LINK
MODEL_URL = "https://drive.google.com/file/d/1_5eZPAan9XfL9NCroBqZJF2vty8Pve7s/view?usp=drivesdk"

# -------------------------------------------------------------------
# HELPER: Check if file is a valid PyTorch model
# -------------------------------------------------------------------
def is_valid_model_file(path):
    """Return True if file exists and starts with PyTorch ZIP magic bytes."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        header = f.read(4)
        # PyTorch models are ZIP files (PK..) or legacy (0x80...)
        return header[:2] == b'PK' or header[0] == 0x80

# -------------------------------------------------------------------
# HELPER: Download model from URL (with progress)
# -------------------------------------------------------------------
def download_model(url, dest_path):
    print(f"Downloading model from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# -------------------------------------------------------------------
# LOAD OR DOWNLOAD MODEL
# -------------------------------------------------------------------
if not is_valid_model_file(MODEL_PATH):
    print("Model file missing or corrupt. Attempting to download...")
    try:
        download_model(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please ensure the MODEL_URL is correct and the file is publicly accessible.")
        sys.exit(1)

    # Verify again after download
    if not is_valid_model_file(MODEL_PATH):
        print("Downloaded file is still invalid. Check the URL or file source.")
        sys.exit(1)

# Load model
print("Loading model...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()
print("Model loaded successfully.")

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
    return {"status": "API is running", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    try:
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
    except Exception as e:
        return {"error": str(e)}
