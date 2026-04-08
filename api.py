from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import torch.nn as nn
import os
import gdown
import timm

app = FastAPI()

# 🔥 Model path
MODEL_PATH = "weights.pth"

# 🔥 Download model
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=YOUR_NEW_FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)

# 🔥 Create model architecture
model = timm.create_model('efficientnet_b0', pretrained=False)

# 🔥 Modify final layer (IMPORTANT: 6 classes)
model.classifier = nn.Linear(model.classifier.in_features, 6)

# 🔥 Load weights
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)

model.eval()

# 🔥 Classes
class_names = [
    'Anthracnose fruit',
    'Anthracnose leaf',
    'Bacterial Canker fruit',
    'Gall_Mid leaf',
    'Healthy fruit',
    'Healthy leaf'
]

# 🔥 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔥 API
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
