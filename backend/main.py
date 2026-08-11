from __future__ import annotations

import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "crop_classifier.pt"
LABELS_PATH = ROOT / "artifacts" / "labels.json"

app = FastAPI(title="CropShield AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_labels: list[str] = []

def load_model():
    global _model, _labels
    if _model is not None:
        return
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise HTTPException(503, "Model is not trained yet. Run backend/train.py first.")
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    _labels = json.loads(LABELS_PATH.read_text())
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(_labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model

def recommendation(label: str):
    crop, disease = (label.replace("___", " ").split(" ", 1) + ["Healthy"])[:2]
    healthy = "healthy" in label.lower()
    return {"crop": crop.replace("_", " "), "disease": disease.replace("_", " "), "description": "No disease pattern detected." if healthy else "Detected visual pattern matches a trained PlantVillage disease class.", "organic": "Neem Oil Spray" if not healthy else "Continue regular monitoring", "chemical": "Mancozeb 75 WP" if not healthy else "No chemical treatment needed", "severity": 0 if healthy else 72}

@app.get("/health")
def health():
    return {"ready": MODEL_PATH.exists() and LABELS_PATH.exists(), "model": "MobileNetV3-Small"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Upload a valid image file.")
    load_model()
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights
    try:
        leaf = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, "Could not read image.") from e
    tensor = MobileNet_V3_Small_Weights.DEFAULT.transforms()(leaf).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(_model(tensor)[0], dim=0)
    confidence, index = torch.max(probabilities, 0)
    label = _labels[index.item()]
    return {**recommendation(label), "label": label, "confidence": round(float(confidence) * 100, 1)}
