# ======================================================
# server.py — Specialized Deepfake Face Manipulation Detector (FastAPI + Face Detection)
# ======================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    SiglipForImageClassification,
    AutoImageProcessor
)
from PIL import Image, ImageFile
import torch
import io
import requests
import cv2
import os

# ------------------------------------------------------
# FACE DETECTION FUNCTION (OpenCV)
# ------------------------------------------------------
def is_human_face(image_path):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"⚠️ File not found: {image_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"⚠️ Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# ------------------------------------------------------
# MODEL CONFIGURATION (Face-Manipulation Focused)
# ------------------------------------------------------
MODEL_PATHS = {
    "model_1": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "model_3": "Wvolf/ViT_Deepfake_Detection"  # ✅ Replaced missing model
}

MODELS = {}
PROCESSORS = {}

print("🔄 Loading specialized deepfake face manipulation models...")
for name, path in MODEL_PATHS.items():
    try:
        if "Siglip" in path or "siglip" in path:
            model = SiglipForImageClassification.from_pretrained(path)
            processor = AutoImageProcessor.from_pretrained(path)
        else:
            model = ViTForImageClassification.from_pretrained(path)
            processor = ViTImageProcessor.from_pretrained(path)

        MODELS[name] = model
        PROCESSORS[name] = processor
        print(f"✅ Loaded {name} from {path}")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

print("✅ Model loading complete.\n")

# ------------------------------------------------------
# IMAGE CLASSIFICATION FUNCTION
# ------------------------------------------------------
def classify_image_with_model_from_pil(model_name, image: Image.Image):
    model = MODELS[model_name]
    processor = PROCESSORS[model_name]

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_idx].item()

    raw_label = model.config.id2label.get(predicted_class_idx, "Unknown")

    # 🧠 Normalize label logic for consistent results
    label_lower = raw_label.lower()
    if any(k in label_lower for k in ["real", "authentic"]) or confidence < 0.5:
        label = "✅ Real"
    elif any(k in label_lower for k in ["fake", "deepfake", "ai", "synthetic", "manipulated", "forged"]) or confidence >= 0.5:
        label = "❌ Deepfake"
    else:
        label = "Unknown"

    return {
        "model": model_name,
        "label": label,
        "confidence": round(confidence * 100, 2)
    }

# ------------------------------------------------------
# FINAL DECISION — highest confidence wins
# ------------------------------------------------------
def ensemble_decision(results):
    # Pick the result with the highest confidence
    final_result = max(results, key=lambda r: r["confidence"])
    return {
        "final_label": final_result["label"],
        "confidence": final_result["confidence"]
    }

# ------------------------------------------------------
# FASTAPI SETUP
# ------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Pydantic model for URL input
# ------------------------------------------------------
class ImageURL(BaseModel):
    url: str

# ------------------------------------------------------
# ROUTE — CLASSIFY IMAGE FROM URL
# ------------------------------------------------------
@app.post("/classify_url")
async def classify_url(data: ImageURL):
    try:
        resp = requests.get(data.url)
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        image_path = "temp_url.jpg"
        image.save(image_path)
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    # ✅ Face detection check
    if not is_human_face(image_path):
        return {"message": "❌ No human face detected. Please upload a valid face image."}

    model_results = [classify_image_with_model_from_pil(m, image) for m in MODELS.keys()]
    ensemble_result = ensemble_decision(model_results)

    summary = {
        "Model Results": [
            f"{r['model']} → {r['label']} ({r['confidence']}%)" for r in model_results
        ],
        "Final Decision": f"{ensemble_result['final_label']} ({ensemble_result['confidence']}% confidence)"
    }

    return {
        "status": "success",
        "details": summary,
        "individual_results": model_results,
        "final_decision": ensemble_result
    }

# ------------------------------------------------------
# ROUTE — CLASSIFY IMAGE (Upload)
# ------------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes or len(image_bytes) == 0:
            return {"error": "Uploaded file is empty."}

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = "temp_upload.jpg"
        image.save(image_path)

        # ✅ Face detection check before deepfake detection
        if not is_human_face(image_path):
            return {"message": "❌ No human face detected. Please upload a valid face image."}

    except Exception as e:
        return {"error": f"Failed to process uploaded image: {e}"}

    # Run deepfake detection if a face is found
    model_results = [classify_image_with_model_from_pil(m, image) for m in MODELS.keys()]
    ensemble_result = ensemble_decision(model_results)

    return {
        "status": "success",
        "individual_results": model_results,
        "final_decision": ensemble_result
    }

# ------------------------------------------------------
# ROOT ROUTE
# ------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "🔥 Specialized Deepfake Face Manipulation Detector API with Face Detection is running!"}

@app.post("/detect_face")
async def detect_face(file: UploadFile = File(...)):
    import cv2
    import numpy as np
    from io import BytesIO

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return {"face_detected": len(faces) > 0}
