# ======================================================
# server.py — Enhanced Deepfake Face Manipulation Detector (Clean ViT Integration)
# ======================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io
import requests
import cv2
import os
import numpy as np

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
# MODEL CONFIGURATION
# ------------------------------------------------------
MODEL_PATHS = {
    "vit_deepfake": "Wvolf/ViT_Deepfake_Detection",
    "vit_alt": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "vit_finetuned": "train/fine_tuned_vit_rvf10k"

}

# MODEL_PATHS = {
#     "vit_finetuned": "train/fine_tuned_vit_rvf10k"
# }


MODELS, PROCESSORS = {}, {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Loading specialized deepfake face manipulation models...")
for name, path in MODEL_PATHS.items():
    try:
        model = ViTForImageClassification.from_pretrained(path).to(device)
        processor = ViTImageProcessor.from_pretrained(path)

        MODELS[name] = model
        PROCESSORS[name] = processor
        print(f"✅ Loaded {name} model from {path}")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

print("✅ Model loading complete.\n")


# ------------------------------------------------------
# IMAGE CLASSIFICATION FUNCTION
# ------------------------------------------------------
def classify_image_with_model_from_pil(model_name, image: Image.Image):
    model = MODELS[model_name]
    processor = PROCESSORS[model_name]

    image = image.convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        confidence, predicted = torch.max(logits, dim=-1)
        confidence = confidence.item()
        label = model.config.id2label[predicted.item()].lower()

    if "fake" in label or "synthetic" in label or "manipulated" in label:
        normalized_label = "FAKE"
    elif "real" in label or "authentic" in label:
        normalized_label = "REAL"
    else:
        normalized_label = "UNKNOWN"

    return {
        "model": model_name,
        "label": normalized_label,
        "confidence": confidence
    }


# ------------------------------------------------------
# ENSEMBLE DECISION (Fake Threshold Override Logic)
# ------------------------------------------------------
def ensemble_decision(results):
    real_conf = np.mean([r["confidence"] for r in results if r["label"] == "REAL"]) if any(
        r["label"] == "REAL" for r in results) else 0
    fake_conf = np.mean([r["confidence"] for r in results if r["label"] == "FAKE"]) if any(
        r["label"] == "FAKE" for r in results) else 0

    diff = abs(real_conf - fake_conf)
    max_conf = max(real_conf, fake_conf)

    # ✅ Override: If fake confidence ≥ 0.75, final decision = Deepfake
    if fake_conf >= 0.70:
        final_label = "❌ Deepfake"
    elif real_conf >= 0.70 and real_conf > fake_conf:
        final_label = "✅ Real"
    elif 0.5 <= max_conf < 0.70 or diff < 0.1:
        final_label = "⚠️ Uncertain — Possible Manipulation"
    else:
        final_label = "❌ Deepfake"

    return {
        "final_label": final_label,
        "real_confidence": round(real_conf * 100, 2),
        "fake_confidence": round(fake_conf * 100, 2)
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
# ROUTES
# ------------------------------------------------------
class ImageURL(BaseModel):
    url: str


@app.post("/classify_url")
async def classify_url(data: ImageURL):
    try:
        resp = requests.get(data.url)
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        image_path = "temp_url.jpg"
        image.save(image_path)
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    if not is_human_face(image_path):
        return {"message": "❌ No human face detected. Please upload a valid face image."}

    model_results = [classify_image_with_model_from_pil(m, image) for m in MODELS.keys()]
    ensemble_result = ensemble_decision(model_results)

    return {
        "status": "success",
        "model_results": model_results,
        "final_decision": ensemble_result
    }


@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = "temp_upload.jpg"
        image.save(image_path)
    except Exception as e:
        return {"error": f"Failed to process uploaded image: {e}"}

    if not is_human_face(image_path):
        return {"message": "❌ No human face detected. Please upload a valid face image."}

    model_results = [classify_image_with_model_from_pil(m, image) for m in MODELS.keys()]
    ensemble_result = ensemble_decision(model_results)

    return {
        "status": "success",
        "model_results": model_results,
        "final_decision": ensemble_result
    }


@app.post("/detect_face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return {"face_detected": len(faces) > 0}


@app.get("/")
async def root():
    return {"message": "🔥 AI-Based Face Manipulation Detection API is running!"}


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Deepfake Detection Server...")
    print("🌐 Server running at http://127.0.0.1:8000")
    print("🛑 Press Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
