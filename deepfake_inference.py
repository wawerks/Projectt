# ======================================================
# deepfake_inference.py — Enhanced Deepfake Face Detection (Fixed Version)
# ======================================================

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    SiglipForImageClassification,
    AutoImageProcessor
)
from PIL import Image, ImageEnhance
import torch
import numpy as np
import io
import requests

# ------------------------------------------------------
# MODEL CONFIGURATIONS
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load Vision Transformer (ViT)
vit_name = "Wvolf/ViT_Deepfake_Detection"
vit_model = ViTForImageClassification.from_pretrained(vit_name).to(device)
vit_processor = ViTImageProcessor.from_pretrained(vit_name)

# Load SigLIP model
siglip_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
siglip_model = SiglipForImageClassification.from_pretrained(siglip_name).to(device)
siglip_processor = AutoImageProcessor.from_pretrained(siglip_name)


# ------------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# ------------------------------------------------------
def preprocess_image(image: Image.Image):
    """
    Preprocess image before feeding into model.
    - Resize to 224x224
    - Convert to RGB
    - Slightly enhance contrast
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))

    # Slight contrast enhancement for better feature distinction
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    return image


# ------------------------------------------------------
# ENSEMBLE DECISION FUNCTION
# ------------------------------------------------------
def classify_with_ensemble(results):
    """
    Ensemble classification logic using stricter confidence rules.
    results: list of dicts like
        [{"label": "REAL", "confidence": 0.83},
         {"label": "FAKE", "confidence": 0.75}]
    """

    real_conf = np.mean([r["confidence"] for r in results if r["label"] == "REAL"]) if any(
        r["label"] == "REAL" for r in results) else 0
    fake_conf = np.mean([r["confidence"] for r in results if r["label"] == "FAKE"]) if any(
        r["label"] == "FAKE" for r in results) else 0

    diff = abs(real_conf - fake_conf)
    max_conf = max(real_conf, fake_conf)

    # Apply strict rules
    if fake_conf > real_conf and diff > 0.05:
        final_label = "FAKE"
    elif real_conf >= 0.75 and real_conf > fake_conf:
        final_label = "REAL"
    elif 0.5 <= max_conf < 0.75 or diff < 0.1:
        final_label = "UNCERTAIN"
    else:
        final_label = "FAKE"

    return {
        "final_label": final_label,
        "real_confidence": round(real_conf * 100, 2),
        "fake_confidence": round(fake_conf * 100, 2)
    }


# ------------------------------------------------------
# CLASSIFICATION FUNCTION
# ------------------------------------------------------
def classify_image(image: Image.Image):
    """
    Classify image using both ViT and SigLIP models.
    """

    image = preprocess_image(image)

    # -----------------------------
    # Vision Transformer (ViT)
    # -----------------------------
    vit_inputs = vit_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vit_outputs = vit_model(**vit_inputs)
    vit_logits = torch.nn.functional.softmax(vit_outputs.logits, dim=-1)[0]
    vit_conf, vit_pred = torch.max(vit_logits, dim=-1)
    vit_label = vit_model.config.id2label[vit_pred.item()]
    vit_conf = vit_conf.item()

    # -----------------------------
    # SigLIP Model
    # -----------------------------
    sig_inputs = siglip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        sig_outputs = siglip_model(**sig_inputs)
    sig_logits = torch.nn.functional.softmax(sig_outputs.logits, dim=-1)[0]
    sig_conf, sig_pred = torch.max(sig_logits, dim=-1)
    sig_label = siglip_model.config.id2label[sig_pred.item()]
    sig_conf = sig_conf.item()

    # -----------------------------
    # Weighted Ensemble
    # -----------------------------
    vit_weight = 0.6
    sig_weight = 0.4

    results = [
        {"label": vit_label.upper(), "confidence": vit_conf * vit_weight},
        {"label": sig_label.upper(), "confidence": sig_conf * sig_weight}
    ]

    final_result = classify_with_ensemble(results)

    # Display final decision
    print("\n=================== RESULT ===================")
    print(f"ViT Prediction: {vit_label.upper()} ({vit_conf*100:.2f}%)")
    print(f"SigLIP Prediction: {sig_label.upper()} ({sig_conf*100:.2f}%)")
    print(f"----------------------------------------------")
    print(f"FINAL RESULT: {final_result['final_label']}")
    print(f"Real Confidence: {final_result['real_confidence']}%")
    print(f"Fake Confidence: {final_result['fake_confidence']}%")
    print("==============================================\n")

    return final_result


# ------------------------------------------------------
# IMAGE INPUT HANDLING
# ------------------------------------------------------
def load_image(input_path):
    """
    Load an image either from URL or file path.
    """
    if input_path.startswith("http://") or input_path.startswith("https://"):
        response = requests.get(input_path)
        image = Image.open(io.BytesIO(response.content))
    else:
        image = Image.open(input_path)
    return image


# ------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print(" AI-Based Face Manipulation Detection System ")
    print("==============================================")

    user_input = input("\nEnter image file path or URL: ").strip()
    try:
        img = load_image(user_input)
        result = classify_image(img)

        label = result["final_label"]
        if label == "REAL":
            print("✅ Result: Real Face Detected")
        elif label == "FAKE":
            print("❌ Result: Manipulated Face Detected")
        else:
            print("⚠️ Result: Uncertain — Possible Manipulation Detected")

    except Exception as e:
        print(f"Error: {e}")
