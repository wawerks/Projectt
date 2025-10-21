from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import SiglipForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Define available models
MODEL_CONFIGS = {
    "vit": {
        "name": "prithivMLmods/Deepfake-Detection-Exp-02-21",
        "loader": (ViTForImageClassification, ViTImageProcessor)
    },
    "siglip": {
        "name": "prithivMLmods/open-deepfake-detection",
        "loader": (SiglipForImageClassification, AutoImageProcessor)
    }
}

_loaded = {}

def load_model(model_key: str):
    """Loads a model and processor only once."""
    if model_key not in _loaded:
        cfg = MODEL_CONFIGS[model_key]
        print(f"✅ Loading {model_key} model: {cfg['name']}")
        ModelClass, ProcessorClass = cfg["loader"]
        model = ModelClass.from_pretrained(cfg["name"])
        processor = ProcessorClass.from_pretrained(cfg["name"])
        _loaded[model_key] = (model, processor)
    return _loaded[model_key]


def classify_single(image: Image.Image, model_key: str):
    """Run inference on one model."""
    model, processor = load_model(model_key)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=1).item())

    label = model.config.id2label.get(idx, "Unknown")
    score = float(probs[idx])
    return label, score


def classify_ensemble(image: Image.Image):
    """Run both models and combine predictions."""
    results = {}
    for key in MODEL_CONFIGS.keys():
        try:
            label, score = classify_single(image, key)
            results[key] = {"label": label, "score": score}
        except Exception as e:
            print(f"❌ {key} model failed: {e}")
            results[key] = {"label": "Error", "score": 0.0}

    # Majority vote logic
    labels = [v["label"] for v in results.values()]
    final_label = max(set(labels), key=labels.count)
    avg_conf = sum(v["score"] for v in results.values()) / len(results)

    return final_label, avg_conf, results
