import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import onnx

# ✅ Model setup
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# ✅ Dummy input (1 image, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# ✅ Export path
onnx_path = "model/model_q4f16.onnx"

# ✅ Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)

print(f"✅ ViT model exported successfully to: {onnx_path}")
