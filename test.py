import onnxruntime as ort
from PIL import Image
import numpy as np

# ✅ Paths
MODEL_PATH = r"model\model_vit.onnx"
IMAGE_PATH = r"C:\Users\rjhon\OneDrive\Pictures\R.jpg"  # replace with your valid image path

# ✅ Load ONNX model
session = ort.InferenceSession(MODEL_PATH)

# ✅ Load and preprocess image using PIL
try:
    img = Image.open(IMAGE_PATH).convert("RGB")  # ensures 3 channels
except Exception as e:
    raise ValueError(f"Cannot open image: {IMAGE_PATH}. Error: {e}")

img = img.resize((224, 224))
img = np.array(img).astype(np.float32) / 255.0
input_tensor = np.expand_dims(img.transpose(2, 0, 1), axis=0)  # shape: (1,3,224,224)

# ✅ Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

# ✅ Safely handle outputs
output_array = outputs[0]
print("Raw model output shape:", output_array.shape)

# Top 5 predicted indices and their values
top_indices = output_array[0].argsort()[-5:][::-1]
print("Top 5 predicted indices and scores:")
for i in top_indices:
    print(f"Index: {i}, Score: {output_array[0][i]}")
