import base64
import json
from PIL import Image
import io
import numpy as np

from inference import model_fn, input_fn, predict_fn, output_fn

# --------------------------------------------------------------
# Step 1: load model
# --------------------------------------------------------------
print("Loading model...")
model = model_fn("./")

# --------------------------------------------------------------
# Step 2：test image
# --------------------------------------------------------------
test_img_path = "test.jpg"
with open(test_img_path, "rb") as f:
    img_bytes = f.read()

img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# --------------------------------------------------------------
# Step 3：require bbox
# --------------------------------------------------------------
req_single_bbox = {
    "prompt": "skier in red",
    "task": "bbox",
    "image_base64": img_b64
}

# input_fn
raw = json.dumps(req_single_bbox).encode("utf-8")
input_parsed = input_fn(raw, "application/json")

# predict_fn
pred = predict_fn(input_parsed, model)

# JSON
resp = output_fn(pred, "application/json")
print("\n\n===== Single Image BBOX Result =====")
print(resp[0])


# --------------------------------------------------------------
# Step 4： require segmentation
# --------------------------------------------------------------
req_single_seg = {
    "prompt": "skier in red",
    "task": "seg",
    "image_base64": img_b64
}

raw = json.dumps(req_single_seg).encode("utf-8")
input_parsed = input_fn(raw, "application/json")
pred = predict_fn(input_parsed, model)
resp = output_fn(pred, "application/json")

print("\n\n===== Single Image SEG Result =====")
print(resp[0])
