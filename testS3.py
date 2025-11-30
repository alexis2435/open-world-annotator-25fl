import base64
import json
import os
from PIL import Image
import io
import numpy as np

import boto3

from backend.inference.inference import model_fn, input_fn, predict_fn, output_fn

from dotenv import load_dotenv
load_dotenv()


# 设置 bucket
bucekt = os.environ["S3_OUTPUT_BUCKET"]

# 测试的输入图像（模拟用户上传的图像）
INPUT_IMG_S3 = "s3://openworld-annotator/test_input/test2.jpg"

s3 = boto3.client("s3")
bucket, key = INPUT_IMG_S3.replace("s3://", "").split("/", 1)

# 模拟 SageMaker 从 S3 拉取图像进行处理
local_img = "tmp/input_test.jpg"  # 
s3.download_file(bucket, key, local_img)

with open(local_img, "rb") as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

print("Loading model...")
model = model_fn("./")

req = {
    "prompt": "Stop Sign",
    "task": "bbox",
    "image_base64": img_b64
}


raw = json.dumps(req).encode("utf-8")
parsed = input_fn(raw, "application/json")
pred = predict_fn(parsed, model)

resp_json, _ = output_fn(pred, "application/json")

print("\n===== INFERENCE RESULT =====")
print(resp_json)