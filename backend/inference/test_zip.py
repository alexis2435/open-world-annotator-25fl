import json
import shutil
import tempfile
import os
from unittest.mock import MagicMock
import boto3

# Step 1: Mock 掉所有 boto3.client("s3")
fake_s3 = MagicMock()
fake_s3.download_file = MagicMock()
boto3.client = MagicMock(return_value=fake_s3)

print("[LocalTest] 全局 boto3.client 已被 mock，不会访问 AWS")

from inference import model_fn, input_fn, predict_fn

# Step 2: 加载模型
model = model_fn("./")

# Step 3: 构造 input
body = {
    "prompt": "bear",
    "task": "seg",
    "s3_zip": "s3://whatever/local_test.zip"
}
raw = json.dumps(body).encode("utf-8")
parsed = input_fn(raw, "application/json")

# Step 4: 把 ZIP 放到 predict_fn 会找的 /tmp/input.zip
tmp_zip = "/tmp/input.zip"
shutil.copy("local_test.zip", tmp_zip)
print("ZIP 已放入:", tmp_zip)

# Step 5: 运行推理
pred = predict_fn(parsed, model)
print(json.dumps(pred, indent=2))
