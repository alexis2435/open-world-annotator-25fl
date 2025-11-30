# ----- Python Built-ins -----
import os
import io
import json
import base64
import uuid
import zipfile

# ----- Image / Array -----
from PIL import Image
import numpy as np

# ----- Torch / OWLv2 -----
import torch
from pycocotools import mask as mask_utils

# ----- AWS -----
import boto3

# ----- Your own modules (from backend.inference) -----
from backend.inference.model_loader import (
    load_processor,
    load_owl_model,
    load_sam_model
)
from backend.inference.visualize import visualize_results


# from backend.inference.inference_core import (
#     run_owlvit_detection,
#     run_mobilesam_segmentation,
#     visualize_and_save_local
# )

from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------
# 1.  SageMaker MODEL LOAD
# -----------------------------------------------------
from backend.inference.model_loader import load_all

# -----------------------------------------------------
def model_fn(model_dir):
    """
    SageMaker 在容器启动时调用，模型只会加载一次。
    下载权重 + 初始化模型
    """
    print("[SageMaker] model_fn: start loading models from S3...")

    processor, owl_model, sam_model, device = load_all()

    print("[SageMaker] model_fn: all models loaded successfully.")

    return {
        "processor": processor,     # CPU
        "owlvit": owl_model,        # GPU/CPU
        "mobilesam": sam_model,     # GPU/CPU
        "device": device            # torch.device("cuda") or cpu
    }




# -----------------------------------------------------
# 2.  input_fn：解析 HTTP 输入
# -----------------------------------------------------
def input_fn(request_body, request_content_type):
    """
    统一解析输入：
    支持 4 类：
      - 单图 + bbox
      - 单图 + seg
      - 批量 ZIP + bbox
      - 批量 ZIP + seg

    输入 JSON 必须包含：
        prompt: str
        task: "bbox" or "seg"
        image_base64: str?  (单图)
        s3_zip: str?        (批量)
    """

    # ----------------------------------------------------
    # Content-Type 必须是 application/json
    # ----------------------------------------------------
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # ----------------------------------------------------
    # JSON 解析
    # ----------------------------------------------------
    raw = request_body.decode("utf-8")
    data = json.loads(raw)

    # ---------------- Prompt ----------------
    if "prompt" not in data:
        raise ValueError("JSON must contain field: 'prompt'")
    prompt = data["prompt"]

    # ---------------- Task ----------------
    # 算法类型：bbox-only 或 segmentation（bbox+sam）
    if "task" not in data:
        raise ValueError("JSON must contain field: 'task' ('bbox' or 'seg')")
    task = data["task"]

    if task not in ("bbox", "seg"):
        raise ValueError("task must be 'bbox' or 'seg'")

    # ----------------------------------------------------
    # Case 1：单图（base64）
    # ----------------------------------------------------
    if "image_base64" in data:
        img_b64 = data["image_base64"]
        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return {
            "mode": "single",    # <-- 表示单张图片
            "task": task,        # <-- 算法类型：bbox / seg
            "prompt": prompt,
            "images": [image]
        }

    # ----------------------------------------------------
    # Case 2：批量（S3 ZIP）
    # ----------------------------------------------------
    if "s3_zip" in data:
        s3_uri = data["s3_zip"]
        bucket, key = parse_s3_uri(s3_uri)

        return {
            "mode": "batch",     # <-- 表示批量 ZIP 推理
            "task": task,
            "prompt": prompt,
            "s3_bucket": bucket,
            "s3_key": key
        }

    # ----------------------------------------------------
    # 两种模式都没有 → 用户输入有误
    # ----------------------------------------------------
    raise ValueError(
        "JSON must contain either 'image_base64' for single image "
        "or 's3_zip' for batch ZIP inference."
    )





# -----------------------------------------------------
# 3.  predict_fn：真正推理
# -----------------------------------------------------
import torch
import numpy as np

def run_owlvit_detection(images, prompt_list, model):
    """
    images: List[PIL.Image]
    prompt_list: List[str]，比如 ["cat"] 或 ["cat", "dog"]
    返回：每张图一个 dict，包含 boxes, scores, labels
    """
    device = model["device"]
    processor = model["processor"]
    owlvit_model = model["owlvit"]

    # prompt 形状要匹配 batch：每张图都用同一批 prompts
    prompts = [prompt_list] * len(images)

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owlvit_model(**inputs)

    target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

    # 把 tensor 转为 python list，方便 JSON 序列化
    all_results = []
    for i, result in enumerate(results):
        boxes = result["boxes"].cpu().numpy().tolist()
        scores = result["scores"].cpu().numpy().tolist()
        labels = result["labels"].cpu().numpy().tolist()

        all_results.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        })
    return all_results


def run_mobilesam_segmentation(image_np, boxes, model):
    """
    对一张 image 做 MobileSAM 分割
    image_np: H x W x 3 的 numpy 数组
    boxes: [[x1,y1,x2,y2], ...]
    返回：
      masks: List[np.ndarray]  (H x W, 0/1)
      scores: List[float]
    """
    mobilesam_model = model["mobilesam"]

    all_masks = []
    all_scores = []
    for box in boxes:
        box_arr = np.array(box, dtype=np.float32).reshape(1, 4)
        masks, sam_scores = mobilesam_model.predict(image_np, box_arr, multimask_output=False)
        all_masks.append(masks[0])         # 取第一张 mask
        all_scores.append(float(sam_scores[0]))

    return all_masks, all_scores


# version 1: 本地测试通过, 有问题可以回退参考
# def predict_fn(input_data, model):
#     """
#     input_data: 来自 input_fn
#     model: 来自 model_fn
#     """
#
#     mode = input_data["mode"]         # "single" / "batch"
#     task = input_data["task"]         # "bbox" / "seg"
#     prompt = input_data["prompt"]
#     prompt_list = [prompt]
#
#     processor = model["processor"]
#     owlvit = model["owlvit"]
#     mobilesam = model["mobilesam"]
#     device = model["device"]
#
#     prompt_list = [prompt]
#     results = []
#
#     # ============================================================
#     #  Case 1: 单图推理
#     # ============================================================
#     if mode == "single":
#         img = input_data["images"][0]
#         image_np = np.array(img)
#
#         # 1. OwlViT detection
#         det_list = run_owlvit_detection([img], prompt_list, model)
#         det = det_list[0]
#
#         boxes = det["boxes"]
#         scores = det["scores"]
#         labels = det["labels"]
#         matched_prompts = [prompt_list[idx] for idx in labels]
#
#         # bbox 结果
#         record = {
#             "filename": "input_image",
#             "boxes": boxes,
#             "scores": scores,
#             "labels": labels,
#         }
#
#         # 2. segmentation（可选）
#         if task == "seg":
#             masks, sam_scores = run_mobilesam_segmentation(image_np, boxes, model)
#
#             # 转 RLE
#             rle_masks = [encode_mask_rle(m) for m in masks]
#
#             record["sam_scores"] = sam_scores
#             record["masks_rle"] = rle_masks
#
#             # 3. 可视化图（bbox + mask）
#             vis_path = visualize_and_save_local(
#                 image_np, boxes, masks, labels
#             )
#
#         else:
#             # bbox-only 可视化
#             vis_path = visualize_and_save_local(
#                 image_np, boxes, None, labels
#             )
#
#         # 上传到 S3
#         vis_url = upload_visualization_to_s3(vis_path)
#         record["visualization_url"] = vis_url
#
#         results.append(record)
#
#         return {
#             "mode": mode,
#             "task": task,
#             "results": results
#         }
#
#     # ============================================================
#     #  Case 2: 批量 ZIP 推理
#     # ============================================================
#     elif mode == "batch":
#         bucket = input_data["s3_bucket"]
#         key = input_data["s3_key"]
#
#         # 下载 ZIP → /tmp
#         local_zip = "/tmp/input.zip"
#         s3_client = boto3.client("s3")
#         s3_client.download_file(bucket, key, local_zip)
#
#         extract_dir = "/tmp/images"
#         os.makedirs(extract_dir, exist_ok=True)
#         with zipfile.ZipFile(local_zip, "r") as zf:
#             zf.extractall(extract_dir)
#
#         # 遍历图片
#         image_paths = []
#         for root, _, files in os.walk(extract_dir):
#             for f in files:
#                 if f.lower().endswith((".jpg", ".jpeg", ".png")):
#                     image_paths.append(os.path.join(root, f))
#
#         # 一张一张推理（你的 OWLv2 batch_size=1）
#         for path in sorted(image_paths):
#             img = Image.open(path).convert("RGB")
#             image_np = np.array(img)
#
#             # ---- OwlViT detection ----
#             det_list = run_owlvit_detection([img], prompt_list, model)
#             det = det_list[0]
#
#             boxes = det["boxes"]
#             scores = det["scores"]
#             labels = det["labels"]
#
#             # bbox-only record
#             record = {
#                 "filename": os.path.relpath(path, extract_dir),
#                 "boxes": boxes,
#                 "scores": scores,
#                 "labels": labels,
#             }
#
#             # ---- segmentation ----
#             if task == "seg":
#                 masks, sam_scores = run_mobilesam_segmentation(image_np, boxes, model)
#                 rle_masks = [encode_mask_rle(m) for m in masks]
#
#                 record["sam_scores"] = sam_scores
#                 record["masks_rle"] = rle_masks
#
#                 # 可视化图（bbox + mask）
#                 vis_path = visualize_and_save_local(
#                     image_np, boxes, masks, labels
#                 )
#             else:
#                 # bbox-only 可视化
#                 vis_path = visualize_and_save_local(
#                     image_np, boxes, None, labels
#                 )
#
#             # ---- 上传到 S3 ----
#             vis_url = upload_visualization_to_s3(vis_path)
#             record["visualization_url"] = vis_url
#
#             results.append(record)
#
#         return {
#             "mode": mode,
#             "task": task,
#             "results": results
#         }
#
#     # ============================================================
#     # 否则输入无效
#     # ============================================================
#     else:
#         raise ValueError(f"Invalid mode: {mode}")


def predict_fn(input_data, model):
    mode = input_data["mode"]
    task = input_data["task"]      # "bbox" / "seg"
    prompt = input_data["prompt"]
    prompt_list = [prompt]

    processor = model["processor"]
    owlvit = model["owlvit"]
    mobilesam = model["mobilesam"]

    # ===========================================================
    # 1) 生成请求 ID & S3 路径
    # ===========================================================
    request_id, paths = build_request_paths()

    results = []

    # ===========================================================
    # ============   Case 1: 单图推理   =========================
    # ===========================================================
    if mode == "single":
        img = input_data["images"][0]
        filename = "input_image.jpg"
        image_np = np.array(img)

        # ---- 原图上传（uploaded_raw） ----
        local_raw = f"/tmp/{uuid.uuid4().hex}.jpg"
        img.save(local_raw)
        raw_url = upload_file_to_s3(local_raw, paths["uploaded_raw"])

        # ---- 检测 ----
        det_list = run_owlvit_detection([img], prompt_list, model)
        det = det_list[0]
        boxes, scores, labels = det["boxes"], det["scores"], det["labels"]

        # ---- 如果没有匹配，不需要放入 matched/raw，只返回 uploaded_raw ----
        if len(boxes) == 0:
            return {
                "request_id": request_id,
                "mode": mode,
                "task": task,
                "uploaded_raw": raw_url,
                "results": []
            }

        # ---- 匹配 → 上传到 matched/raw ----
        matched_raw_url = upload_file_to_s3(local_raw, paths["matched_raw"])

        # ---- bbox-only ----
        if task == "bbox":
            vis_local = save_visualization_local(image_np, boxes, None, labels)
            vis_url = upload_file_to_s3(vis_local, paths["bbox_images"])

            # JSON 数据
            json_url = upload_json_to_s3(det, paths["bbox_data"], "bbox_single")

            results.append({
                "filename": filename,
                "raw_url": raw_url,
                "matched_raw_url": matched_raw_url,
                "bbox_image_url": vis_url,
                "bbox_json_url": json_url
            })

        # ---- segmentation ----
        else:
            masks, sam_scores = run_mobilesam_segmentation(image_np, boxes, model)
            rle_masks = [encode_mask_rle(m) for m in masks]

            # 保存图
            vis_local = save_visualization_local(image_np, boxes, masks, labels)
            vis_url = upload_file_to_s3(vis_local, paths["seg_images"])

            # JSON
            data = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "sam_scores": sam_scores,
                "masks_rle": rle_masks
            }
            json_url = upload_json_to_s3(data, paths["seg_data"], "seg_single")

            results.append({
                "filename": filename,
                "raw_url": raw_url,
                "matched_raw_url": matched_raw_url,
                "seg_image_url": vis_url,
                "seg_json_url": json_url
            })

        return {
            "request_id": request_id,
            "mode": mode,
            "task": task,
            "results": results
        }

    # ===========================================================
    # ============   Case 2: 批量 ZIP 推理   ====================
    # ===========================================================
    elif mode == "batch":
        bucket = input_data["s3_bucket"]
        key = input_data["s3_key"]

        # ---- 下载 ZIP ----
        local_zip = "/tmp/input.zip"
        boto3.client("s3").download_file(bucket, key, local_zip)

        # ---- 解压 ----
        extract_dir = "/tmp/upload_raw"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(extract_dir)

        # ---- 找所有图片 ----
        image_paths = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_paths.append(os.path.join(root, f))

        # ---- 遍历推理 ----
        for path in sorted(image_paths):
            img = Image.open(path).convert("RGB")
            image_np = np.array(img)
            fname = os.path.basename(path)

            # ---- 上传 raw 到 uploaded_raw ----
            raw_url = upload_file_to_s3(path, paths["uploaded_raw"])

            det_list = run_owlvit_detection([img], prompt_list, model)
            det = det_list[0]
            boxes = det["boxes"]
            scores = det["scores"]
            labels = det["labels"]

            # 无匹配对象 → 不放入 matched
            if len(boxes) == 0:
                continue

            # ---- matched/raw ----
            matched_raw_url = upload_file_to_s3(path, paths["matched_raw"])

            # ------------ bbox-only ------------
            if task == "bbox":
                vis_local = save_visualization_local(image_np, boxes, None, labels)
                vis_url = upload_file_to_s3(vis_local, paths["bbox_images"])
                json_url = upload_json_to_s3(det, paths["bbox_data"], fname + "_bbox")

                results.append({
                    "filename": fname,
                    "raw_url": raw_url,
                    "matched_raw_url": matched_raw_url,
                    "bbox_image_url": vis_url,
                    "bbox_json_url": json_url
                })

            # ------------ seg ------------
            else:
                masks, sam_scores = run_mobilesam_segmentation(image_np, boxes, model)
                rle_masks = [encode_mask_rle(m) for m in masks]

                # 图片
                vis_local = save_visualization_local(image_np, boxes, masks, labels)
                vis_url = upload_file_to_s3(vis_local, paths["seg_images"])

                # JSON
                data = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "sam_scores": sam_scores,
                    "masks_rle": rle_masks
                }
                json_url = upload_json_to_s3(data, paths["seg_data"], fname + "_seg")

                results.append({
                    "filename": fname,
                    "raw_url": raw_url,
                    "matched_raw_url": matched_raw_url,
                    "seg_image_url": vis_url,
                    "seg_json_url": json_url
                })

        return {
            "request_id": request_id,
            "mode": mode,
            "task": task,
            "results": results
        }

    else:
        raise ValueError(f"Invalid mode: {mode}")



# -----------------------------------------------------
# 4.  output_fn：返回 JSON
# -----------------------------------------------------
def output_fn(prediction, accept):
    """
    SageMaker 返回给 HTTP 客户端的最终响应。
    prediction 是字典，必须转换为 JSON 字符串。
    """
    if accept in ("application/json", "application/json; charset=utf-8"):
        return json.dumps(prediction), accept

    raise ValueError(f"Unsupported accept type: {accept}")




# -----------------------------------------------------
# 工具函数
# -----------------------------------------------------
def parse_s3_uri(uri: str):
    uri = uri.strip()

    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}. Must start with s3://")

    try:
        no_scheme = uri[5:]  # 去掉 s3://
        bucket, key = no_scheme.split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid S3 URI structure: {uri}")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")

    return bucket, key


def encode_mask_rle(mask: np.ndarray):
    """
    mask: H x W, uint8 or bool, values 0/1
    返回 COCO 格式 RLE，可 JSON 序列化
    """
    # 必须确保是 Fortran contiguous，否则 encode 会报错
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

    # pycocotools 生成的 rle['counts'] 是 bytes，需要 decode 才能 JSON dump
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def save_visualization_local(image_np, boxes, masks, labels):
    is_local = os.environ.get("S3_OUTPUT_BUCKET") is None

    if is_local:
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    else:
        out_dir = "/tmp"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.jpg")

    matched_prompts = [str(l) for l in labels]
    visualize_results(
        image_np,
        np.array(boxes),
        masks if masks is not None else np.zeros((len(boxes), *image_np.shape[:2])),
        matched_prompts,
        None,
        save_path=out_path
    )
    return out_path


def visualize_and_save_local(image_np, boxes, masks, labels):
    """
    在本地开发环境下，把图保存到项目：backend/inference/outputs/
    在 SageMaker 环境下仍然保存到 /tmp（由 upload_visualization_to_s3 处理）
    """
    # 本地开发：S3 bucket 不存在
    is_local = os.environ.get("S3_OUTPUT_BUCKET") is None

    if is_local:
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    else:
        out_dir = "/tmp"

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.jpg")

    # 你的 visualize_results 需要 matched_prompts、scores
    matched_prompts = [str(l) for l in labels]
    scores = None

    visualize_results(
        image_np,
        np.array(boxes),
        masks if masks is not None else np.zeros((len(boxes), *image_np.shape[:2])),
        matched_prompts,
        scores,
        save_path=out_path
    )

    return out_path

# 希望的s3应该是:
# result/
#     REQ_20251117_xxx/
#         uploaded_raw/
#         matched/
#             raw/
#             bbox/
#                 images/
#                 data/
#             seg/
#                 images /
#                 data /



import boto3
import uuid
import os

s3_client = boto3.client("s3")

def upload_visualization_to_s3(local_path):
    """
    上传 /tmp 下生成的可视化图像到 S3。
    自动生成唯一 key，并返回 s3://bucket/key
    """
    bucket = os.environ.get("S3_OUTPUT_BUCKET")

    if not bucket:
        print(f"[LocalTest] Skip S3 upload. File saved at: {local_path}")
        return f"file://{local_path}"

    # 文件名
    key = f"visualizations/{uuid.uuid4().hex}.jpg"

    s3_client.upload_file(local_path, bucket, key)

    return f"s3://{bucket}/{key}"


def build_request_paths():
    request_id = f"REQ_{uuid.uuid4().hex[:8]}"
    base_prefix = f"results/{request_id}/"

    paths = {
        # 用户上传 ZIP 解压后的所有图片
        "uploaded_raw": base_prefix + "uploaded_raw/",

        # 匹配提示词的图片（raw、bbox、seg）
        "matched_raw": base_prefix + "matched/raw/",

        "bbox_images": base_prefix + "matched/bbox/images/",
        "bbox_data": base_prefix + "matched/bbox/data/",

        "seg_images": base_prefix + "matched/seg/images/",
        "seg_data": base_prefix + "matched/seg/data/",
    }

    return request_id, paths


s3_client = boto3.client("s3")

def upload_file_to_s3(local_path, s3_prefix):
    bucket = os.environ.get("S3_OUTPUT_BUCKET")
    if not bucket:
        print(f"[LocalTest] Skip upload → {local_path}")
        return f"file://{local_path}"

    filename = os.path.basename(local_path)
    key = s3_prefix + filename
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def upload_json_to_s3(data, s3_prefix, filename):
    """上传 JSON 标注结果"""
    bucket = os.environ.get("S3_OUTPUT_BUCKET")

    local_json = f"/tmp/{uuid.uuid4().hex}.json"
    with open(local_json, "w") as f:
        json.dump(data, f)

    key = s3_prefix + f"{filename}.json"

    if not bucket:
        print(f"[LocalTest] Skip S3 upload → JSON {local_json}")
        return f"file://{local_json}"

    s3_client.upload_file(local_json, bucket, key)
    return f"s3://{bucket}/{key}"



if __name__ == "__main__":
    print("[SageMaker] Launching model server...")

