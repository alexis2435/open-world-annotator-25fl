import os
import boto3
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from backend.inference.mobilesam_official import MobileSAMOfficial

# -------- Global cache --------
_processor = None
_owl_model = None
_sam_model = None
_device = None

# -------- S3 config --------
S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "openworld-annotator")
S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "model")
LOCAL_MODEL_DIR = "/tmp/models"

s3 = boto3.client("s3")


# -------------------------------------------------------
# Utils
# -------------------------------------------------------
def download_dir_from_s3(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if not os.path.exists(local_path):
                print(f"[ModelLoader] Downloading {key}")
                s3.download_file(bucket, key, local_path)


def ensure_models_downloaded():
    print("[ModelLoader] Syncing models from S3...")
    download_dir_from_s3(
        S3_BUCKET,
        S3_PREFIX,
        LOCAL_MODEL_DIR
    )


# -------------------------------------------------------
# Processor (local cache)
# -------------------------------------------------------
def load_processor() -> Owlv2Processor:
    global _processor

    if _processor is None:
        print("[ModelLoader] Loading processor from local dir ...")
        processor_path = os.path.join(
            LOCAL_MODEL_DIR, "owlv2-large-patch14-ensemble"
        )
        _processor = Owlv2Processor.from_pretrained(processor_path)

    return _processor


# -------------------------------------------------------
# OWLv2 Loader (local model)
# -------------------------------------------------------
def load_owl_model(device=None):
    global _owl_model, _device

    if _owl_model is None:
        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[ModelLoader] Loading OWLv2 on {_device} ...")

        model_path = os.path.join(
            LOCAL_MODEL_DIR,
            "owlv2-large-patch14-ensemble"
        )

        _owl_model = (
            Owlv2ForObjectDetection
            .from_pretrained(model_path)
            .to(_device)
            .eval()
        )

    return _owl_model, _device


# -------------------------------------------------------
# MobileSAM Loader
# -------------------------------------------------------
def load_sam_model(device=None):
    global _sam_model, _device

    if _device is None:
        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if _sam_model is None:
        print(f"[ModelLoader] Loading MobileSAM on {_device}")

        ckpt = os.path.join(LOCAL_MODEL_DIR, "mobile_sam.pt")

        _sam_model = MobileSAMOfficial(
            checkpoint_path=ckpt,
            device=_device
        )

    return _sam_model, _device


# -------------------------------------------------------
# Entry point for SageMaker
# -------------------------------------------------------
def load_all(device=None):
    ensure_models_downloaded()

    processor = load_processor()
    owl, device = load_owl_model(device)
    sam, device = load_sam_model(device)

    return processor, owl, sam, device
