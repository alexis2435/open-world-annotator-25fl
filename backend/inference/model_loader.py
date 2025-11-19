# backend/inference/model_loader.py

"""
Model Loader Module
===================

Centralized, lazy-loaded initialization of all models required by the
object detection and segmentation pipeline (OWLv2 + MobileSAM).

This module ensures:

- Models are instantiated exactly once per process.
- OWLv2 and MobileSAM share the same device (GPU if available).
- The processor (tokenizer + image preprocessor) always stays on CPU.
- Thread-safe behavior in multi-worker environments
  (FastAPI, Gunicorn, SageMaker).
- Faster startup by deferring model loading until the first request.

Public API:

- :func:`load_processor` — Loads OWLv2 processor.
- :func:`load_owl_model` — Loads OWLv2 detection model.
- :func:`load_sam_model` — Loads MobileSAM.
- :func:`load_all` — Convenience loader for local debugging.

This module contains **no inference logic**; its only responsibility
is model initialization and caching.
"""
import os

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from backend.models.mobilesam_official import MobileSAMOfficial


# Global cached instances (lazy-loaded)
_processor = None
_owl_model = None
_sam_model = None
_device = None


# ----------------------------------------------------------------------
# Processor Loader (CPU-only)
# ----------------------------------------------------------------------
def load_processor() -> Owlv2Processor:
    """
    Load the OWLv2 processor (tokenizer + image preprocessor).

    The processor performs CPU-bound operations such as:
    - resizing
    - normalization
    - tokenization

    The processor **never** moves to GPU because it is not a torch module.

    Returns:
        Owlv2Processor: The shared processor instance.
    """
    global _processor
    if _processor is None:
        print("[ModelLoader] Initializing OWLv2 processor...")
        _processor = Owlv2Processor.from_pretrained(
            "google/owlv2-large-patch14-ensemble"
        )
    return _processor


# ----------------------------------------------------------------------
# OWLv2 Detection Model Loader
# ----------------------------------------------------------------------
def load_owl_model(device: str | torch.device | None = None):
    """
    Load the OWLv2 object detection model.

    The model is always placed on:
    - the user-specified device, if provided
    - otherwise GPU if available
    - otherwise CPU

    Args:
        device (str | torch.device | None, optional):
            Device where the model should be loaded.
            Examples: ``"cuda"``, ``"cpu"``, ``torch.device("cuda:0")``.

    Returns:
        tuple:
            A tuple ``(model, device)`` where:

            - **model** (*Owlv2ForObjectDetection*): Initialized detection model.
            - **device** (*torch.device*): The resolved device used for model loading.
    """
    global _owl_model, _device

    if _owl_model is None:
        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[ModelLoader] Loading OWLv2 model to {_device}...")
        _owl_model = (
            Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-large-patch14-ensemble"
            )
            .to(_device)
            .eval()
        )

    return _owl_model, _device


# ----------------------------------------------------------------------
# MobileSAM Segmentation Model Loader
# ----------------------------------------------------------------------
def load_sam_model(device: str | torch.device | None = None):
    """
    Load the MobileSAM segmentation model.

    If an OWLv2 model was already loaded, the same device is reused.
    Otherwise:

    - GPU is preferred if available
    - CPU is used as fallback

    Args:
        device (str | torch.device | None, optional):
            Preferred device for the model.

    Returns:
        tuple:
            A tuple ``(model, device)`` where:

            - **model** (*MobileSAMOfficial*): The MobileSAM instance.
            - **device** (*torch.device*): Actual device used.
    """
    global _sam_model, _device

    if _device is None:
        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if _sam_model is None:
        print(f"[ModelLoader] Loading MobileSAM to {_device}...")
        root = os.path.dirname(os.path.dirname(__file__))
        ckpt = os.path.join(root, "models", "mobile_sam.pt")

        _sam_model = MobileSAMOfficial(
            checkpoint_path=ckpt,
            device=_device
        )

    return _sam_model, _device


# ----------------------------------------------------------------------
# Load Everything (Local Development Utility)
# ----------------------------------------------------------------------
def load_all(device: str | torch.device | None = None):
    """
    Load the processor, OWLv2 model, and MobileSAM model in one call.

    This is intended primarily for **local testing**.
    In production (FastAPI, SageMaker), each component is loaded on-demand.

    Args:
        device (str | torch.device | None, optional):
            Device to load all models onto.

    Returns:
        tuple:
            ``(processor, owl_model, sam_model, device)`` — all initialized components.
    """
    processor = load_processor()
    owl, device = load_owl_model(device)
    sam, device = load_sam_model(device)
    return processor, owl, sam, device
