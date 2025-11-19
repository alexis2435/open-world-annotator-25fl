# backend/inference/inference_core.py

import numpy as np
import torch
from backend.inference.model_loader import load_models

def run_inference_on_images(images, prompt, mode="bbox"):
    """
    images: List[PIL.Image]
    mode: "bbox" or "bbox+seg"

    Return: list of dict
    """

    processor, owl_model, sam_model, device = load_models()

    # prepare prompts
    prompt_list = [prompt]
    prompts = [prompt_list] * len(images)

    # preprocess
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor(
        [img.size[::-1] for img in images]
    ).to(device)

    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.3
    )

    # postprocess
    final_out = []
    for idx, (img, det) in enumerate(zip(images, results)):
        boxes = det["boxes"].cpu().numpy().tolist()
        scores = det["scores"].cpu().numpy().tolist()
        label_ids = det["labels"].cpu().numpy().tolist()
        label_names = [prompt_list[i] for i in label_ids]

        masks = None
        if mode == "bbox+seg" and len(boxes) > 0:
            img_np = np.array(img)
            all_masks = []
            for b in det["boxes"]:
                box = b.cpu().unsqueeze(0).numpy()
                m, _ = sam_model.predict(img_np, box, multimask_output=False)
                all_masks.append(m.squeeze())
            masks = np.array(all_masks).tolist()

        final_out.append({
            "boxes": boxes,
            "scores": scores,
            "labels": label_names,
            "masks": masks
        })

    return final_out
