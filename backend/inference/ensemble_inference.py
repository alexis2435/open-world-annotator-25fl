import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor

from config import cancel_flag
from backend.training.dataloaders.image_folder_dataset import ImageFolderDataset, collate_fn
from backend.models.owlvit_official import OwlvitOfficial
from backend.models.mobilesam_official import MobileSAMOfficial
import backend.inference.box_processing
from backend.inference.visualize import visualize_results
import webview

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from backend.inference.expand_prompt import expand_prompt

api_key = "the api_key"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to("cuda").eval()
# owlvit_model = owlvit_model.half()
mobilesam_model = MobileSAMOfficial(checkpoint_path="backend/models/mobile_sam.pt")


def run_bounding_box_only(folder_path, prompt):
    output_folder = os.path.join(folder_path, "output_bbox")
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset & dataloader
    dataset = ImageFolderDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # prompt_list = expand_prompt(prompt, api_key)
    prompt_list = [prompt]
    print(prompt_list)

    for images, filenames, original_sizes in dataloader:
        if cancel_flag["stop"]:
            print("🛑 Stopped by user.")
            break

        prompts = [prompt_list] * len(images)  # [prompt_list] since OwlViT expects batch of list-of-prompts

        # Use OwlViTProcessor to handle preprocessing (normalize, resize, tensor conversion)
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        print(f"\n[Batch Info] Num images: {len(images)}, prompt per image: {len(prompts[0])}")
        print(f"Input tensor shapes:")
        for k, v in inputs.items():
            print(f"  {k}: {v.shape}")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owlvit_model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

        for i in range(len(images)):
            print(f"\nProcessing [{i + 1}/{len(images)}] {filenames[i]}")
            result = results[i]
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]

            if len(boxes) == 0:
                print(f"[{filenames[i]}] No objects detected with confidence > 0.1")
                webview.windows[0].evaluate_js("updateProgressBar()")
                continue

            print(f"→ {len(boxes)} boxes detected")
            print("Matched Prompts:")
            for idx in labels[:5]:
                print(f" - {prompt_list[int(idx)]}")

            matched_prompts = [prompt_list[int(idx)] for idx in labels]
            image_np = np.array(images[i])
            all_masks = np.zeros((len(boxes), *image_np.shape[:2]))

            save_path = os.path.join(output_folder, f"{filenames[i].rsplit('.', 1)[0]}_bbox.jpg")
            visualize_results(image_np, boxes, all_masks, matched_prompts, None, save_path=save_path)

            # print("save_path:", save_path)
            # # filename_display = os.path.basename(save_path).replace("\\", "/").replace("'", "\\'")
            # print("filename:", filenames[i])
            filename = f"{filenames[i].rsplit('.', 1)[0]}_bbox.jpg"
            webview.windows[0].evaluate_js(f"addToProcessedList('{filename}')")
            webview.windows[0].evaluate_js("updateProgressBar()")

    print("Finished processing.")
    return output_folder


def run_box_and_segmentation(folder_path, prompt):
    output_folder = os.path.join(folder_path, "output_seg")
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset & dataloader
    dataset = ImageFolderDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # prompt_list = expand_prompt(prompt, api_key)
    prompt_list = [prompt]
    print(prompt_list)

    for images, filenames, original_sizes in dataloader:
        if cancel_flag["stop"]:
            print("🛑 Stopped by user.")
            break

        prompts = [prompt_list] * len(images)
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owlvit_model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

        for i in range(len(images)):
            print(f"\nProcessing [{i + 1}/{len(images)}] {filenames[i]}")
            result = results[i]
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]

            if len(boxes) == 0:
                print(f"[{filenames[i]}] No objects detected with confidence > 0.1")
                webview.windows[0].evaluate_js("updateProgressBar()")
                continue

            print(f"→ {len(boxes)} boxes detected")
            matched_prompts = [prompt_list[int(idx)] for idx in labels]
            image_np = np.array(images[i])

            all_masks = []
            all_scores = []
            for box in boxes:
                box_tensor = box.cpu().unsqueeze(0)
                masks, sam_scores = mobilesam_model.predict(image_np, box_tensor.numpy(), multimask_output=False)
                all_masks.append(masks)
                all_scores.append(sam_scores)

            all_masks = np.array(all_masks).squeeze()
            if all_masks.ndim == 2:
                all_masks = np.expand_dims(all_masks, axis=0)

            save_path = os.path.join(output_folder, f"{filenames[i].rsplit('.', 1)[0]}_seg.jpg")
            visualize_results(image_np, boxes, all_masks, matched_prompts, all_scores, save_path=save_path)

            filename = f"{filenames[i].rsplit('.', 1)[0]}_seg.jpg"
            webview.windows[0].evaluate_js(f"addToProcessedList('{filename}')")
            webview.windows[0].evaluate_js("updateProgressBar()")

    print("Finished segmentation processing.")
    return output_folder
