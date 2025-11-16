import torch
import numpy as np
import time
import logging
from transformers import OwlViTProcessor
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_official import MobileSAMOfficial
from coco_loader import get_coco_dataloader
from flickr_loader import get_flickr_dataloader
from visualize import visualize_results
import box_processing

# ========== setup log ==========
logging.basicConfig(
    filename="inference.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ========== setup device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

# ========== initialize model ==========
owlvit_model = OwlvitOfficial().to(device)
processor = OwlViTProcessor.from_pretrained("models/owlvit-large-patch14")
owlvit_model.eval()

mobilesam_model = MobileSAMOfficial(checkpoint_path="models/mobile_sam.pt")

# ========== load dataset ==========
dataloader = get_coco_dataloader(
    root="datasets/coco/train2017",
    annotation="datasets/coco/annotations/instances_train2017.json",
    batch_size=1,
    shuffle=False
)

# dataloader = get_flickr_dataloader(
#     image_dir="datasets/flicker30k/flickr30k-images",
#     annotation_dir="datasets/flicker30k/Annotations",
#     sentence_dir="datasets/flicker30k/Sentences",
#     batch_size=1
# )


# ========== config ==========
max_images = 10
processed_images = 0

# ========== start timing ==========
if torch.cuda.is_available():
    torch.cuda.synchronize()
total_start_time = time.time()

# ========== inference main loop ==========
for batch_idx, batch in enumerate(dataloader):
    if processed_images >= max_images:
        break

    images, bboxes, prompts = batch
    print("images.size:", images.size())
    prompts = [[prompts[0][0]]]
    images = images.to(device=device, dtype=torch.float32)

    inputs = processor(text=prompts, images=images, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])

    logits = outputs[0]
    pred_boxes = outputs[1]
    print(logits.squeeze(0).squeeze(-1).sigmoid().min(), logits.squeeze(0).squeeze(-1).sigmoid().max())

    valid_boxes, valid_scores = box_processing.filter_boxes_by_score(logits, pred_boxes, show_plot=False)
    if valid_boxes.shape[0] == 0:
        logging.info("No high-confidence objects detected.")
        print("---------No high-confidence objects detected.")
        continue

    converted_boxes = box_processing.convert_boxes(valid_boxes, images.shape[-2:])
    filtered_boxes, filtered_scores = box_processing.apply_nms(converted_boxes, valid_scores, iou_threshold=0.3)
    final_boxes = box_processing.merge_high_iou_boxes(filtered_boxes, filtered_scores, iou_threshold=0.7)
    print(f"Detected {final_boxes.shape[0]} objects after NMS + Merge")
    log_msg = f"Detected {final_boxes.shape[0]} high-confidence objects after NMS + Merge"
    logging.info(log_msg)
    print(log_msg)

    for img_idx in range(images.shape[0]):
        image_np = images[img_idx].permute(1, 2, 0).cpu().numpy()

        all_masks = []
        all_scores = []

        final_boxes = final_boxes.cpu().numpy().astype(np.float32)
        for final_box in final_boxes:
            final_box = final_box.reshape(1, 4)
            masks, sam_scores = mobilesam_model.predict(image_np, final_box, multimask_output=False)
            all_masks.append(masks)
            all_scores.append(sam_scores)

        all_masks = np.array(all_masks).squeeze()
        if all_masks.ndim == 2:
            all_masks = np.expand_dims(all_masks, axis=0)

        visualize_results(image_np, final_boxes, all_masks, prompts[0], all_scores)

        log_info = f"Processed Image {processed_images + 1}/{max_images} | Category: {prompts[0]} | Objects: {len(final_boxes)}"
        logging.info(log_info)
        print(log_info)
        print("=" * 50)

        processed_images += 1

# ========== end timing ==========
if torch.cuda.is_available():
    torch.cuda.synchronize()
total_end_time = time.time()
total_time = total_end_time - total_start_time

final_msg = f"\n=== Total inference time for {processed_images} images: {total_time:.3f} seconds ==="
logging.info(final_msg)
print(final_msg)

if processed_images > 0:
    avg_time = total_time / processed_images
    avg_msg = f"Average time per image: {avg_time:.3f} seconds"
    logging.info(avg_msg)
    print(avg_msg)


# ========== integrate for grid_search ==========
def run_single_inference(confidence_threshold=0.3, top_k=5, merge_iou_threshold=0.7):
    processed_images = 0
    max_images = 10

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if processed_images >= max_images:
            break

        images, bboxes, prompts = batch
        prompts = [[prompts[0][0]]]
        images = images.to(device=device, dtype=torch.float32)

        inputs = processor(text=prompts, images=images, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
        logits = outputs[0]
        pred_boxes = outputs[1]

        valid_boxes, valid_scores = box_processing.filter_boxes_by_score(logits, pred_boxes, top_k=top_k,
                                                                         threshold=confidence_threshold)
        if valid_boxes.shape[0] == 0:
            print("No high-confidence objects detected.")
            continue

        converted_boxes = box_processing.convert_boxes(valid_boxes, images.shape[-2:])
        filtered_boxes, filtered_scores = box_processing.apply_nms(converted_boxes, valid_scores, iou_threshold=0.3)
        final_boxes = box_processing.merge_high_iou_boxes(filtered_boxes, filtered_scores,
                                                          iou_threshold=merge_iou_threshold)

        for img_idx in range(images.shape[0]):
            image_np = images[img_idx].permute(1, 2, 0).cpu().numpy()
            all_masks = []
            all_scores = []

            final_boxes = final_boxes.cpu().numpy().astype(np.float32)
            for final_box in final_boxes:
                final_box = final_box.reshape(1, 4)
                masks, sam_scores = mobilesam_model.predict(image_np, final_box, multimask_output=False)
                all_masks.append(masks)
                all_scores.append(sam_scores)

            all_masks = np.array(all_masks).squeeze()
            if all_masks.ndim == 2:
                all_masks = np.expand_dims(all_masks, axis=0)

            visualize_results(image_np, final_boxes, all_masks, prompts[0], all_scores)

            processed_images += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_end_time = time.time()
    print(f"Total time: {total_end_time - total_start_time:.2f}s")
