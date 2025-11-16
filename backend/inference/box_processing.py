import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt


def compute_elbow_threshold_from_logits(logits, apply_sigmoid=True):
    """
    Compute elbow threshold from raw logits. Optionally apply sigmoid to output a probability.
    """
    if logits.numel() < 2:
        print("Too few logits to compute elbow.")
        return 0.0

    print("logits.shape:", logits.shape)
    sorted_scores, _ = torch.sort(logits, descending=True)
    logits = logits.view(-1)
    sorted_logits, _ = torch.sort(logits, descending=True)
    diffs = sorted_logits[:-1] - sorted_logits[1:]

    if diffs.numel() == 0:
        print("No score differences to compute elbow.")
        return 0.0

    elbow_idx = torch.argmax(diffs)
    elbow_logit = sorted_logits[elbow_idx].item()

    print("elbow_logit:", elbow_logit)
    print("elbow_logit_sigmoid:", torch.sigmoid(torch.tensor(elbow_logit)).item())
    return torch.sigmoid(torch.tensor(elbow_logit)).item() if apply_sigmoid else elbow_logit


def filter_boxes_by_score(logits, pred_boxes, min_threshold=0.09, show_plot=False):
    """
    Filter out low-confidence boxes with a dynamic threshold.

    Args:
        logits (Tensor): Detection logits, shape [num_boxes, 1]
        pred_boxes (Tensor): Predicted boxes, shape [num_boxes, 4]
        min_threshold (float): Minimum threshold to avoid filtering out all boxes

    Returns:
        valid_boxes (Tensor): Filtered boxes
        valid_scores (Tensor): Filtered confidence scores
    """
    scores = logits.squeeze(-1).sigmoid()
    max_score = scores.max().item()
    print("max_score:", max_score)

    if max_score > 0.05:
        threshold = max(min_threshold, scores.mean() - scores.std())
    else:
        print("Scores too low, applying logits normalization...")
        scores = (logits - logits.mean()).squeeze(-1).sigmoid()
        threshold = max(min_threshold, scores.mean() - scores.std())

    print(f"Using threshold: {threshold:.4f}")

    if show_plot:
        flat_scores = scores.cpu().flatten().numpy()

        quantile_thresh = scores.quantile(0.85).item()
        elbow_thresh = compute_elbow_threshold_from_logits(logits, apply_sigmoid=True)

        plt.figure(figsize=(6, 4))
        plt.hist(flat_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        plt.axvline(threshold, color='red', linestyle='--', label=f"Mean - Std = {threshold:.4f}")
        plt.axvline(quantile_thresh, color='green', linestyle='-.', label=f"85% Quantile = {quantile_thresh:.4f}")
        plt.axvline(elbow_thresh, color='orange', linestyle=':', label=f"Elbow = {elbow_thresh:.4f}")

        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title("Distribution of OWL-ViT Scores")
        plt.legend()
        plt.tight_layout()
        plt.show()

    valid_indices = scores > threshold
    valid_boxes = pred_boxes[valid_indices]
    valid_scores = scores[valid_indices]
    print("============================================")

    return valid_boxes, valid_scores


def convert_boxes(pred_boxes, image_shape):
    """Convert normalized coordinates to pixel coordinates"""
    H, W = image_shape
    x_center, y_center, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    x_min = (x_center - w / 2) * W
    y_min = (y_center - h / 2) * H
    x_max = (x_center + w / 2) * W
    y_max = (y_center + h / 2) * H

    return torch.stack([x_min.clamp(0, W), y_min.clamp(0, H), x_max.clamp(0, W), y_max.clamp(0, H)], dim=1)


def apply_nms(boxes, scores, iou_threshold=0.3):
    """Apply Non-Maximum Suppression (NMS)"""
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    return boxes[keep_indices], scores[keep_indices]


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def merge_high_iou_boxes(boxes, scores, iou_threshold=0.7):
    """
    Merge boxes with IoU greater than the threshold using weighted averaging.

    Args:
        boxes: Tensor of shape (N, 4), each box is [x1, y1, x2, y2]
        scores: Tensor of shape (N,), confidence score for each box
        iou_threshold: IoU threshold above which boxes are merged

    Returns:
        Tensor of shape (M, 4), merged boxes
    """
    print(f"Total boxes before merge: {len(boxes)}")
    assert boxes.shape[0] == scores.shape[0], "Boxes and scores must match in length"
    if boxes.shape[0] == 0:
        return boxes  # Return empty tensor if no boxes

    device = boxes.device
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        # Initialize a group of overlapping boxes
        overlapping = [(boxes[i], scores[i])]
        used.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                overlapping.append((boxes[j], scores[j]))
                used.add(j)

        # === Merge the group ===
        group_scores = torch.tensor([score for _, score in overlapping], dtype=torch.float32, device=device)
        group_boxes = torch.stack([box for box, _ in overlapping])

        group_weights = group_scores / group_scores.sum()
        merged_box = (group_weights[:, None] * group_boxes).sum(dim=0)  # Weighted merge

        merged.append(merged_box)

    return torch.stack(merged, dim=0)
