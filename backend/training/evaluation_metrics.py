import numpy as np


# Bounding Box Metrics

def compute_iou(box1, box2):
    """Compute IoU between two boxes: [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return interArea / (box1Area + box2Area - interArea + 1e-6)


def compute_map(ious, threshold=0.5):
    """Compute mAP from a list of IoUs (one sample per phrase)"""
    true_positives = [1 if iou >= threshold else 0 for iou in ious]
    return np.mean(true_positives)


def recall_at_k(ious, k=5, threshold=0.5):
    """Evaluate Recall@K for list of K predicted IoUs"""
    return np.mean([np.any(np.array(sample)[:k] >= threshold) for sample in ious])


# Segmentation Metrics

def compute_mask_iou(mask_pred, mask_gt):
    """Binary masks: 2D numpy arrays with 0 or 1"""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / (union + 1e-6)


def compute_dice(mask_pred, mask_gt):
    """Binary masks: 2D numpy arrays"""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    return 2 * intersection / (mask_pred.sum() + mask_gt.sum() + 1e-6)


def compute_mean_metric(metric_fn, preds, gts):
    """Compute mean of a metric over a list of mask pairs"""
    return np.mean([metric_fn(p, g) for p, g in zip(preds, gts)])
