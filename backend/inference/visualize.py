import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# def visualize_results(image, boxes, masks, category, scores):
#     """
#     Visualize MobileSAM results:
#     - Draws bounding boxes on the original image
#     - Overlays segmentation masks with different colors
#
#     Args:
#     - image: numpy array of shape (H, W, C), should be in uint8 format
#     - boxes: numpy array of shape (N, 4), each row is [x_min, y_min, x_max, y_max] in pixel coordinates
#     - masks: numpy array of shape (N, H, W), binary masks (0 or 1)
#     - category: str, category name (used as label)
#     - scores: numpy array of shape (N,), confidence scores per box
#     """
#     image_vis = image.copy()
#     image_vis = (image_vis * 255).astype(np.uint8)
#
#     # Draw bounding boxes
#     for i, box in enumerate(boxes):
#         x_min, y_min, x_max, y_max = map(int, box)
#         cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
#         cv2.putText(image_vis, f"{category} {float(scores[i]):.2f}",
#                     (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
#
#     # Prepare colored mask overlay
#     color_mask = np.zeros_like(image_vis, dtype=np.uint8)
#
#     # Use a colormap for unique colors
#     cmap = plt.get_cmap("tab10")
#     num_masks = len(masks)
#
#     for i, mask in enumerate(masks):
#         color = np.array(cmap(i / num_masks)[:3]) * 255
#         color = color.astype(np.uint8)
#         color_mask[mask == 1] = color
#
#     # Blend masks with the original image
#     alpha = 0.5
#     image_overlay = cv2.addWeighted(image_vis, 1 - alpha, color_mask, alpha, 0)
#
#     # Display the result
#     plt.figure(figsize=(8, 8))
#     plt.imshow(image_overlay)
#     plt.axis("off")
#     plt.title(f"{category} - All Detections")
#     plt.show()


def visualize_results(image, boxes, masks, category, scores, save_path=None):
    """
    Visualize MobileSAM results:
    - Draws bounding boxes on the original image
    - Overlays segmentation masks with different colors
    - Optionally saves the visualization to disk

    Args:
    - image: numpy array of shape (H, W, C), dtype float or uint8
    - boxes: numpy array of shape (N, 4)
    - masks: numpy array of shape (N, H, W)
    - category: list of str or str
    - scores: numpy array of shape (N,)
    - save_path: optional str, path to save the image
    """
    image_vis = image.copy()
    if image_vis.max() <= 1.0:
        image_vis = (image_vis * 255).astype(np.uint8)
    else:
        image_vis = image_vis.astype(np.uint8)

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        label = category if isinstance(category, str) else category[0]
        score_text = f"{label} {float(scores[i]):.2f}" if scores is not None else f"{label}"
        cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(image_vis, score_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 255, 0), 1)

    # Prepare mask
    color_mask = np.zeros_like(image_vis, dtype=np.uint8)
    cmap = plt.get_cmap("tab10")
    num_masks = len(masks)

    for i, mask in enumerate(masks):
        color = np.array(cmap(i / max(num_masks, 1))[:3]) * 255
        color = color.astype(np.uint8)
        color_mask[mask == 1] = color

    # Blend
    alpha = 0.5
    image_overlay = cv2.addWeighted(image_vis, 1 - alpha, color_mask, alpha, 0)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_overlay, cv2.COLOR_RGB2BGR))
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(image_overlay)
        plt.axis("off")
        plt.title(f"{category} - All Detections")
        plt.show()
