import torch
from mobile_sam import sam_model_registry, SamPredictor


class MobileSAMOfficial:
    def __init__(self, checkpoint_path, device='cuda'):
        # Load MobileSAM model with pre-trained checkpoint
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        self.model.to(self.device)

        # Initialize predictor
        self.predictor = SamPredictor(self.model)

    def predict(self, image, boxes, multimask_output=True):
        """
        Args:
            image: numpy array [H, W, 3] - input image
            boxes: numpy array [num_boxes, 4] - bounding box (x_min, y_min, x_max, y_max)
            multimask_output: bool - whether to return multiple mask predictions

        Returns:
            masks: numpy array [num_boxes, num_masks_per_box, H, W] - predicted segmentation masks
            scores: numpy array [num_boxes, num_masks_per_box] - confidence scores for each mask
        """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            box=boxes,
            multimask_output=multimask_output
        )
        return masks, scores
