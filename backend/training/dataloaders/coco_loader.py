import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from PIL import Image


class COCODataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        """
        Args:
            root (str): Path to COCO images directory.
            annotation (str): Path to COCO annotations JSON file.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.coco = CocoDetection(root=root, annFile=annotation)
        self.transform = transform

        # Load category mappings
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.coco.cats.values()}

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, annotations = self.coco[idx]
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract bounding boxes and category labels
        bboxes = torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32)  # Convert bbox to Tensor
        category_ids = [ann["category_id"] for ann in annotations]
        category_names = [self.categories[cat_id] for cat_id in category_ids]

        return image, bboxes, category_names


# Define transformations
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize images to fit ViT input
    transforms.ToTensor(),
])


# Custom collate function to handle batch processing
def collate_fn(batch):
    images, bboxes, category_names = zip(*batch)
    images = torch.stack(images)  # Convert list of Tensors to batch Tensor [B, C, H, W]
    return images, list(bboxes), list(category_names)


# Initialize dataset and dataloader
def get_coco_dataloader(root, annotation, batch_size=8, shuffle=True):
    dataset = COCODataset(root=root, annotation=annotation, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
