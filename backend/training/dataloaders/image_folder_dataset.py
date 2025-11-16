from torch.utils.data import Dataset
from PIL import Image
import os


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing image files
        """
        self.folder_path = folder_path
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            image (PIL.Image.Image): The image in RGB mode, not transformed
            filename (str): The filename of the image, e.g. "00123.jpg"
            original_size (tuple[int, int]): The original size of the image as (width, height)
        """
        filename = self.image_files[idx]
        image_path = os.path.join(self.folder_path, filename)
        image = Image.open(image_path).convert("RGB")  # PIL.Image.Image
        original_size = image.size  # (width, height)

        return image, filename, original_size


def collate_fn(batch):
    """
    Args:
        batch (list): A list of items, where each item is a tuple of
                      (PIL.Image.Image, str, (int, int))

    Returns:
        images (list[PIL.Image.Image]): List of images in RGB mode, untransformed
        filenames (list[str]): List of image filenames
        original_sizes (list[tuple[int, int]]): List of original sizes (width, height) for each image
    """
    images, filenames, original_sizes = zip(*batch)
    # lists for owlvit processor
    return list(images), list(filenames), list(original_sizes)
