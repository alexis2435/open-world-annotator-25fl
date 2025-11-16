import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from mobile_sam.build_sam import build_sam_vit_t
from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    return transform(Image.open(image_path).convert("RGB"))


class PromptGenerator:
    def __call__(self, image, mask):
        H, W = mask.shape
        coords = torch.nonzero(mask > 0.5, as_tuple=False)
        if len(coords) > 0:
            idx = torch.randint(0, coords.size(0), (1,))
            point = coords[idx].squeeze().flip(0).unsqueeze(0).float()  # [1, 2]
            label = torch.tensor([1])  # [1]
        else:
            point = torch.tensor([[W // 2, H // 2]])
            label = torch.tensor([0])
        return {
            "point_coords": point.unsqueeze(0),  # [1, 1, 2]
            "point_labels": label.unsqueeze(0),  # [1, 1]
            "boxes": None,
            "mask_inputs": None,
        }


class SAMTrainDataset(Dataset):
    def __init__(self, image_dir, prompt_generator, mask_generator):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.prompt_generator = prompt_generator
        self.mask_generator = mask_generator

    def __getitem__(self, idx):
        image = preprocess_image(self.image_paths[idx])  # [3, 1024, 1024]
        np_image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        masks = self.mask_generator.generate(np_image)
        if len(masks) == 0:
            gt_mask = torch.zeros((1, 256, 256))
        else:
            gt_mask = torch.from_numpy(masks[0]["segmentation"]).unsqueeze(0).float()
            gt_mask = F.interpolate(gt_mask.unsqueeze(0), size=(256, 256), mode="nearest").squeeze(0)

        prompt = self.prompt_generator(image, gt_mask[0])
        return image, prompt, gt_mask

    def __len__(self):
        return len(self.image_paths)


def custom_collate(batch):
    images, prompts, masks = zip(*batch)
    return torch.stack(images), list(prompts), torch.stack(masks)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_sam_vit_t(checkpoint=None).to(device)
model.train()
prompt_generator = PromptGenerator()
mask_generator = SamAutomaticMaskGenerator(model)
dataset = SAMTrainDataset("data/images", prompt_generator, mask_generator)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
    total_loss = 0
    for image, prompt, gt_mask in dataloader:
        image = image.to(device)
        gt_mask = gt_mask.to(device)

        image_embedding = model.image_encoder(image)

        losses = []
        for i in range(image.shape[0]):
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(prompt[i]["point_coords"].to(device), prompt[i]["point_labels"].to(device)),
                boxes=prompt[i].get("boxes"),
                masks=prompt[i].get("mask_inputs"),
            )

            pred_mask, _ = model.mask_decoder(
                image_embedding[i:i + 1],
                model.prompt_encoder.get_dense_pe()[i:i + 1],
                sparse_embeddings,
                dense_embeddings,
                False
            )

            loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask[i:i + 1])
            losses.append(loss)

        total_batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
