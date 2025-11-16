import os
import re
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FlickrDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, sentence_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sentence_dir = sentence_dir
        self.transform = transform
        self.image_ids = [fname[:-4] for fname in os.listdir(image_dir) if fname.endswith(".jpg")]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        xml_path = os.path.join(self.annotation_dir, image_id + ".xml")
        txt_path = os.path.join(self.sentence_dir, image_id + ".txt")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        entity_boxes = {}
        for obj in root.findall("object"):
            eid = obj.find("name").text
            if obj.find("nobndbox") is not None:
                continue
            bbox_elem = obj.find("bndbox")
            xmin = int(bbox_elem.find("xmin").text)
            ymin = int(bbox_elem.find("ymin").text)
            xmax = int(bbox_elem.find("xmax").text)
            ymax = int(bbox_elem.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin
            box = [xmin, ymin, w, h]
            entity_boxes.setdefault(eid, []).append(box)

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        entity_phrases = {}
        matches = re.findall(r'\[\/EN#(\d+)\/[^ ]+ ([^\]]+)\]', text)
        for eid, phrase in matches:
            entity_phrases.setdefault(eid, set()).add(phrase.strip())

        final_boxes = []
        final_phrases = []
        for eid in entity_boxes:
            if eid in entity_phrases:
                for box in entity_boxes[eid]:
                    for phrase in entity_phrases[eid]:
                        final_boxes.append(box)
                        final_phrases.append(phrase)

        bboxes = torch.tensor(final_boxes, dtype=torch.float32) if final_boxes else torch.zeros((0, 4),
                                                                                                dtype=torch.float32)
        return image, bboxes, final_phrases


transform = transforms.Compose([transforms.ToTensor()])


def collate_fn(batch):
    images, bboxes, category_names = zip(*batch)
    images = torch.stack(images)
    return images, list(bboxes), list(category_names)


def get_flickr_dataloader(image_dir, annotation_dir, sentence_dir, batch_size=8, shuffle=True):
    dataset = FlickrDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        sentence_dir=sentence_dir,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
