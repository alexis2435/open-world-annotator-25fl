import os
import re
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import OwlViTProcessor


class FlickrTrainingDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, sentence_dir, processor, vocab, num_queries=10, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sentence_dir = sentence_dir
        self.processor = processor
        self.vocab = vocab  # phrase -> label_id
        self.num_queries = num_queries
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
        width, height = image.size

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
            box = [xmin / width, ymin / height, w / width, h / height]  # ⚡归一化
            entity_boxes.setdefault(eid, []).append(box)

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        entity_phrases = {}
        matches = re.findall(r'\[\/EN#(\d+)\/[^ ]+ ([^\]]+)\]', text)
        for eid, phrase in matches:
            entity_phrases.setdefault(eid, set()).add(phrase.strip())

        matched_labels = []
        matched_boxes = []
        for eid in entity_boxes:
            if eid in entity_phrases:
                for phrase in entity_phrases[eid]:
                    if phrase in self.vocab:
                        label_id = self.vocab[phrase]
                        for box in entity_boxes[eid]:
                            matched_labels.append(label_id)
                            matched_boxes.append(box)

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)  # 默认空query
        boxes = torch.zeros((self.num_queries, 4), dtype=torch.float32)

        num_valid = min(len(matched_labels), self.num_queries)
        if num_valid > 0:
            labels[:num_valid] = torch.tensor(matched_labels[:num_valid], dtype=torch.long)
            boxes[:num_valid] = torch.tensor(matched_boxes[:num_valid], dtype=torch.float32)

        prompt_text = " ".join(list(self.vocab.keys()))
        encoding = self.processor(text=prompt_text,
                                  images=image,
                                  return_tensors="pt",
                                  padding="max_length",
                                  truncation=True)

        pixel_values = encoding['pixel_values'].squeeze(0)  # [3, H, W]
        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [seq_len]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "boxes": boxes
        }


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    boxes = torch.stack([item['boxes'] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "boxes": boxes
    }


def get_flickr_training_loader(image_dir, annotation_dir, sentence_dir, vocab, processor, batch_size=8, shuffle=True):
    dataset = FlickrTrainingDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        sentence_dir=sentence_dir,
        processor=processor,
        vocab=vocab,
        num_queries=10,
        transform=transforms.ToTensor()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
