# Automatic Annotation System

A zero-shot visual annotation pipeline combining OWL-ViT and MobileSAM, with prompt engineering, bounding box post-processing, and optional GUI frontend for fast and flexible dataset labeling.

---

## Features

-  **Zero-shot object detection** using OWL-ViT
-  **Segmentation masks** via MobileSAM
-  **Prompt expansion** using LLM-guided augmentation
-  **Box filtering and merging** (confidence thresholding, NMS, IoU merging)
-  **Visualization** of boxes and masks with score overlays
-  **Evaluation and metrics**
-  **Web-based frontend** (HTML/JS/CSS)
---

## System Pipeline

![Pipeline Overview](pipeline_diagram.png)

---

## User Interface

![GUI Overview](GUI_Showcase.png)

---

## Project Structure

```
automatic-annotation-system/
│
├── datasets/
│   ├── coco/
│   └── flicker30k/
│
├── frontend/                     # Optional GUI frontend
│   ├── Ademo/                    # Functionnally Demo
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/                       # Model weights and wrappers
│   ├── owlvit-base-patch32/
│   ├── owlvit-large-patch14/
│   ├── owlv2-large-patch14-ensemble/
│   ├── mobile_sam.pt             # MobileSAM weights
│   ├── mobilesam_official.py
│   ├── owlvit.py
│   └── owlvit_official.py
│
├── app.py                        # Launches GUI and API backend
├── inference.py                  # OWL-ViT + SAM pipeline showcase
├── train.py                      # For optional fine-tuning
├── config.py                     # Paths and config values
├── box_processing.py             # NMS, merge, filtering
├── ensemble_inference.py         # Inference used in GUI
├── flickr_loader.py
├── flickr30k_entities_utils.py
├── FlickrTrainingDataset.py
├── image_folder_dataset.py
├── expand_prompt.py              # LLM-based prompt generator
├── evaluation_metrics.py
├── grid_search.py
├── coco_loader.py
├── visualize.py
├── environment.yml               # Conda environment config
├── inference.log                 # Logs from recent run
├── pipeline_diagram.png          # System pipeline diagram
└── README.md
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Gin-Xia/automatic-annotation-system.git
cd automatic-annotation-system
```

### 2. Set up the environment

```bash
conda env create -f environment.yml
conda activate automatic-annotation-system
```


### 3. Launch frontend

```bash
python app.py
```

---

## Outputs

- Bounding boxes with scores
- Instance masks from MobileSAM
- Expanded prompts for diverse detection
- Interactive visual result display

---



## TODOs

- [ ] Add Dockerfile for reproducibility
- [ ] LLM-in-the-loop prompt refinement
- [ ] Custom training on pseudo-labeled dataset

---

## License

MIT License

---

## Contact

Questions or suggestions? Open an issue please.
