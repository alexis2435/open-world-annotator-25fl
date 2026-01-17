# Open World Annotator

**Zero-shot Object Detection & Segmentation System Powered by OWL-V2 and MobileSAM**  
Cloud-Ready · Batch Inference · S3-Integrated · SageMaker-Compatible

---

The system supports:

- Single-image inference
- Batch inference (multiple images)
- Folder / ZIP inference
- GPU batch acceleration
- Cloud-ready API interface
- S3-based storage
- SageMaker real-time inference compatibility
---

## System Pipeline
```

Frontend (Web)
      │
      ▼
Backend API (FastAPI · Docker · ECS/Fargate)
      │
      ├── Upload images to S3
      └── Invoke SageMaker Endpoint
                │
                ▼
        OWL-V2 + MobileSAM
                │
                ▼
  JSON results / S3 image outputs
      │
      ▼
User Interface
```

---


## Project Structure

```
open_world_annotator/
│
├── backend/
│   ├── api/                ← FastAPI routes (single/batch/zip)
│   │     ├── app.py
│   │     └── routers/
│   │
│   ├── inference/          ← Unified inference engine (batch-first)
│   │     ├── inference_core.py
│   │     ├── loader.py
│   │     └── utils/
│   │
│   ├── models/
│   │     ├── mobile_sam.pt
│   │     ├── owl_v2_downloaded.json
│   │     └── ...
│   │
│   └── training/
│
├── frontend/               ← Minimal web interface for uploads/tests
│
├── sagemaker_container/    ← Docker image for SageMaker GPU inference
│   ├── Dockerfile
│   ├── inference.py        ← model_fn, input_fn, predict_fn, output_fn
│   ├── requirements.txt
│   └── model/
│         └── mobile_sam.pt
│
├── docker/                 ← Local/API container files
│   ├── Dockerfile.api
│   └── start.sh
│
└── README.md
```


## 🎓 Acknowledgements

This project was developed as a group assignment for **Fall_2025.CSE.4207.01 - Cloud Computing With Big Data Applications** at **Washington University in St. Louis**.

**Team Members:**
* Haohao Xia
* Xuyang Zheng
* Zilong Wang
* Alexis Liao

*Please note: This repository is a copy of the original group work hosted at [here](https://github.com/Gin-Xia/open_world_annotator/tree/main).*
