FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/ml/model

# 只装必须的系统库
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# 统一底层数值和 transformers 版本
RUN pip uninstall -y numpy transformers timm opencv-python || true

RUN pip install \
    numpy==1.26.4 \
    transformers==4.41.2 \
    timm==0.9.12 \
    pillow==10.2.0 \
    opencv-python-headless==4.9.0.80 \
    pycocotools==2.0.7

# 安装你项目依赖
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# 拷贝代码（不拷模型）
COPY backend /opt/ml/model/backend
COPY backend/inference/inference.py /opt/ml/model/inference.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/model

ENTRYPOINT ["python", "inference.py"]
