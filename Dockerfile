FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src src