FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    tk8.6 \
    libtk8.6 \
    libxcb1 \
    libxkbcommon0 \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir requirements/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

RUN pip install --no-cache-dir .

CMD ["python", "app/main.py"]