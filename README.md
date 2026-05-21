# YOLOv8 PCB Detection

Real-time detection and classification of electronic components on printed circuit boards using YOLOv8.


## Environment Setup

> **Note:** Python 3.10 is required for Orange Pi to ensure compatibility with the pre-built ARM64 dependencies.

1. **Create and activate a virtual environment:**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install architecture-specific dependencies (RKNN Toolkit):**
   ```bash
   pip install requirements/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```

3. **Install the project in editable mode:**
   ```bash
   pip install -e .
   ```

## Usage

After installation, use CLI commands from any directory:

| Command | Description |
|---------|-------------|
| `pcb-download` | Download dataset from Roboflow |
| `pcb-tile` | Tile images for better training |
| `pcb-train` | Train YOLOv8 model |

### Examples

```bash
# Download dataset
pcb-download

# Tile images (default: 640px, 20% overlap)
pcb-tile --size 640 --overlap 0.2

# Train model
pcb-train

# best.pt -> best.onnx
python app/utils/converters/pt_to_onnx.py --pt runs/detect/one-board-dataset/weights/best.pt

# best.onnx -> best.rknn
python app/utils/converters/onnx_to_rknn.py --onnx runs/detect/one-board-dataset/weights/best.onnx
```

## 🛠️ Tech Stack

- **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
