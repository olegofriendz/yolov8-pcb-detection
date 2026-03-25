# YOLOv8 PCB Detection

Real-time detection and classification of electronic components on printed circuit boards using YOLOv8.

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
```

## 🛠️ Tech Stack

- **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
