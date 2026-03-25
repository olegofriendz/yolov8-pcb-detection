# src/training/train.py
import logging
from pathlib import Path
from typing import Dict

from ultralytics import YOLO
from src.config import load_config

logger = logging.getLogger(__name__)


def train_model(config: Dict) -> None:

    model_name = config["training"].get("model", "yolov8m.pt")
    data_yaml = config["training"].get("data_yaml")
    epochs = config["training"].get("epochs", 100)
    imgsz = config["training"].get("imgsz", 640)
    batch = config["training"].get("batch", 8)
    device = config["training"].get("device", 0)  # 0 = первая GPU
    experiment_name = config["training"]["project_name"]

    if data_yaml and not Path(data_yaml).exists():
        logger.warning(f"Файл {data_yaml} не найден в текущей папке. Убедитесь, что программа запущена из корня проекта.")

    logger.info(f"Загружаем модель: {model_name}")
    model = YOLO(model_name)

    logger.info(f"Старт обучения: epochs={epochs}, imgsz={imgsz}, batch={batch}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=experiment_name,
        exist_ok=True,
        verbose=True
    )
    
    logger.info("✅ Обучение завершено!")
    
    if hasattr(results, 'best') and results.best:
        logger.info(f"Лучшие веса сохранены: {results.best}")
    if hasattr(results, 'results_dict'):
        logger.info(f"Метрики: {results.results_dict}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = load_config("config.yaml")
    
    train_model(config)

if __name__ == "__main__":
    main()