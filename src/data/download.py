# src/data/download.py
import os
import logging
from pathlib import Path
from typing import Optional, Dict
from roboflow import Roboflow
from src.config import load_config

logger = logging.getLogger(__name__)


def download_dataset(config: Dict) -> Path:

    api_key = config["dataset"].get("roboflow_api_key") # .env -> config
    workspace = config["dataset"]["workspace"]
    project_name = config.get("dataset", {}).get("name", "one-board-dataset")
    version = config["dataset"]["version"]
    format = config["dataset"]["download_format"]
    download_location = config["paths"]["dataset_dir"]
    
    if not api_key:
        raise ValueError(
            "API-ключ Roboflow не найден! "
            "Проверьте файл .env и переменную ROBOFLOW_API_KEY"
        )
    
    logger.info(f"Инициализация Roboflow (workspace: {workspace})...")
    
    try:
        rf = Roboflow(api_key=api_key)
        
        logger.info(f"Подключение к проекту: {workspace}/{project_name}...")
        project = rf.workspace(workspace).project(project_name)
        
        logger.info(f"Загрузка версии {version} в формате {format}...")
        version_obj = project.version(version)

        dataset_obj = version_obj.download(format, location=download_location)
        dataset_path = dataset_obj.location
        
        logger.info(f"✅ Датасет успешно загружен: {dataset_path}")
        return Path(dataset_path)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке датасета: {e}")
        raise


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = load_config("config.yaml")
    download_dataset(config)

if __name__ == "__main__":
    main()