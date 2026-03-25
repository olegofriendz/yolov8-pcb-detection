# src/config.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent # корень


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    config_file = PROJECT_ROOT / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dataset"]["roboflow_api_key"] = os.getenv("ROBOFLOW_API_KEY")

    if not config["dataset"]["roboflow_api_key"]:
        raise ValueError(
            "API-ключ Roboflow не найден в файле .env (ROBOFLOW_API_KEY отсутствует)!"
        )

    project_name = config.get("dataset", {}).get("name", "one-board-dataset") # название проекта
    config["paths"]["dataset_dir"] = str(PROJECT_ROOT / project_name) # полный путь к датасету
    config["paths"]["project_root"] = str(PROJECT_ROOT) # корень
    config["training"]["data_yaml"] = str(PROJECT_ROOT / project_name / "data.yaml") # полный путь к data.yaml в датасете
    config["training"]["project_name"] = project_name # копия названия для training

    return config


def get_config() -> Dict[str, Any]:
    return load_config()
