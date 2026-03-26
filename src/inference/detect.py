# src/inference/detect.py
import logging
import argparse
import cv2
from pathlib import Path
from typing import Dict
from ultralytics import YOLO
from src.config import load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

def run_detection(config: Dict, camera_id: int, conf: float, imgsz: int) -> None:
    
    runs_dir = PROJECT_ROOT / config["paths"].get("runs_dir", "runs")
    project_name = config["training"]["project_name"]
    weights_path = runs_dir / "detect" / project_name / "weights" / "best.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {weights_path}")
    
    logger.info(f"Загружаем модель: {weights_path}")
    model = YOLO(str(weights_path))
    
    logger.info(f"Открываем камеру {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    
    if not cap.isOpened():
        logger.error("Камера не найдена!")
        return
    
    logger.info("Нажмите 'q' для выхода.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Не удалось получить кадр.")
            break
        
        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=config["inference"].get("device", "cuda:0"),
            verbose=False,
            agnostic_nms=True
        )
        
        annotated_frame = results[0].plot(
            line_width=2,
            font_size=1.2,
            labels=True,
            conf=True
        )
        
        display_frame = cv2.resize(annotated_frame, (1920, 1080))
        cv2.imshow("PCB Detection", display_frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Детекция завершена.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="PCB Component Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()
    
    config = load_config("config.yaml")
    run_detection(config, args.camera, args.conf, args.imgsz)


if __name__ == "__main__":
    main()