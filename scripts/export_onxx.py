"""Экспорт YOLOv8 .pt → ONNX с параметрами для RKNN"""
from ultralytics import YOLO
import argparse
import os

def export_to_onnx(pt_path: str, output_dir: str = "weights/onnx", imgsz: int = 640):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Загрузка модели: {pt_path}")
    model = YOLO(pt_path)
    
    print(f"🔄 Экспорт в ONNX (imgsz={imgsz}, simplify=True)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,      # Критично для RKNN!
        opset=12,           # Совместимая версия ONNX opset
        dynamic=False,      # Фиксированные размеры для NPU
        half=False          # FP32 для начала (FP16 можно потом)
    )
    
    print(f"✅ ONNX модель сохранена: {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Путь к best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входа")
    parser.add_argument("--output", default="weights/onnx")
    args = parser.parse_args()
    
    export_to_onnx(args.pt, args.output, args.imgsz)