#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort

def export_to_onnx(pt_path: str, output_dir: str = "weights/onnx", imgsz: int = 640):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Загрузка .pt модели
    print(f"Загрузка модели: {pt_path}")
    model = YOLO(pt_path)
    
    # 2. Информация о модели
    print(f"Количество классов: {len(model.names)}")
    print(f"Классы: {model.names}")
    
    # 3. Экспорт в ONNX
    print(f"Экспорт в ONNX (imgsz={imgsz})...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=False,
        opset=11,
        dynamic=False,
        half=False,
        batch=1,
        verbose=True
    )
    print(f"✅ ONNX сохранён: {onnx_path}")
    
    # 4. Проверка ONNX файла
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("\nONNX ВХОДЫ:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"   {inp.name}: {shape}")
    
    print("\nONNX ВЫХОДЫ:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"   {out.name}: {shape}")
    
    # 5. Тестовый инференс
    print("\nТЕСТОВЫЙ ИНФЕРЕНС...")
    test_image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8) + 128  # Серое изображение
    
    # Инференс через .pt
    print("Инференс .pt модели...")
    pt_results = model(test_image, verbose=False)
    pt_boxes = pt_results[0].boxes
    print(f"   .pt детекций: {len(pt_boxes)}")
    if len(pt_boxes) > 0:
        print(f"   .pt boxes: {pt_boxes.xyxy[0].cpu().numpy()}")
        print(f"   .pt conf: {pt_boxes.conf[0].cpu().numpy()}")
        print(f"   .pt cls: {pt_boxes.cls[0].cpu().numpy()}")
    
    # Инференс через ONNX
    print("Инференс ONNX модели...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    img_input = test_image.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[None, ...]
    
    onnx_outputs = session.run(None, {input_name: img_input})
    print(f"   ONNX выходов: {len(onnx_outputs)}")
    for i, out in enumerate(onnx_outputs):
        print(f"   Output[{i}] shape: {out.shape}")
        print(f"   Output[{i}] min/max: {out.min():.4f} / {out.max():.4f}")
    
    print("\n✅ Экспорт завершён!")
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Путь к best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="weights/onnx")
    args = parser.parse_args()
    export_to_onnx(args.pt, args.output, args.imgsz)