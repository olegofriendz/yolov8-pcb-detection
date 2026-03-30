#!/usr/bin/env python3
"""Конвертация ONNX → RKNN для Orange Pi (RK3588) — версия для rknn-toolkit2 v2.3.2"""
from rknn.api import RKNN
import os
import argparse

def convert_onnx_to_rknn(
    onnx_path: str,
    target_platform: str = "rk3588",
    output_dir: str = "weights/rknn",
    do_quantization: bool = False,
    dataset_txt: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Инициализация RKNN для {target_platform}...")
    rknn = RKNN(verbose=True)
    
    # Конфигурация под RK3588
    print("⚙️ Настройка конфигурации...")
    
    # Базовый конфиг (ТОЛЬКО параметры для config())
    config = {
        'target_platform': target_platform,
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'optimization_level': 3,
    }
    
    # Добавляем quantized_dtype ТОЛЬКО если включено квантование
    if do_quantization:
        if not dataset_txt or not os.path.exists(dataset_txt):
            raise ValueError("Для квантования нужен файл dataset.txt с путями к изображениям")
        config['quantized_dtype'] = 'w8a8'  # INT8 для весов и активаций
    
    # 🔥 КРИТИЧНО: НЕ добавляем 'do_quantization' в config! Он только для build()
    print(f"📋 Конфиг для config(): {config}")
    rknn.config(**config)
    
    # Загрузка ONNX
    print(f"📥 Загрузка ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки ONNX (код {ret})")
    
    # Сборка модели — здесь указываем do_quantization!
    if do_quantization:
        print("🔨 Сборка с квантованием INT8 (w8a8)...")
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        print("🔨 Сборка без квантования (FP16)...")
        ret = rknn.build(do_quantization=False)  # 🔥 Здесь, а не в config!
    
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка сборки (код {ret})")
    
    # Экспорт
    output_name = os.path.basename(onnx_path).replace('.onnx', '.rknn')
    output_path = os.path.join(output_dir, output_name)
    
    print(f"💾 Экспорт в {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка экспорта (код {ret})")
    
    rknn.release()
    print(f"✅ Готово! Модель: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Путь к .onnx файлу")
    parser.add_argument("--platform", default="rk3588", choices=["rk3588", "rk3568", "rk3566"])
    parser.add_argument("--output", default="weights/rknn")
    parser.add_argument("--quantize", action="store_true", help="Включить квантование")
    parser.add_argument("--dataset", help="Путь к dataset.txt (только для --quantize)")
    args = parser.parse_args()
    
    convert_onnx_to_rknn(
        args.onnx,
        args.platform,
        args.output,
        do_quantization=args.quantize,
        dataset_txt=args.dataset
    )