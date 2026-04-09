#!/usr/bin/env python3
from rknn.api import RKNN
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse

def convert_onnx_to_rknn(
    onnx_path: str,
    target_platform: str = "rk3588",
    output_dir: str = "runs/detect/one-board-dataset/weights",
    do_quantization: bool = False,
    dataset_txt: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Анализ входного ONNX
    print(f"📥 Анализ ONNX: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    
    print("\n📊 ONNX ВХОДЫ:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"   {inp.name}: {shape}")
    
    print("\n📊 ONNX ВЫХОДЫ:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"   {out.name}: {shape}")
    
    # 2. Тестовый инференс через ONNX (для сравнения)
    print("\n🔬 Тест ONNX инференса...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    test_input = (np.random.rand(1, 3, 640, 640).astype(np.float32) * 0.5)  # 0-0.5 как после /255
    onnx_output = session.run(None, {input_name: test_input})[0]
    
    print(f"   ONNX output shape: {onnx_output.shape}")
    print(f"   ONNX output min/max: {onnx_output.min():.4f} / {onnx_output.max():.4f}")
    print(f"   ONNX output mean: {onnx_output.mean():.4f}")
    
    # 3. Инициализация RKNN
    print(f"\n🔄 Инициализация RKNN для {target_platform}...")
    rknn = RKNN(verbose=True)
    
    # 4. Конфигурация
    print("\n⚙️ Конфигурация RKNN:")
    config = {
        'target_platform': target_platform,
        'mean_values': [[0, 0, 0]],
        'std_values': [[1, 1, 1]],
        'optimization_level': 3,
    }
    if do_quantization:
        if not dataset_txt or not os.path.exists(dataset_txt):
            raise ValueError("Нужен dataset.txt для квантования")
        config['quantized_dtype'] = 'w8a8'
    
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    rknn.config(**config)
    
    # 5. Загрузка и сборка
    print(f"\n📥 Загрузка ONNX в RKNN...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки ONNX (код {ret})")
    
    if do_quantization:
        print("🔨 Сборка INT8...")
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        print("🔨 Сборка FP16...")
        ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка сборки (код {ret})")
    
    # 6. Экспорт
    output_name = os.path.basename(onnx_path).replace('.onnx', '.rknn')
    output_path = os.path.join(output_dir, output_name)
    print(f"\n💾 Экспорт: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка экспорта (код {ret})")
    
    # 7. Тест RKNN инференса
    print(f"\n🔬 Тест RKNN инференса...")
    ret = rknn.init_runtime(target=target_platform, perf_debug=False, eval_mem=False)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка init_runtime (код {ret})")
    
    # 🔥 Важно: input для RKNN должен быть в том же формате что и для ONNX
    rknn_output = rknn.inference(inputs=[test_input])
    rknn_output = rknn_output[0]
    
    print(f"   RKNN output shape: {rknn_output.shape}")
    print(f"   RKNN output min/max: {rknn_output.min():.4f} / {rknn_output.max():.4f}")
    print(f"   RKNN output mean: {rknn_output.mean():.4f}")
    
    # 8. Сравнение ONNX vs RKNN
    print(f"\n📊 СРАВНЕНИЕ ONNX ↔ RKNN:")
    if onnx_output.shape == rknn_output.shape:
        print(f"   ✅ Shape совпадает: {onnx_output.shape}")
    else:
        print(f"   ❌ Shape разный: ONNX {onnx_output.shape} vs RKNN {rknn_output.shape}")
    
    diff = np.abs(onnx_output - rknn_output)
    print(f"   Max absolute diff: {diff.max():.6f}")
    print(f"   Mean absolute diff: {diff.mean():.6f}")
    
    if diff.max() < 0.1:
        print(f"   ✅ Выходы практически идентичны (разница < 0.1)")
    elif diff.max() < 1.0:
        print(f"   ⚠️ Есть небольшие расхождения (разница < 1.0) — нормально для FP16")
    else:
        print(f"   ❌ Большие расхождения — проверьте mean/std_values!")
    
    rknn.release()
    print(f"\n✅ Конвертация завершена: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--output", default="weights/rknn")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    
    convert_onnx_to_rknn(
        args.onnx, args.platform, args.output,
        do_quantization=args.quantize,
        dataset_txt=args.dataset
    )