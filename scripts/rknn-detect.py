#!/usr/bin/env python3

import cv2
import numpy as np
import time
from rknn.api import RKNN

# === КОНФИГУРАЦИЯ ===
RKNN_MODEL = "runs/detect/one-board-dataset/weights/best.rknn"
IMG_SIZE = 640
CONF_THRES = 0.5
CAMERA_ID = 0
# ====================

def load_rknn_model(model_path, target_platform='rk3588'):
    """Загрузка модели RKNN"""
    print(f"🔄 Загрузка RKNN модели: {model_path}")
    rknn = RKNN(verbose=False)
    
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки .rknn (код {ret})")
    
    print(f"🚀 Инициализация NPU runtime (target={target_platform})...")
    # 🔥 КРИТИЧНО: указать target для работы с реальным NPU
    ret = rknn.init_runtime(
        target=target_platform,
        perf_debug=False,    # Не выводить детальную статистику
        eval_mem=False       # Не оценивать память
    )
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка инициализации NPU (код {ret})")
    
    return rknn

def preprocess_image(img, img_size=640):
    """Препроцессинг как при обучении YOLO"""
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = img_normalized.transpose(2, 0, 1)[None, ...]  # (1, 3, 640, 640)
    return img_input

def main():
    rknn = load_rknn_model(RKNN_MODEL, target_platform='rk3588')
    
    print(f"📷 Открытие камеры {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        raise RuntimeError("❌ Не удалось открыть камеру")
    
    print("✅ Нажмите 'q' для выхода")
    
    fps_counter = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Препроцессинг
        img_input = preprocess_image(frame, IMG_SIZE)
        
        # Инференс
        start = time.time()
        outputs = rknn.inference(inputs=[img_input])
        elapsed = time.time() - start
        fps_counter.append(1 / elapsed if elapsed > 0 else 0)
        
        # ⚠️ ВНИМАНИЕ: outputs — это сырые тензоры
        # Для полноценной детекции нужен постпроцессинг (NMS, декодирование боксов)
        # Для быстрого теста просто покажем FPS
        avg_fps = np.mean(fps_counter[-30:])
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {elapsed*1000:.1f}ms", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("RKNN Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    print(f"📊 Средний FPS: {np.mean(fps_counter):.1f}")

if __name__ == "__main__":
    main()