#!/usr/bin/env python3
import cv2
import numpy as np
import time
from rknn.api import RKNN

# === КОНФИГУРАЦИЯ ===
RKNN_MODEL = "runs/detect/one-board-dataset/weights/best.rknn"
IMG_SIZE = 640
CONF_THRES = 0.25  # 🔥 Понизили для теста
NMS_THRES = 0.45
CAMERA_ID = 0
CLASS_NAMES = ["chip-resistor", "chip-capacitor", "diode", "ic", "transistor"]  # 🔥 Укажите ваши классы
NUM_CLASSES = 5  # 🔥 Сколько классов в датасете?
# ====================

def load_rknn_model(model_path, target_platform='rk3588'):
    """Загрузка модели RKNN"""
    print(f"🔄 Загрузка RKNN модели: {model_path}")
    rknn = RKNN(verbose=True)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки .rknn (код {ret})")
    
    print(f"🚀 Инициализация NPU runtime (target={target_platform})...")
    ret = rknn.init_runtime(
        target=target_platform,
        perf_debug=False,
        eval_mem=False
    )
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка инициализации NPU (код {ret})")
    
    # 🔥 Правильный способ получить инфо о модели
    print("✅ Модель загружена успешно")
    return rknn

def preprocess_image(img, img_size=640):
    """Препроцессинг как при обучении YOLO"""
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = img_normalized.transpose(2, 0, 1)[None, ...]  # (1, 3, 640, 640)
    return img_input, img_resized

def debug_outputs(outputs, frame_count=0):
    """🔍 Детальная проверка выходов модели"""
    if frame_count % 30 != 0 and frame_count != 0:
        return  # Показываем не каждый кадр
    
    print("\n" + "="*60)
    print(f"🔬 ДИАГНОСТИКА ВЫХОДОВ (кадр #{frame_count})")
    print("="*60)
    
    for i, output in enumerate(outputs):
        print(f"\n📊 Output[{i}]:")
        print(f"   Shape: {output.shape}")
        print(f"   Dtype: {output.dtype}")
        print(f"   Min:   {output.min():.6f}")
        print(f"   Max:   {output.max():.6f}")
        print(f"   Mean:  {output.mean():.6f}")
        
        # Проверка на NaN/Inf
        if np.isnan(output).any():
            print("   ⚠️ WARNING: Обнаружены NaN значения!")
        if np.isinf(output).any():
            print("   ⚠️ WARNING: Обнаружены Inf значения!")
    
    # 🔥 Проверка формата YOLOv8
    output = outputs[0]
    if output.shape == (1, 84, 8400):
        print("\n✅ Формат: (1, 84, 8400) - NCHW (ожидаемый для YOLOv8)")
        expected_classes = 84 - 4
        print(f"✅ Ожидаемое количество классов: {expected_classes}")
    elif output.shape == (1, 8400, 84):
        print("\n✅ Формат: (1, 8400, 84) - NHWC")
    elif output.shape == (1, 25200, 85):
        print("\n⚠️ Формат: (1, 25200, 85) - YOLOv5/v7 формат!")
    else:
        print(f"\n⚠️ НЕОЖИДАННЫЙ формат: {output.shape}")
        print("🔥 Возможно модель конвертирована с другими параметрами!")
    
    print("="*60 + "\n")

def postprocess_yolov8(outputs, img_shape, conf_thres=0.25, nms_thres=0.45, num_classes=1):
    output = outputs[0]
    
    # Транспозиция
    if output.shape[1] > output.shape[2]:
        output = output[0].transpose(1, 0)  # (8400, 84)
    else:
        output = output[0]
    
    boxes = output[:, :4]
    scores = output[:, 4:4+num_classes]
    
    # 🔥 КРИТИЧНО: Применяем sigmoid к scores!
    scores = 1 / (1 + np.exp(-scores))  # Сигмоида
    
    class_max = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    
    print(f"📊 Scores (после sigmoid): min={class_scores.min():.4f}, max={class_scores.max():.4f}")
    
    mask = class_scores >= conf_thres
    print(f"📊 Детекций выше порога {conf_thres}: {mask.sum()} из {len(class_scores)}")
    
    if mask.sum() == 0:
        print("⚠️ Нет детекций выше порога confidence!")
        print("🔥 Попробуйте понизить CONF_THRES до 0.1 для теста")
        return []
    
    boxes = boxes[mask]
    class_max = class_max[mask]
    class_scores = class_scores[mask]
    
    # Конвертация cx,cy,w,h -> x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Масштабирование к оригинальному изображению
    scale_x = img_shape[1] / IMG_SIZE
    scale_y = img_shape[0] / IMG_SIZE
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    # NMS
    detections = []
    unique_classes = np.unique(class_max)
    
    for cls in unique_classes:
        cls_mask = class_max == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = class_scores[cls_mask]
        
        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(),
            cls_scores.tolist(),
            conf_thres,
            nms_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                detections.append({
                    'box': cls_boxes[idx],
                    'class': int(cls),
                    'score': float(cls_scores[idx])
                })
    
    print(f"✅ Итоговых детекций после NMS: {len(detections)}")
    return detections

def draw_detections(frame, detections, class_names):
    """Отрисовка детекций на кадре"""
    for det in detections:
        box = det['box'].astype(int)
        x1, y1, x2, y2 = box
        cls = det['class']
        score = det['score']
        
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names[cls] if cls < len(class_names) else cls}: {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def main():
    rknn = load_rknn_model(RKNN_MODEL, target_platform='rk3588')
    
    print(f"📷 Открытие камеры {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("❌ Не удалось открыть камеру")
    
    print("✅ Нажмите 'q' для выхода, 'd' для подробной диагностики")
    
    fps_counter = []
    frame_count = 0
    debug_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig_shape = frame.shape[:2]
        
        img_input, _ = preprocess_image(frame, IMG_SIZE)
        
        start = time.time()
        outputs = rknn.inference(inputs=[img_input])
        elapsed = time.time() - start
        fps_counter.append(1 / elapsed if elapsed > 0 else 0)
        
        # 🔥 Диагностика по нажатию 'd' или первый кадр
        if debug_mode or frame_count == 1:
            debug_outputs(outputs, frame_count)
            debug_mode = False
        
        detections = postprocess_yolov8(
            outputs, 
            orig_shape, 
            conf_thres=CONF_THRES, 
            nms_thres=NMS_THRES,
            num_classes=NUM_CLASSES
        )
        
        frame = draw_detections(frame, detections, CLASS_NAMES)
        
        avg_fps = np.mean(fps_counter[-30:])
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detect: {len(detections)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {CONF_THRES}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("RKNN Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = True
    
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    print(f"📊 Средний FPS: {np.mean(fps_counter):.1f}")

if __name__ == "__main__":
    main()