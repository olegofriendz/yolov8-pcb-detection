#!/usr/bin/env python3
import cv2
import numpy as np
import time
from rknn.api import RKNN

# === КОНФИГУРАЦИЯ ===
RKNN_MODEL = "runs/detect/one-board-dataset/weights/best.rknn"
IMG_SIZE = 640
CONF_THRES = 0.6
NMS_THRES = 0.45
CAMERA_ID = 0
NUM_CLASSES = 5
CLASS_NAMES = ['chip-capacitor', 'chip-resistor', 'diode', 'ic', 'transistor']
# ====================

def load_rknn_model(model_path, target_platform='rk3588'):
    print(f"🔄 Загрузка RKNN модели: {model_path}")
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки .rknn (код {ret})")
    print(f"🚀 Инициализация NPU runtime...")
    ret = rknn.init_runtime(target=target_platform, perf_debug=False, eval_mem=False)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка инициализации NPU (код {ret})")
    return rknn

def preprocess_image(img, img_size=640):
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = img_normalized.transpose(2, 0, 1)[None, ...]
    return img_input

def postprocess_yolov8(outputs, img_shape, conf_thres=0.5, nms_thres=0.45, num_classes=5):
    output = outputs[0][0].transpose(1, 0)  # (8400, 9)
    
    boxes = output[:, :4]  # cx, cy, w, h
    scores = output[:, 4:4+num_classes]
    
    # 🔥 КРИТИЧНО: Применяем sigmoid к scores!
    scores = 1 / (1 + np.exp(-scores))
    
    class_max = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    
    mask = class_scores >= conf_thres
    boxes = boxes[mask]
    class_max = class_max[mask]
    class_scores = class_scores[mask]
    
    if len(boxes) == 0:
        return []
    
    # cx,cy,w,h -> x1,y1,x2,y2
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
        
        indices = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_scores.tolist(), conf_thres, nms_thres)
        
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    'box': cls_boxes[idx],
                    'class': int(cls),
                    'score': float(cls_scores[idx])
                })
    
    return detections

def draw_detections(frame, detections, class_names):
    for det in detections:
        box = det['box'].astype(int)
        x1, y1, x2, y2 = box
        cls = det['class']
        score = det['score']
        
        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names[cls] if cls < len(class_names) else cls}: {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def main():
    rknn = load_rknn_model(RKNN_MODEL, target_platform='rk3588')
    
    print(f"📷 Открытие камеры {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 10)
    if not cap.isOpened():
        raise RuntimeError("❌ Не удалось открыть камеру")
    
    print("✅ Нажмите 'q' для выхода")
    
    fps_counter = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_shape = frame.shape[:2]
        img_input = preprocess_image(frame, IMG_SIZE)
        
        start = time.time()
        outputs = rknn.inference(inputs=[img_input], data_format='nchw')  # 🔥 Явно указываем
        elapsed = time.time() - start
        fps_counter.append(1 / elapsed if elapsed > 0 else 0)
        
        detections = postprocess_yolov8(outputs, orig_shape, CONF_THRES, NMS_THRES, NUM_CLASSES)
        frame = draw_detections(frame, detections, CLASS_NAMES)
        
        avg_fps = np.mean(fps_counter[-30:])
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detect: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("RKNN Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    print(f"📊 Средний FPS: {np.mean(fps_counter):.1f}")

if __name__ == "__main__":
    main()