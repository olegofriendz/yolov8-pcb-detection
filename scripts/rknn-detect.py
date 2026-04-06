import cv2
import numpy as np
import time
import threading
import serial
from rknn.api import RKNN

RKNN_MODEL = "runs/detect/one-board-dataset/weights/best.rknn"
IMG_SIZE = 640
CONF_THRES = 0.6 # уверенность
NMS_THRES = 0.8 # близкие объекты
CAMERA_ID = 0
NUM_CLASSES = 5
CLASS_NAMES = ['chip-capacitor', 'chip-resistor', 'diode', 'ic', 'transistor']

PORT = '/dev/ttyUSB0'
BAUD = 115200


class CameraCapture:
    def __init__(self, camera_id, width, height):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 450)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
    
    def _capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None
    
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()



def load_rknn_model(model_path, target_platform='rk3588'):
    print(f"Загрузка RKNN модели: {model_path}.")
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"❌ Ошибка загрузки .rknn (код {ret})")
    print(f"Инициализация NPU runtime...")
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

def postprocess_yolov8(outputs, orig_shape, x_off, y_off, conf_thres=0.5, nms_thres=0.45, num_classes=5):
    output = outputs[0][0].transpose(1, 0)
    boxes = output[:, :4]
    scores = 1 / (1 + np.exp(-output[:, 4:4+num_classes]))
    class_max = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    mask = class_scores >= conf_thres
    boxes, class_max, class_scores = boxes[mask], class_max[mask], class_scores[mask]
    if len(boxes) == 0: return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    detections = []
    for cls in np.unique(class_max):
        m = class_max == cls
        # NMS [x, y, w, h]
        xywh = np.stack([x1[m], y1[m], boxes[m, 2], boxes[m, 3]], axis=1)
        xyxy = np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1)
        
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), class_scores[m].tolist(), conf_thres, nms_thres)

        if len(indices) > 0:
            for idx in indices.flatten():
                x1_c, y1_c, x2_c, y2_c = xyxy[idx]
                
                # если бокс касается или пересекает любую границу кропа (0 или 640) — пропускаем
                if x1_c <= 0 or y1_c <= 0 or x2_c >= IMG_SIZE or y2_c >= IMG_SIZE:
                    continue

                box = np.array([x1_c, y1_c, x2_c, y2_c])
                box[[0, 2]] += x_off
                box[[1, 3]] += y_off
                box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_shape[1])
                box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_shape[0])
                detections.append({'box': box, 'class': int(cls), 'score': float(class_scores[m][idx])})

    if len(detections) > 1:
        detections.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        for det in detections:
            is_inside = False
            for kept in keep:
                if (det['box'][0] >= kept['box'][0] and det['box'][1] >= kept['box'][1] and
                    det['box'][2] <= kept['box'][2] and det['box'][3] <= kept['box'][3]):
                    is_inside = True
                    break
            if not is_inside:
                keep.append(det)
        detections = keep

    # удаление элементов по краям
    crop_x1, crop_y1 = x_off, y_off
    crop_x2, crop_y2 = x_off + IMG_SIZE, y_off + IMG_SIZE
    detections = [
        det for det in detections
        if (det['box'][0] >= crop_x1 and det['box'][1] >= crop_y1 and
            det['box'][2] <= crop_x2 and det['box'][3] <= crop_y2)
    ]
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

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    def send(cmd: str):
        ser.write(f"{cmd}\n".encode())
        while True:
            line = ser.readline().decode().strip()
            if not line: continue
            print(f"  <- {line}")
            if line == 'ok' or line.startswith('error'):
                break

    try:
        rknn = load_rknn_model(RKNN_MODEL, target_platform='rk3588')
        print(f"Индекс камеры: {CAMERA_ID}.")
        camera = CameraCapture(CAMERA_ID, width=3840, height=2160)
        fps_counter = []
        print("Нажмите 'q' для выхода")

        # ser.write(b'$X\n')
        # ser.write(b'G90\n')
        # ser.write(b'G1 X0 Y0 F500\n') # начало координат
        # ser.write(b'G1 X200 Y0 F500\n')
        # ser.write(b'G1 X200 Y200 F500\n')
        # ser.write(b'G1 X0 Y200 F500\n')
        # ser.write(b'G1 X0 Y0 F500\n')

        step = 3.0
        send('G91')

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            x_off = (w - IMG_SIZE) // 2
            y_off = (h - IMG_SIZE) // 2
            crop_frame = frame[y_off:y_off+IMG_SIZE, x_off:x_off+IMG_SIZE]

            img_input = preprocess_image(crop_frame, IMG_SIZE)

            start = time.time()
            outputs = rknn.inference(inputs=[img_input], data_format='nchw')
            elapsed = time.time() - start
            fps_counter.append(1 / elapsed if elapsed > 0 else 0)
            
            detections = postprocess_yolov8(outputs, orig_shape=(h, w), x_off=x_off, y_off=y_off, 
                                            conf_thres=CONF_THRES, nms_thres=NMS_THRES, num_classes=NUM_CLASSES)
            frame = draw_detections(frame, detections, CLASS_NAMES)

            cv2.rectangle(frame, (x_off, y_off), (x_off + IMG_SIZE, y_off + IMG_SIZE), (0, 255, 0), 1)
            cv2.putText(frame, f"Detect: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("RKNN Detection", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            if key in (82, 65362):
                send(f"G1 X-{step} Y0 F800")
            elif key in (84, 65364): 
                send(f"G1 X{step} Y0 F800") 
            elif key in (81, 65361):
                send(f"G1 X0 Y-{step} F800")
            elif key in (83, 65363):
                send(f"G1 X0 Y{step} F800")

    finally:
        ser.close()
        send('G90')
        camera.release()
        cv2.destroyAllWindows()
        rknn.release()
        print(f"Средний FPS: {np.mean(fps_counter):.1f}")


if __name__ == "__main__":
    main()