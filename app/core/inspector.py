import cv2
import time
import numpy as np
from pathlib import Path
import sys

from hardware.camera import CameraCapture
from hardware.motion import MotionContoller
from inference.rknn_detect import RKNNdetect

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class Inspector:

    CAMERA_ID = 0
    FRAME_WIDTH = 3840
    FRAME_HEIGHT = 2160
    
    COM_PORT = "/dev/ttyUSB0"
    BAUD_RATE = 115200
    
    CONF_THRES = 0.65
    NMS_THRES = 0.4
    NUM_CLASSES = 5
    CLASS_NAMES = ['chip-capacitor', 'chip-resistor', 'diode', 'ic', 'transistor']
    
    STEP_MM = 3.0 # шаг для ручного управления


    def __init__(self, model_path=None):
        print("\n[1/3] Запуск камеры...")
        self.camera = CameraCapture(camera_id=self.CAMERA_ID, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT)
        print(f"\nКамера {self.CAMERA_ID} запущена ({self.FRAME_WIDTH}x{self.FRAME_HEIGHT}).")

        print("\n[2/3] Загрузка модели детекции...")
        self.detector = RKNNdetect(model_path=model_path, img_size=640, conf_thres=self.CONF_THRES, 
                                   nms_thres=self.NMS_THRES, num_classes=self.NUM_CLASSES, class_names=self.CLASS_NAMES)
        self.detector.load_rknn_model()

        print("\n[3/3] Подключение к контроллеру движения...")
        self.motion = MotionContoller(port=self.COM_PORT, baud=self.BAUD_RATE)
        self.motion.connect()
        print(f"Контроллер на {self.COM_PORT} подключён")

        self.total_detections = 0


    # обработка кадра: захват -> детекция -> отрисовка
    def process_frame(self, x_off=None, y_off=None):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None, []
        
        detections, crop_frame, (x_off_actual, y_off_actual) = self.detector.detect(
            frame, x_off=x_off, y_off=y_off
        )
        
        frame_with_boxes = self.draw_detections(frame, detections)
        frame_with_boxes = self.draw_info(frame_with_boxes, detections, x_off_actual, y_off_actual)
        
        return frame_with_boxes, detections
    

    def draw_detections(self, frame, detections):
        colors = [
            (0, 255, 0),    # chip-capacitor
            (255, 0, 0),    # chip-resistor
            (0, 255, 255),  # diode
            (255, 0, 255),  # ic
            (0, 165, 255)   # transistor
        ]

        for det in detections:
            box = det['box'].astype(int)
            x1, y1, x2, y2 = box
            cls = det['class']
            score = det['score']
            
            color = colors[cls] if cls < len(colors) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            class_name = self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else f"cls_{cls}"
            label = f"{class_name}: {score:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    

    def draw_info(self, frame, detections, x_off, y_off):

        cv2.rectangle(frame, (x_off, y_off), (x_off + 640, y_off + 640), (0, 255, 0), 1)
    
        info_lines = [f"Detections: {len(detections)}", "Q - quit | Arrows - move | H - home"]
        
        y_pos = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        return frame
    

    def manual_control(self):

        print("=" * 50)
        print("Управление: Стрелки - движение | H - home | Q - выход")
        print("=" * 50 + "\n")
        
        self.motion.send("G91")
        
        try:
            while True:
                result = self.process_frame()
                if result is None:
                    continue

                frame, detections = result
                
                if frame is not None:
                    cv2.imshow("PCB Inspector - Manual Control", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nВыход из ручного режима...")
                    break
                elif key == ord('h'):
                    print("Парковка (G28)...")
                    self.motion.home()
                elif key == 82:   # Вверх
                    self.motion.move_relative(x=-self.STEP_MM)
                elif key == 84:   # Вниз
                    self.motion.move_relative(x=self.STEP_MM)
                elif key == 81:   # Влево
                    self.motion.move_relative(y=-self.STEP_MM)
                elif key == 83:   # Вправо
                    self.motion.move_relative(y=self.STEP_MM)
                    
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
        finally:
            self.motion.send("G90")


    # завершить работу
    def shutdown(self):
        print("\nЗавершение работы системы...")
        self.camera.release()
        self.detector.release()
        self.motion.disconnect()
        cv2.destroyAllWindows()
        print("Система остановлена.")