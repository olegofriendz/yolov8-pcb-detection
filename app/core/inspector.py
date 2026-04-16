import cv2
import time
import numpy as np
from pathlib import Path
import sys
from threading import Thread

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
    
    STEP_MM = 4.0 # шаг для ручного управления

    PX_PER_MM = 10.0 # количество пикселей в мм
    CROP_SIZE_MM = 640 / PX_PER_MM # размер кропа в мм

    PLATE_WIDTH = 200 # мм
    PLATE_HEIGHT = 200
    OVERLAP_PERCENT = 40 # перекрытие в процентах


    def __init__(self, model_path=None):
        print("\n[1/3] Запуск камеры...")
        self.camera = CameraCapture(camera_id=self.CAMERA_ID, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT)
        print(f"\nКамера {self.CAMERA_ID} запущена ({self.FRAME_WIDTH}x{self.FRAME_HEIGHT}).")

        print("\n[2/3] Загрузка модели детекции...\n")
        self.detector = RKNNdetect(model_path=model_path, img_size=640, conf_thres=self.CONF_THRES, 
                                   nms_thres=self.NMS_THRES, num_classes=self.NUM_CLASSES, class_names=self.CLASS_NAMES)
        self.detector.load_rknn_model()

        print("\n[3/3] Подключение к контроллеру движения...\n")
        self.motion = MotionContoller(port=self.COM_PORT, baud=self.BAUD_RATE)
        self.motion.connect()

        self.total_detections = 0
        self.current_frame = None # для gui
        self.current_detections = []

        self.running = True
        self.processing_thread = Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    # поток для постоянной обработки кадров
    def _process_loop(self):
        while self.running:
            self.process_frame()
            time.sleep(0.05)

    # обработка кадра: захват -> детекция -> отрисовка
    def process_frame(self, x_off=None, y_off=None):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.current_frame = None
            return None, []
        
        detections, crop_frame, (x_off_actual, y_off_actual) = self.detector.detect(
            frame, x_off=x_off, y_off=y_off
        )
        
        frame_with_boxes = self.draw_detections(frame, detections)
        frame_with_boxes = self.draw_info(frame_with_boxes, detections, x_off_actual, y_off_actual)

        self.current_frame = frame_with_boxes
        self.current_detections = detections
        
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
    
    
    # ручное управление камерой
    def manual_control_step(self, direction=None):
        if direction == 'up':
            self.motion.move_relative(x=-self.STEP_MM)
        elif direction == 'down':
            self.motion.move_relative(x=self.STEP_MM)
        elif direction == 'left':
            self.motion.move_relative(y=-self.STEP_MM)
        elif direction == 'right':
            self.motion.move_relative(y=self.STEP_MM)
        elif direction == 'home':
            self.motion.home()
        elif direction == 'set_home':
            self.motion.set_home()
        
        # self.process_frame()

    # автоматический проход платы
    def scan_plate(self):
        points = self.generate_snake_points(plate_width=self.PLATE_WIDTH, plate_height=self.PLATE_HEIGHT, crop_size=self.CROP_SIZE_MM, overlap_percent=self.OVERLAP_PERCENT)
        print(f"Всего точек: {len(points)}")

        self.motion.set_home()
        for x, y in points:
            print(f"Движение в ({x:.1f}, {y:.1f})")
            self.motion.move_absolute(x, y, feedrate=2000)
            self.motion.wait_for_stop(show_live_callback=self.update_live_frame)
            
            # self.process_frame()

        print("Сканирование завершено.")
        self.motion.home()

    def update_live_frame(self):
        # self.process_frame()
        pass

    # получить список координат для прохода
    def generate_snake_points(self, plate_width, plate_height, crop_size=64, overlap_percent=20):
        step = crop_size * (1 - overlap_percent / 100)
        
        points = []
        y = 0.0
        direction = 1  # 1 = вправо, -1 = влево
        
        while y < plate_height:
            if direction == 1:
                x = 0.0
                while x < plate_width:
                    points.append((x, y))
                    x += step
                if points[-1][0] < plate_width:
                    points.append((plate_width, y))
            else:
                x = plate_width
                while x > 0:
                    points.append((x, y))
                    x -= step
                if points[-1][0] > 0:
                    points.append((0, y))
            
            y += step
            direction *= -1  # смена направления
        
        if points[-1][1] < plate_height:
            # Добавляем строку по нижнему краю
            y = plate_height
            if direction == 1:
                # Последняя строка была справа налево, значит сейчас едем слева направо
                x = 0.0
                while x <= plate_width:
                    points.append((x, y))
                    x += step
                if points[-1][0] < plate_width:
                    points.append((plate_width, y))
            else:
                # Последняя строка была слева направо, значит сейчас едем справа налево
                x = plate_width
                while x >= 0:
                    points.append((x, y))
                    x -= step
                if points[-1][0] > 0:
                    points.append((0, y))
                
        return points

    # получить данные всех элементов на плате
    def get_components_in_crop(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return []

        h, w = frame.shape[:2]
        x_off = (w - self.detector.img_size) // 2
        y_off = (h - self.detector.img_size) // 2
        
        crop_frame = frame[y_off:y_off+640, x_off:x_off+640]
        img_input = self.detector.preprocess(crop_frame)
        outputs = self.detector.rknn.inference(inputs=[img_input], data_format='nchw')
        
        raw_detections = self.detector.postprocess(outputs, 0, 0, (640, 640))
        
        components = []
        for det in raw_detections:
            box = det['box']  # [x1, y1, x2, y2] в координатах кропа
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            components.append({
                'class_name': self.CLASS_NAMES[det['class']],
                'class_id': det['class'],
                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                'center': [cx, cy],    # [center_x, center_y]
                'confidence': det['score']
            })
        
        return components
    
    # получить последний кадр для gui
    def get_current_frame(self):
        return self.current_frame

    # завершить работу
    def shutdown(self):
        print("Завершение работы системы...")
        self.running = False

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

        self.camera.release()
        self.detector.release()
        self.motion.disconnect()
        cv2.destroyAllWindows()
        print("Система остановлена.")