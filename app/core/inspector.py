import cv2
import time
import json
import numpy as np
from pathlib import Path
import sys
from threading import Thread
from datetime import datetime

from hardware.camera import CameraCapture
from hardware.motion import MotionContoller
from inference.rknn_detect import RKNNdetect

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class Inspector:

    CAMERA_ID = 0
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    
    COM_PORT = "/dev/ttyUSB0"
    BAUD_RATE = 115200
    
    CONF_THRES = 0.65
    NMS_THRES = 0.8
    NUM_CLASSES = 5
    CLASS_NAMES = ['chip-capacitor', 'chip-resistor', 'diode', 'ic', 'transistor']
    
    STEP_MM = 4.0 # шаг для ручного управления

    CROP_SIZE_MM = 75.0 # размер кропа в мм
    PX_PER_MM = 640 / CROP_SIZE_MM # количество пикселей в мм

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

        self.crop_offset = (0, 0)

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
        
        h, w = frame.shape[:2]
        if x_off is None:
            x_off = (w - 640) // 2
        if y_off is None:
            y_off = (h - 640) // 2

        detections, crop_frame, (x_off_actual, y_off_actual) = self.detector.detect(
            frame, x_off=x_off, y_off=y_off
        )
        
        self.crop_offset = (x_off_actual, y_off_actual) # смещение для пересчёта координат

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
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_info(self, frame, detections, x_off, y_off):
        cv2.rectangle(frame, (x_off, y_off), (x_off + 640, y_off + 640), (0, 255, 0), 1)
        info_lines = [f"Detections: {len(detections)}", "Arrows - move", "Z - set home", "H - go home"]
        
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

    # автоматический проход платы
    def scan_plate(self):
        points = self.generate_snake_points(plate_width=self.PLATE_WIDTH,
                                            plate_height=self.PLATE_HEIGHT,
                                            crop_size=self.CROP_SIZE_MM,
                                            overlap_percent=self.OVERLAP_PERCENT)
        print(f"Всего точек: {len(points)}")

        all_components = []

        self.motion.go_zero()
        self.motion.move_relative(100, 100)
        self.motion.set_home() # установить дом в левом верхнем углы платы

        for i, (x, y) in enumerate(points):
            print(f"[{i+1}/{len(points)}] Движение в ({x:.1f}, {y:.1f})")
            self.motion.move_absolute(x, y, feedrate=2000)
            self.motion.wait_for_stop()
            time.sleep(2)

            components = self.scan_at_position(x, y)
            all_components.extend(components)

        print(f"Всего компонентов найдено: {len(all_components)}")
        self.save_results(all_components)
        print(f"Сканирование завершено.")
        self.motion.home()
        # self.motion.go_zero()

    # получить список координат для прохода "змейкой"
    def generate_snake_points(self, plate_width, plate_height, crop_size_mm, overlap_percent):
        step = crop_size_mm * (1 - overlap_percent / 100) # шаг в мм = размер кропа в мм * (1 - перекрытие)
        
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
            y = plate_height
            if direction == 1:
                x = 0.0
                while x <= plate_width:
                    points.append((x, y))
                    x += step
                if points[-1][0] < plate_width:
                    points.append((plate_width, y))
            else:
                x = plate_width
                while x >= 0:
                    points.append((x, y))
                    x -= step
                if points[-1][0] > 0:
                    points.append((0, y))
                
        return points

    # получить все элементы в кропе в мм
    def scan_at_position(self, plate_x, plate_y):
        x_off, y_off = self.crop_offset
        components = []
        for det in self.current_detections:
            box = det['box']
            
            box_crop = [
                round(box[0] - x_off, 4),
                round(box[1] - y_off, 4),
                round(box[2] - x_off, 4),
                round(box[3] - y_off, 4)
            ]
            
            cx = round((box_crop[0] + box_crop[2]) / 2, 4) # центр бокса ??????
            cy = round((box_crop[1] + box_crop[3]) / 2, 4)
            
            cx_crop_mm = round((cx / 640) * self.CROP_SIZE_MM, 4)
            cy_crop_mm = round((cy / 640) * self.CROP_SIZE_MM, 4)

            global_x = round(plate_y + cx_crop_mm, 4)
            global_y = round(plate_x + cy_crop_mm, 4)
            
            components.append({
                'class_name': self.CLASS_NAMES[det['class']],
                'class_id': det['class'],
                'bbox_crop': box_crop,
                'center_px': [cx, cy],
                'center_mm': [global_x, global_y],
                'confidence': round(det['score'], 4)
            })
        
        return components

    # сохранить результаты в файл
    def save_results(self, components, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_results_{timestamp}.json"

        results = {
            'metadata': {
                'plate_width': self.PLATE_WIDTH,
                'plate_height': self.PLATE_HEIGHT,
                'crop_size_mm': self.CROP_SIZE_MM,
                'overlap_percent': self.OVERLAP_PERCENT,
                'total_components': len(components),
                'timestamp': datetime.now().isoformat()
            },
            'components': components
        }

        filepath = Path(__file__).parent.parent / "results" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Результаты сохранены в {filepath}.")
        return str(filepath)
    
    # удалить элементы которые встретились повторно
    def remove_duplicate_components(self, components, distance_threshold_mm=3.0, iou_threshold=0.5):
        if len(components) <= 1:
            return components
        
        by_class = {}
        for comp in components:
            cls = comp['class_id']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(comp)
        
        unique = []
        
        for cls, items in by_class.items():
            # сортировка по confidence
            items = sorted(items, key=lambda x: x['confidence'], reverse=True)
            used = set()
            
            for i, comp1 in enumerate(items):
                if i in used:
                    continue
                
                best = comp1
                duplicates = [i]
                
                for j, comp2 in enumerate(items[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    # расстояние
                    dx = abs(comp1['center_mm'][0] - comp2['center_mm'][0])
                    dy = abs(comp1['center_mm'][1] - comp2['center_mm'][1])
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance > distance_threshold_mm:
                        continue
                    
                    # перекрытие прямоугольников
                    iou = self.compute_iou(comp1['bbox_crop'], comp2['bbox_crop'])
                    
                    if iou >= iou_threshold:
                        duplicates.append(j)
                        if comp2['confidence'] > best['confidence']:
                            best = comp2
                
                # добавление лучшего
                unique.append(best)
                for idx in duplicates:
                    used.add(idx)
        
        return unique
        
    # получить iou боксов    
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


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