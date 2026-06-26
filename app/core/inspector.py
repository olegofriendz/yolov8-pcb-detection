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
    
    CONF_THRES = 0.7
    NMS_THRES = 0.4
    NUM_CLASSES = 5
    CLASS_NAMES = ['chip-capacitor', 'chip-resistor', 'diode', 'ic', 'transistor']
    
    STEP_MM = 4.0 # шаг для ручного управления

    CROP_SIZE_PX = 640
    CROP_SIZE_MM = 89.5 # размер кропа в мм
    PX_PER_MM = CROP_SIZE_PX / CROP_SIZE_MM # количество пикселей в мм

    PLATE_WIDTH = 181 # мм
    PLATE_HEIGHT = 276

    PLATE_WIDTH_SNAKE = PLATE_WIDTH - CROP_SIZE_MM * 0.7
    PLATE_HEIGHT_SNAKE = PLATE_HEIGHT - CROP_SIZE_MM * 0.7

    OVERLAP_PERCENT = 50 # перекрытие в процентах

    STANDARD_FILENAME = "standard_plate.json"

    def __init__(self, model_path=None):
        print("\n[1/3] Запуск камеры...")
        self.camera = CameraCapture(camera_id=self.CAMERA_ID, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT)
        print(f"\nКамера {self.CAMERA_ID} запущена ({self.FRAME_WIDTH}x{self.FRAME_HEIGHT}).")

        print("\n[2/3] Загрузка модели детекции...\n")
        self.detector = RKNNdetect(model_path=model_path, 
                                   img_size=self.CROP_SIZE_PX, 
                                   conf_thres=self.CONF_THRES, 
                                   nms_thres=self.NMS_THRES, 
                                   num_classes=self.NUM_CLASSES, 
                                   class_names=self.CLASS_NAMES)
        self.detector.load_rknn_model()

        print("\n[3/3] Подключение к контроллеру движения...\n")
        self.motion = MotionContoller(port=self.COM_PORT, baud=self.BAUD_RATE)
        self.motion.connect()

        self.total_detections = 0
        self.current_frame = None # для gui
        self.current_detections = []
        self.active_defect = None # дефект для подсветки в кадре после проверки
        self.hide_detections = False # флаг для отображения контуров элементов
        self.scan_aborted = False # флаг для принудительной остановки

        self.running = True
        self.processing_thread = Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    # поток для постоянной обработки кадров
    def _process_loop(self):
        while self.running:
            self.process_frame()
            time.sleep(0.05)

    # обработка кадра
    def process_frame(self, x_off=None, y_off=None):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.current_frame = None
            return None, []
        
        h, w = frame.shape[:2]
        if x_off is None:
            x_off = (w - self.CROP_SIZE_PX) // 2
        if y_off is None:
            y_off = (h - self.CROP_SIZE_PX) // 2

        detections, crop_frame, (x_off_actual, y_off_actual) = self.detector.detect(
            frame, x_off=x_off, y_off=y_off
        )
        
        detections = self.filter_nested_boxes(detections) # отсечь вложенные контуры

        # в режиме проверки ошибок контура не отображаются
        if not self.hide_detections:
            frame_with_boxes = self.draw_detections(frame, detections)
        else:
            frame_with_boxes = frame

        frame_with_boxes = self.draw_defect_overlay(frame_with_boxes)
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
        cv2.rectangle(frame, (x_off, y_off), (x_off + self.CROP_SIZE_PX, y_off + self.CROP_SIZE_PX), (0, 255, 0), 1)
        info_lines = [f"Detections: {len(detections)}"]
        
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
        self.scan_aborted = False
        points = self.generate_snake_points(plate_width=self.PLATE_WIDTH_SNAKE,
                                            plate_height=self.PLATE_HEIGHT_SNAKE,
                                            crop_size_mm=self.CROP_SIZE_MM,
                                            overlap_percent=self.OVERLAP_PERCENT)
        print(f"Всего точек: {len(points)}")

        all_components = []

        self.motion.go_zero()
        self.motion.wait_for_stop()
        time.sleep(1)
        self.motion.move_relative(90, 90, feedrate=2000)
        self.motion.wait_for_stop()
        time.sleep(0.3)
        self.motion.set_home() # установить дом в левом верхнем углы платы
        time.sleep(0.3)

        FRAMES_PER_POINT = 5 # количество кадров в одной точке

        for i, (x, y) in enumerate(points):
            if self.scan_aborted:
                print("Сканирование завершено принудительно.")
                break

            print(f"[{i+1}/{len(points)}] Движение в ({x:.1f}, {y:.1f})")
            self.motion.move_absolute(x, y, feedrate=2000)
            self.motion.wait_for_stop()
            time.sleep(1)

            position_components = [] # детекции из нескольких кадров в одном кадре
            for i in range(FRAMES_PER_POINT):
                components = self.scan_at_position(x, y)
                position_components.extend(components)
                print(f"Кадр {i} | Найдено: {len(position_components)}")
                time.sleep(0.2)

            unique_at_position = self.remove_duplicate_components(position_components, distance_threshold_mm=3.0)
            print(f"-> Найдено: {len(position_components)}, уникальных: {len(unique_at_position)}")

            all_components.extend(unique_at_position)
            time.sleep(1)

        unique_components = self.remove_duplicate_components(all_components, distance_threshold_mm=3.0)
        unique_components = self.filter_components_inside_ic(unique_components) # отсечь ложные элементы которые находятся внутри других

        print(f"Всего найдено компонентов: {len(all_components)}")
        print(f"Компонентов после фильтрации: {len(unique_components)}")
        print(f"Сканирование завершено.")
        self.motion.home()
        return unique_components

    # получить список координат для прохода "змейкой"
    def generate_snake_points(self, plate_width, plate_height, crop_size_mm, overlap_percent) -> list:
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
        components = []
        
        # offset кропа внутри полного кадра 1920x1080 (всегда 640 и 220)
        x_off = (self.FRAME_WIDTH - self.CROP_SIZE_PX) // 2
        y_off = (self.FRAME_HEIGHT - self.CROP_SIZE_PX) // 2
        
        for det in self.current_detections:
            box = det['box'] # координаты в глобальных пикселях
            
            area = self.calculate_box_area(box)
            if area < 300:
                print(f'удалён элемент с площадью {area}')
                continue

            # перевод в локальные пиксели кропа (от 0 до 640)
            x1_local = box[0] - x_off
            y1_local = box[1] - y_off
            x2_local = box[2] - x_off
            y2_local = box[3] - y_off
            
            # локальный центр в пикселях
            cx_local = (x1_local + x2_local) / 2
            cy_local = (y1_local + y2_local) / 2
            
            # локальные миллиметры (от 0 до CROP_SIZE_PX)
            cx_crop_mm = (cx_local / self.CROP_SIZE_PX) * self.CROP_SIZE_MM
            cy_crop_mm = (cy_local / self.CROP_SIZE_PX) * self.CROP_SIZE_MM

            # глобальные координаты платы
            global_x = round(cx_crop_mm + plate_y, 4)
            global_y = round(cy_crop_mm + plate_x, 4)
            
            components.append({
                'class_name': self.CLASS_NAMES[det['class']],
                'class_id': det['class'],
                'bbox_crop': [round(box[0], 4), round(box[1], 4), round(box[2], 4), round(box[3], 4)], # оставляем глобальные для отрисовки
                'center_px': [round(cx_local, 4), round(cy_local, 4)],
                'center_mm': [global_x, global_y],
                'confidence': round(det['score'], 4),
                'crop_origin_mm': [plate_x, plate_y] 
            })
            
        return components
    
    # отсечь детекции которые полностью лежат внутри другой
    def filter_nested_boxes(self, detections, area_ratio_threshold=0.5):
        if not detections or len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda x: self.calculate_box_area(x['box']), reverse=True)
        filtered = []

        for current_det in sorted_dets:
            box_c = current_det['box']
            is_nested = False
            
            for accepted_det in filtered:
                box_a = accepted_det['box']
                
                if (box_c[0] >= box_a[0] and box_c[1] >= box_a[1] and box_c[2] <= box_a[2] and box_c[3] <= box_a[3]):
                    
                    area_c = self.calculate_box_area(box_c)
                    area_a = self.calculate_box_area(box_a)
                    
                    if area_c < area_a * area_ratio_threshold:
                        is_nested = True
                        break
            
            if not is_nested:
                filtered.append(current_det)
                
        return filtered
        


    # вычислить площадь элемента
    def calculate_box_area(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height

    # сохранить результаты в файл
    def save_results(self, components, filename=None):

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
    
    # удалить элементы которые встретились повторно (поиск по расстоянию от центров)
    def remove_duplicate_components(self, components, distance_threshold_mm=3.0) -> list:
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
            # items = sorted(items, key=lambda x: x['confidence'], reverse=True) # сортировка по confidence убрана чтобы строго брать первый элемент

            used = set()
            
            for i, comp1 in enumerate(items):
                if i in used:
                    continue
                
                unique.append(comp1)
                used.add(i)

                for j, comp2 in enumerate(items[i+1:], start=i+1):
                    if j in used:
                        continue

                    dx = abs(comp1['center_mm'][0] - comp2['center_mm'][0])
                    dy = abs(comp1['center_mm'][1] - comp2['center_mm'][1])
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance <= distance_threshold_mm:
                        used.add(j)
        
        return unique
        
    # сравнение текущей платы с эталоном
    # match_distance_mm - расстояние между центрами пар, shift_distance_mm - допустимое смещение 
    def compare_with_standard(self, standard_components, current_components, match_distance_mm, shift_distance_mm) -> list:
        defects = []
        
        # множества для отслеживания использованных элементов
        used_standard = set()
        used_current = set()

        print("\n" + "="*50)
        print(f" Эталон: {len(standard_components)} элем. | Текущая: {len(current_components)} элем. ")
        print(f" Порог поиска пары: {match_distance_mm} мм | Порог сдвига: {shift_distance_mm} мм")
        print("="*50)
        
        # поиск всех кандидатов на пару (расстояние < match_distance_mm)
        pairs = []
        for i, s_comp in enumerate(standard_components):
            for j, c_comp in enumerate(current_components):
                dx = abs(s_comp['center_mm'][0] - c_comp['center_mm'][0])
                dy = abs(s_comp['center_mm'][1] - c_comp['center_mm'][1])
                distance = (dx**2 + dy**2)**0.5
                
                if distance <= match_distance_mm:

                    # разница размеров
                    w_s = s_comp['bbox_crop'][2] - s_comp['bbox_crop'][0]
                    h_s = s_comp['bbox_crop'][3] - s_comp['bbox_crop'][1]
                    w_c = c_comp['bbox_crop'][2] - c_comp['bbox_crop'][0]
                    h_c = c_comp['bbox_crop'][3] - c_comp['bbox_crop'][1]

                    pairs.append({
                        's_idx': i, 
                        'c_idx': j, 
                        'distance': distance,
                        'same_class': s_comp['class_id'] == c_comp['class_id'],
                        'size_diff': abs(w_s - w_c) + abs(h_s - h_c) # суммарная разница
                    })

        # сортировка пар по классу, расстоянию от центров, разнице размеров
        pairs.sort(key=lambda x: (not x['same_class'], x['distance'], x['size_diff']))
        
        for pair in pairs:
            s_idx, c_idx = pair['s_idx'], pair['c_idx']
            
            # пропуск элементов которые уже использовались в парах
            if s_idx in used_standard or c_idx in used_current:
                continue
                
            s_comp = standard_components[s_idx]
            c_comp = current_components[c_idx]
            distance = pair['distance']
            
            if pair['same_class']:

                # сравнение по длине и ширине
                box_s = s_comp['bbox_crop']
                box_c = c_comp['bbox_crop']

                w_s = box_s[2] - box_s[0]
                h_s = box_s[3] - box_s[1]
                w_c = box_c[2] - box_c[0]
                h_c = box_c[3] - box_c[1]
                
                delta_w = abs(w_s - w_c)
                delta_h = abs(h_s - h_c)

                SIZE_THRESHOLD_PX = 8.0 # порог допустимого сдвига элемента

                # если классы совпали -> проверка сдвига
                if distance > shift_distance_mm:
                    print(f"[СДВИГ] Эталон: {box_s} Текущий: {box_c}")
                    defects.append({
                        'type': 'SHIFTED',
                        'standard_comp': s_comp,
                        'current_comp': c_comp,
                        'distance': round(distance, 2),
                        'status': 'pending'
                    })
                elif delta_w > SIZE_THRESHOLD_PX or delta_h > SIZE_THRESHOLD_PX:
                    print(f"[РАЗМЕР/ОРИЕНТАЦИЯ] Эталон #{s_idx} ({s_comp['class_name']} {w_s:.0f}x{h_s:.0f}) + Текущий #{c_idx} ({w_c:.0f}x{h_c:.0f}) -> дельта_w={delta_w:.0f}, дельта_h={delta_h:.0f}")
                    defects.append({
                        'type': 'WRONG_SIZE',
                        'standard_comp': s_comp,
                        'current_comp': c_comp,
                        'distance': round(distance, 2),
                        'status': 'pending'
                    })
                else:
                    print(f"[ОК] Эталон #{s_idx} ({s_comp['class_name']} @ {s_comp['center_mm']}) + Текущий #{c_idx} ({c_comp['class_name']} @ {c_comp['center_mm']}) -> dist={distance:.2f} мм")
                # если сдвиг в норме -> дефекта нет
            else:
                print(f"[НЕ ТОТ КЛАСС] Эталон #{s_idx} ({s_comp['class_name']} @ {s_comp['center_mm']}) + Текущий #{c_idx} ({c_comp['class_name']} @ {c_comp['center_mm']}) -> dist={distance:.2f} мм")
                # неверный класс
                defects.append({
                    'type': 'WRONG_CLASS',
                    'standard_comp': s_comp,
                    'current_comp': c_comp,
                    'distance': round(distance, 2),
                    'status': 'pending'
                })
                
            # помечаем элементы как использованные
            used_standard.add(s_idx)
            used_current.add(c_idx)
            
        # поиск отсутствующих элементов (из эталона, которым не нашлось пары)
        for i, s_comp in enumerate(standard_components):
            if i not in used_standard:
                print(f"[ОТСУТСТВУЕТ] Эталон #{i} ({s_comp['class_name']} @ {s_comp['center_mm']}) - НЕ НАШЕЛ ПАРЫ В РАДИУСЕ {match_distance_mm} мм")
                defects.append({
                    'type': 'MISSING',
                    'standard_comp': s_comp,
                    'current_comp': None,
                    'distance': None,
                    'status': 'pending'
                })
                
        # поиск лишних элементов (на текущей плате, которым не нашлось пары)
        for j, c_comp in enumerate(current_components):
            if j not in used_current:
                print(f"[ЛИШНИЙ] Текущий #{j} ({c_comp['class_name']} @ {c_comp['center_mm']}) - НЕТ В ЭТАЛОНЕ В РАДИУСЕ {match_distance_mm} мм")
                defects.append({
                    'type': 'EXTRA',
                    'standard_comp': None,
                    'current_comp': c_comp,
                    'distance': None,
                    'status': 'pending'
                })
                
        return defects
    
    # отрисовка ошибок поверх кадра
    def draw_defect_overlay(self, frame):
        if not self.active_defect:
            return frame
        
        defect = self.active_defect
        
        if defect['type'] == 'MISSING':
            std = defect['standard_comp'] # бокс из эталона
            x1, y1, x2, y2 = [int(v) for v in std['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Отсутствует", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        elif defect['type'] == 'EXTRA':
            cur = defect['current_comp']
            x1, y1, x2, y2 = [int(v) for v in cur['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Лишний", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        elif defect['type'] == 'SHIFTED':
            std = defect['standard_comp']
            cur = defect['current_comp']

            offset_x_mm = std['crop_origin_mm'][0] - cur['crop_origin_mm'][0]  # std_y - cur_y
            offset_y_mm = std['crop_origin_mm'][1] - cur['crop_origin_mm'][1]  # std_x - cur_x
            
            offset_x_px = offset_y_mm * self.PX_PER_MM
            offset_y_px = offset_x_mm * self.PX_PER_MM

            std_box = std['bbox_crop'] # глобальные координаты
            x1 = int(round(std_box[0] + offset_x_px))
            y1 = int(round(std_box[1] + offset_y_px))
            x2 = int(round(std_box[2] + offset_x_px))
            y2 = int(round(std_box[3] + offset_y_px))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Ожидаемый: {std['class_name']}", (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            # Рисуем фактическую позицию
            x1, y1, x2, y2 = [int(v) for v in cur['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Сдвинут: {cur['class_name']}", (x1, y2 + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

        elif defect['type'] == 'WRONG_CLASS':
            std = defect['standard_comp']
            cur = defect['current_comp']
            x1, y1, x2, y2 = [int(v) for v in std['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Ожидаемый: {std['class_name']}", (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            x1, y1, x2, y2 = [int(v) for v in cur['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Найдено: {cur['class_name']}", (x1, y2 + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        elif defect['type'] == 'WRONG_SIZE':
            cur = defect['current_comp']
            x1, y1, x2, y2 = [int(v) for v in cur['bbox_crop']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Неверный размер/поворот", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        return frame

    # ручная корректировка файла эталона при ложном срабатывании, перезапись json файла
    def correct_false_positive(self, defect, standard_components):

        defect_type = defect.get('type')
        std_comp = defect.get('standard_comp')
        cur_comp = defect.get('current_comp')
        
        modified = False
        idx_to_update = -1
        
        # если в дефекте есть эталонный компонент, ищем его индекс в списке
        if std_comp:
            for i, comp in enumerate(standard_components):
                # поиск по точному совпадению координат центра эталона
                if (comp['center_mm'][0] == std_comp['center_mm'][0] and comp['center_mm'][1] == std_comp['center_mm'][1]):
                    idx_to_update = i
                    break
        
        # ЛИШНИЙ -> добавляем текущий элемент в эталон
        if defect_type == 'EXTRA' and cur_comp:
            standard_components.append(cur_comp.copy())
            print(">> [Адаптация] Лишний элемент добавлен в эталон.")
            modified = True
            
        # ОТСУТСТВУЕТ -> удаляем элемент из эталона
        elif defect_type == 'MISSING' and idx_to_update != -1:
            # standard_components.pop(idx_to_update)
            # print(">> [Адаптация] Отсутствующий элемент удален из эталона.")
            modified = True
            
        # СДВИНУТ -> обновляем координаты эталона на текущие
        elif defect_type == 'SHIFTED' and idx_to_update != -1 and cur_comp:
            standard_components[idx_to_update]['center_mm'] = cur_comp['center_mm']
            standard_components[idx_to_update]['center_px'] = cur_comp['center_px'] 
            standard_components[idx_to_update]['bbox_crop'] = cur_comp['bbox_crop']
            standard_components[idx_to_update]['crop_origin_mm'] = cur_comp['crop_origin_mm']
            print(">> [Адаптация] Сдвинутый элемент: координаты в эталоне обновлены.")
            modified = True
            
        # НЕВЕРНЫЙ РАЗМЕР -> обновляем бокс в эталоне на текущий
        elif defect_type == 'WRONG_SIZE' and idx_to_update != -1 and cur_comp:
            standard_components[idx_to_update]['bbox_crop'] = cur_comp['bbox_crop']
            standard_components[idx_to_update]['crop_origin_mm'] = cur_comp['crop_origin_mm']
            print(">> [Адаптация] Неверный размер: бокс в эталоне обновлен.")
            modified = True
            
        # НЕВЕРНЫЙ КЛАСС -> меняем класс в эталоне
        elif defect_type == 'WRONG_CLASS' and idx_to_update != -1 and cur_comp:
            standard_components[idx_to_update]['class_name'] = cur_comp['class_name']
            standard_components[idx_to_update]['class_id'] = cur_comp['class_id']
            print(">> [Адаптация] Неверный класс: класс в эталоне изменен.")
            modified = True
            
        # сохраняем изменения в файл, если были модификации
        if modified:
            self.save_results(standard_components, filename=self.STANDARD_FILENAME)
            return True
        else:
            print(">> [Адаптация] Не удалось применить корректировку (элемент не найден в эталоне).")
            return False


    # отсечь мелкие детали, попавшие внутрь крупных элементов (ИС) и которые ложно детектируются
    def filter_components_inside_ic(self, components):
        if not components:
            return components
            
        ics = [c for c in components if c['class_name'] == 'ic']
        others = [c for c in components if c['class_name'] != 'ic']
        
        if not ics:
            return components
            
        filtered_others = []
        
        for other in others:
            is_inside_ic = False
            ox, oy = other['center_mm']
            
            for ic in ics:
                box = ic['bbox_crop']
                w_px = box[2] - box[0]
                h_px = box[3] - box[1]
                
                w_mm = w_px / self.PX_PER_MM
                h_mm = h_px / self.PX_PER_MM
                
                # берем половину размера
                half_w = w_mm / 2.0
                half_h = h_mm / 2.0
                
                min_x = ic['center_mm'][0] - half_w
                max_x = ic['center_mm'][0] + half_w
                
                min_y = ic['center_mm'][1] - half_h
                max_y = ic['center_mm'][1] + half_h
                
                if min_x <= ox <= max_x and min_y <= oy <= max_y:
                    is_inside_ic = True
                    print(f"[ФИЛЬТР ИС] Отсечен {other['class_name']} @ {ox:.2f},{oy:.2f} - внутри ИС @ {ic['center_mm']}")
                    print(f"   -> Размер рамки ИС: {w_mm:.1f}x{h_mm:.1f} мм. Границы: X[{min_x:.1f}..{max_x:.1f}], Y[{min_y:.1f}..{max_y:.1f}]")
                    break
                    
            if not is_inside_ic:
                filtered_others.append(other)
                
        return ics + filtered_others
            


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
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("Система остановлена.")