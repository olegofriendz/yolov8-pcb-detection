import numpy as np
import cv2
from rknn.api import RKNN
from pathlib import Path


class RKNNdetect:
    def __init__(self, model_path=None, img_size=640, conf_thres=0.6, nms_thres=0.4, num_classes=5, class_names=None):
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent # корень
            model_path = project_root / "runs" / "detect" / "one-board-dataset" / "weights" / "best.rknn"

        self.model_path = str(model_path) # путь к best.rknn
        self.img_size = img_size # размер изображения
        self.conf_thres = conf_thres # уверенность
        self.nms_thres = nms_thres # близкие объекты
        self.num_classes = num_classes
        self.class_names = class_names
        self.history = []
        self.stable_frames = 3


    def load_rknn_model(self, target_platform='rk3588'):
        self.rknn = RKNN(verbose=False)
        print(f"Загрузка RKNN модели: {self.model_path}.")

        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"❌ Ошибка загрузки .rknn (код {ret})")

        print(f"Инициализация NPU runtime...")
        
        ret = self.rknn.init_runtime(target=target_platform)
        if ret != 0:
            raise RuntimeError(f"❌ Ошибка инициализации NPU (код {ret})")
        
        print(f"Модель {self.model_path} загружена.")
        

    # crop_frame - участок 640 на 640 в исходном изображении
    def preprocess(self, crop_frame):
        img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0 # uint8 -> float32
        img_input = np.transpose(img_normalized, (2, 0, 1)) # перестановка осей
        img_input = np.expand_dims(img_input, axis=0) # batch
        return img_input


    # сырой выход нейросети -> список детекций в координатах исходного кадра
    def postprocess(self, outputs, x_off, y_off, orig_shape):
        output = outputs[0][0].transpose(1, 0) # транспонирование тензора в необходимый формат
        boxes = output[:, :4]
        scores = 1 / (1 + np.exp(-output[:, 4:4+self.num_classes])) # логиты -> вероятность [0..1]
        class_ids = np.argmax(scores, axis=1)
        class_scores = np.max(scores, axis=1)
        
        mask = class_scores >= self.conf_thres
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        class_scores = class_scores[mask]
        
        if len(boxes) == 0:
            return []
        
        # [cx, cy, w, h] -> x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        detections = []
        for i in range(len(boxes)):
            
            CROP_MARGIN = 1 # отступ от края
            if (x1[i] <= CROP_MARGIN or
                y1[i] <= CROP_MARGIN or
                x2[i] >= self.img_size - CROP_MARGIN or
                y2[i] >= self.img_size - CROP_MARGIN):
                continue # отбрасываются объекты на краю кропа
            
            box_global = np.array([x1[i] + x_off, y1[i] + y_off, x2[i] + x_off, y2[i] + y_off]) # возвращаемся к координатам исходного кадра добавляя смещение
            box_global[[0, 2]] = np.clip(box_global[[0, 2]], 0, orig_shape[1]) # orig_shape - размера исходного кадра
            box_global[[1, 3]] = np.clip(box_global[[1, 3]], 0, orig_shape[0])
            
            detections.append({
                'box': box_global,              # [x1, y1, x2, y2]
                'class': int(class_ids[i]),     # номер класса
                'score': float(class_scores[i]) # уверенность
            })
        
        if len(detections) > 1:
            detections = self._filter_nested_boxes(detections) # фильтр вложенных боксов
        
        return detections
    

    # фильтр вложенных боксов (если бокс полностью лежит внутри другого бокса -> отбрасываем)
    def _filter_nested_boxes(self, detections):
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
        return keep


    def stabilize(self, detections):
        new_history = []
        for det in detections:
            cx = (det['box'][0] + det['box'][2]) / 2
            cy = (det['box'][1] + det['box'][3]) / 2
            matched = False
            for h in self.history:
                if h['class'] == det['class'] and abs(h['cx'] - cx) < 25 and abs(h['cy'] - cy) < 25:
                    h['count'] += 1
                    h['cx'], h['cy'] = cx, cy
                    h['box'] = det['box']
                    h['score'] = det['score']
                    new_history.append(h)
                    matched = True
                    break
            if not matched:
                new_history.append({'class': det['class'], 'cx': cx, 'cy': cy, 'count': 1,
                                    'box': det['box'], 'score': det['score']})
        
        self.history = new_history
        stable = [h for h in self.history if h['count'] >= self.stable_frames] # объекты, накопившие достаточно кадров stab;e_frames
        return [{'box': h['box'], 'class': h['class'], 'score': h['score']} for h in stable] # формат детекций
    

    def detect(self, frame, x_off=None, y_off=None):
        h, w = frame.shape[:2]
        
        if x_off is None:
            x_off = (w - self.img_size) // 2
        if y_off is None:
            y_off = (h - self.img_size) // 2
        
        crop_frame = frame[y_off:y_off+self.img_size, x_off:x_off+self.img_size]
        img_input = self.preprocess(crop_frame)
        outputs = self.rknn.inference(inputs=[img_input], data_format='nchw')
        detections = self.postprocess(outputs, x_off, y_off, (h, w))
        stable_detections = self.stabilize(detections)
        
        return stable_detections, crop_frame, (x_off, y_off)
    

    def release(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()



