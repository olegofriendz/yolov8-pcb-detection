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
        self.stable_frames = 1


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
        return np.ascontiguousarray(img_input)


    def postprocess(self, outputs, x_off, y_off, orig_shape):
        output = outputs[0][0].transpose(1, 0) # транспонирование тензора в необходимый формат
        boxes = output[:, :4]
        scores = output[:, 4:4+self.num_classes] # логиты -> вероятность [0..1]
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
        
        MARGIN = 5 # отступ от края для удаления объектов
        final_detections = []
        
        for cls in range(self.num_classes):
            cls_mask = class_ids == cls
            if not np.any(cls_mask):
                continue
            
            cls_x1 = x1[cls_mask]
            cls_y1 = y1[cls_mask]
            cls_x2 = x2[cls_mask]
            cls_y2 = y2[cls_mask]
            cls_scores = class_scores[cls_mask]
            
            # отбрасываем элементы на краях
            valid_mask = (cls_x1 > MARGIN) & (cls_y1 > MARGIN) & \
                         (cls_x2 < self.img_size - MARGIN) & (cls_y2 < self.img_size - MARGIN)
            
            if not np.any(valid_mask):
                continue
            
            cls_x1 = cls_x1[valid_mask]
            cls_y1 = cls_y1[valid_mask]
            cls_x2 = cls_x2[valid_mask]
            cls_y2 = cls_y2[valid_mask]
            cls_scores = cls_scores[valid_mask]
            
            xywh = np.stack([
                cls_x1,
                cls_y1,
                cls_x2 - cls_x1,
                cls_y2 - cls_y1
            ], axis=1) # формат [x, y, w, h] для cv2.dnn.NMSBoxes
       
            indices = cv2.dnn.NMSBoxes(xywh.tolist(), cls_scores.tolist(), score_threshold=self.conf_thres, nms_threshold=self.nms_thres) # NMS
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    box_global = np.array([
                        cls_x1[idx] + x_off,
                        cls_y1[idx] + y_off,
                        cls_x2[idx] + x_off,
                        cls_y2[idx] + y_off
                    ])
                    box_global[[0, 2]] = np.clip(box_global[[0, 2]], 0, orig_shape[1])
                    box_global[[1, 3]] = np.clip(box_global[[1, 3]], 0, orig_shape[0])
                    
                    final_detections.append({
                        'box': box_global,
                        'class': cls,
                        'score': float(cls_scores[idx])
                    })
        
        return final_detections

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

        return detections, crop_frame, (x_off, y_off)
    

    def release(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()