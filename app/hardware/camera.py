import cv2
import threading


class CameraCapture:
    def __init__(self, camera_id, width, height):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 600)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
    
    def _capture(self):
        while self.running:
            ret, frame = self.cap.read() # получить кадр
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None
    
    # закрыть камеру
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()