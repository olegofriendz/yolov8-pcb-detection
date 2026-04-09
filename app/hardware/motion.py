import serial
import time


class MotionContoller:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(2)

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.send("G90")
            self.ser.close()

    # отправить g-код и получить ответ
    def send(self, cmd: str) -> str:
        self.ser.write(f"{cmd}\n".encode())
        responce = ""
        while True:
            line = self.ser.readline().decode().strip()
            if not line:
                continue
            response = line
            if line == 'ok' or line.startswith('error'):
                break
        return response
    
    def move_relative(self, x=0, y=0, feedrate=1000) -> str:
        self.send("G91") # относительные координаты
        responce = self.send(f"G1 X{x} Y{y} F{feedrate}")
        return responce
    
    def move_absolute(self, x, y, feedrate=1000) -> str:
        self.send("G90")
        responce = self.send(f"G1 X{x} Y{y} F{feedrate}")
        return responce
    
    def home(self):
        self.send("G90")
        self.send("G1 X0 Y0 F1000")