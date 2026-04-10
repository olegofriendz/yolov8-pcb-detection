import serial
import time


class MotionContoller:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False


    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            print(f"Контроллер {self.port} подключен.")
            self.connected = True
        except:
            print(f"Контроллер {self.port} не найден. Управление отключено.")
            self.ser = None
            self.connected = False


    def disconnect(self):
        if self.ser and self.ser.is_open and self.connected:
            self.send("G90")
            self.ser.close()


    # отправить g-код и получить ответ
    def send(self, cmd: str) -> str:
        if not self.connected:
            return ""

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
        if not self.connected:
            return ""

        self.send("G91") # относительные координаты
        responce = self.send(f"G1 X{x} Y{y} F{feedrate}")
        return responce
    

    def move_absolute(self, x, y, feedrate=1000) -> str:
        if not self.connected:
            return ""

        self.send("G90")
        responce = self.send(f"G1 X{x} Y{y} F{feedrate}")
        return responce
    

    def home(self):
        if not self.connected:
            print("Управление недоступно!")
            return ""

        self.send("G90")
        self.send("G1 X0 Y0 F1000")

    def set_home(self):
        self.send("G92 X0 Y0")