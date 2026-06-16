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
            self.ser.write(b"~\n")
            
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
        response = ""
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            response = line
            if line == 'ok' or line.startswith('error'):
                break
        return response
    
    def move_relative(self, x=0, y=0, feedrate=1000) -> str:
        if not self.connected:
            return ""

        responce = self.send(f"G91 G1 X{x} Y{y} F{feedrate}")
        return responce
    
    def move_absolute(self, x, y, feedrate=1000) -> str:
        if not self.connected:
            return ""

        responce = self.send(f"G90 G1 X{x} Y{y} F{feedrate}")
        return responce
    
    def wait_for_stop(self):
        time.sleep(0.15) # проверка на ложный Idle

        while True:
            self.ser.write(b"?")
            time.sleep(0.05)

            status = None
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                if line.startswith('<') and line.endswith('>'):
                    status = line

            if status:
                if "Idle" in status:
                    break
                elif "Run" in status or "Home" in status:
                    pass
                elif "Alarm" in status:
                    raise Exception(f"Аварийная остановка: {status}")
                
            time.sleep(0.1)


    # вернуться в созданный ноль
    def home(self):
        if not self.connected:
            print("Управление недоступно!")
            return ""

        self.send("G90")
        self.send("G1 X0 Y0 F2000")

    # создать ноль
    def set_home(self):
        self.send("G92 X0 Y0")

    # вернуться в системные нули (замыкание концевиков)
    def go_zero(self):
        self.send("$H")