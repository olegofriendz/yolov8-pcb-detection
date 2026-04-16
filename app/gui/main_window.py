import tkinter as tk
import cv2
import time
from threading import Thread
from core.inspector import Inspector
from hardware.motion import MotionContoller
from PIL import Image, ImageTk

class MainWindow:
    def __init__(self, inspector):
        self.inspector = inspector
        self.mode = None # 'manual' или 'auto'

        self.root = tk.Tk()
        self.root.title("PCB Inspector")
        self.root.geometry("1920x1080")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.btn_manual = tk.Button(btn_frame, text='Ручной', command=self.start_manual)
        self.btn_auto = tk.Button(btn_frame, text='Авто', command=self.start_auto)
        self.btn_manual.pack(side=tk.LEFT, padx=10)
        self.btn_auto.pack(side=tk.LEFT, padx=10)
        

        self.status = tk.Label(self.root, text="Готов", fg="green", font=("Arial", 12))
        self.status.pack(pady=5)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()

        self.update_display()

        # *** добавить поля для введения размера платы ***
        # *** добавить кнопку для проверки эталона ***

    def start_manual(self):
        self.mode = 'manual'
        self.btn_manual.config(state=tk.DISABLED)
        self.btn_auto.config(state=tk.NORMAL)
        self.status.config(text="Ручной режим (Q - выход)", fg="blue")
        self.inspector.motion.send("G91")
        # self.inspector.process_frame()  # первый кадр

    def stop_manual(self):
        self.mode = None
        self.btn_manual.config(state=tk.NORMAL)
        self.status.config(text="Ручной режим завершён", fg="green")
        self.inspector.motion.send("G90")

    def start_auto(self):
        self.mode = 'auto'
        self.btn_manual.config(state=tk.NORMAL)
        self.btn_auto.config(state=tk.DISABLED)
        self.status.config(text="Автоматическое сканирование...", fg="orange")
        
        def run_scan():
            self.inspector.scan_plate()
            self.root.after(0, self.auto_finished)
        
        Thread(target=run_scan, daemon=True).start() # отдельный поток 
    
    def auto_finished(self):
        self.mode = None
        self.btn_auto.config(state=tk.NORMAL)
        self.status.config(text="Сканирование завершено", fg="green")

    # кадр OpenCV -> изображение Tkinter
    def convert_frame_for_tkinter(self, opencv_frame):
        rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        return tk_image
    
    # постоянно обновление изображения из inspector.current_frame
    def update_display(self):
        frame = self.inspector.get_current_frame()
        if frame is not None:
            try:
                tk_image = self.convert_frame_for_tkinter(frame)
                self.video_label.imgtk = tk_image
                self.video_label.configure(image=tk_image)
            except Exception as e:
                print(f"Ошибка отображения кадра: {e}")

        self.root.after(30, self.update_display)

    # обработка нажатий клавиш
    def on_key_press(self, event):
        if self.mode != 'manual':
            return
        
        key = event.keysym.lower()
        
        if key == 'q':
            self.stop_manual()
        
        elif key == 'up':
            self.inspector.manual_control_step('up')
        elif key == 'down':
            self.inspector.manual_control_step('down')
        elif key == 'left':
            self.inspector.manual_control_step('left')
        elif key == 'right':
            self.inspector.manual_control_step('right')
        elif key == 'h':
            self.inspector.manual_control_step('home')
        elif key == 'z':
            self.inspector.manual_control_step('set_home')
        elif key == 's':
            self.show_components()

    
    def run(self):
        self.root.mainloop()