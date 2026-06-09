import tkinter as tk
import cv2
import time
import json
from threading import Thread
from core.inspector import Inspector
from hardware.motion import MotionContoller
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path

class MainWindow:
    def __init__(self, inspector):
        self.inspector = inspector
        self.mode = None  # 'manual', 'standard' или 'plate'
        self.defects_list = []
        self.current_defect_idx = -1 # на какой ошибке произошла остановка

        self.root = tk.Tk()
        self.root.title("PCB Inspector")
        self.root.configure(bg='#f0f0f0')

        # === верхняя панель ===
        top_panel = tk.Frame(self.root, bg='#ffffff', relief='raised', bd=1)
        top_panel.pack(fill='x', padx=10, pady=(10, 5))

        # Левая часть: кнопки режимов
        modes_frame = tk.LabelFrame(top_panel, text=" Режим работы ", bg='#ffffff', font=("Arial", 10, "bold"), padx=10, pady=8)
        modes_frame.pack(side=tk.LEFT, padx=(10, 5), pady=5, fill='y')

        self.btn_standard = tk.Button(modes_frame, text='Эталон', command=self.start_scan_standard, width=8, height=2, bg='#e3f2fd', font=("Arial", 10, "bold"))
        self.btn_plate = tk.Button(modes_frame, text='Проверить плату', command=self.start_scan_plate, width=16, height=2, bg='#e3f2fd', font=("Arial", 10, "bold"))
        self.btn_manual = tk.Button(modes_frame, text='🛠️', command=self.start_manual, width=3, height=2, bg='#e3f2fd', font=("Arial", 10, "bold"))
        self.btn_go_home = tk.Button(modes_frame, text='🏠',command=self.inspector.motion.home, width=3, height=2, bg='#e3f2fd', font=("Arial", 10, "bold"))
        self.btn_stop = tk.Button(modes_frame, text='⚠️', command=self.stop_scanning, width=3, height=2, bg='#ff3333', fg='white', font=("Arial", 10, "bold"))
        
        self.btn_standard.pack(side=tk.LEFT, padx=2)
        self.btn_plate.pack(side=tk.LEFT, padx=2)
        self.btn_manual.pack(side=tk.LEFT, padx=2)
        self.btn_go_home.pack(side=tk.LEFT, padx=2)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # Центральная часть: инструменты
        tools_frame = tk.LabelFrame(top_panel, text=" Инструменты ", bg='#ffffff', font=("Arial", 10, "bold"), padx=10, pady=8)
        tools_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='y')

        self.btn_next_defect = tk.Button(tools_frame, command=self.show_next_defect, text='Показать дефекты', width=18, height=2, bg='#ff9800', fg='white', font=("Arial", 10, "bold"))
        self.btn_next_defect.pack(side=tk.LEFT, padx=3)

        # Разделитель
        tk.Frame(tools_frame, bg='#e0e0e0', width=2, height=35).pack(side=tk.LEFT, padx=8, fill='y')

        self.btn_1 = tk.Button(tools_frame, command=self.resolve_defect, text='Ложное срабатывание', width=20, height=2, bg='#f44336', fg='white', font=("Arial", 10, "bold"))
        self.btn_2 = tk.Button(tools_frame, command=self.resolve_defect, text='Дефект исправлен', width=16, height=2, bg='#228c22', fg='white', font=("Arial", 10, "bold"))
        
        self.btn_1.pack(side=tk.LEFT, padx=2)
        self.btn_2.pack(side=tk.LEFT, padx=2)

        # Параметры платы
        settings_frame = tk.LabelFrame(top_panel, text=" Параметры платы ", bg='#ffffff', font=("Arial", 10, "bold"), padx=10, pady=8)
        settings_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='y')

        size_row = tk.Frame(settings_frame, bg='#ffffff')
        size_row.pack(fill='x', expand=True)

        tk.Label(size_row, text="Высота:", bg='#ffffff', font=("Arial", 10)).pack(side=tk.LEFT)
        self.plate_width_entry = tk.Entry(size_row, width=6, font=("Arial", 10), justify='center')
        self.plate_width_entry.pack(side=tk.LEFT, padx=3)
        self.plate_width_entry.insert(0, str(self.inspector.PLATE_WIDTH))
        
        tk.Label(size_row, text="Ширина:", bg='#ffffff', font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))
        self.plate_height_entry = tk.Entry(size_row, width=6, font=("Arial", 10), justify='center')
        self.plate_height_entry.pack(side=tk.LEFT, padx=3)
        self.plate_height_entry.insert(0, str(self.inspector.PLATE_HEIGHT))
        
        tk.Label(size_row, text="мм", bg='#ffffff', font=("Arial", 9)).pack(side=tk.LEFT)
        
        tk.Button(size_row, text="Применить", command=self.apply_plate_size, bg='#4caf50', fg='white', font=("Arial", 10, "bold"),
                  height=1).pack(side=tk.LEFT, padx=(10, 0))

        # Статус
        status_frame = tk.Frame(top_panel, bg='#ffffff')
        status_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        self.status = tk.Label(status_frame, text="● Готов", fg="#4caf50", font=("Arial", 12, "bold"), bg='#ffffff', anchor='center')
        self.status.pack(expand=True)

        # видео
        video_frame = tk.Frame(self.root, bg='#000000', relief='sunken', bd=2)
        video_frame.pack(padx=10, pady=(5, 10), fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        self.update_display()

    # ручное управление
    def start_manual(self):
        if self.mode == "manual":
            self.stop_manual()
        else:
            self.inspector.active_defect = None         
            self.mode = 'manual'
            self.inspector.hide_detections = False
            self.btn_manual.config(text="🛠️", relief=tk.SUNKEN)
            self.btn_standard.config(state=tk.DISABLED)
            self.btn_plate.config(state=tk.DISABLED)
            self.status.config(text="Ручное управление", fg="blue")
            self.inspector.motion.send("G91")

    def stop_manual(self):
        self.mode = None
        self.btn_manual.config(text="🛠️", relief=tk.RAISED)
        self.btn_standard.config(state=tk.NORMAL)
        self.btn_plate.config(state=tk.NORMAL)
        self.status.config(text="Ручное управление отключено", fg="green")
        self.inspector.motion.send("G90")

    # сканировать эталон
    def start_scan_standard(self):
        if self.mode == "standard":
            self.stop_scan_standard()
        else:
            self.inspector.active_defect = None
            self.mode = 'standard'
            self.inspector.hide_detections = False
            self.btn_standard.config(text="Эталон", relief=tk.SUNKEN)
            self.btn_manual.config(state=tk.DISABLED)
            self.btn_plate.config(state=tk.DISABLED)
            self.status.config(text="Сканирование эталона...", fg="orange")
            
            def run_scan():
                components = self.inspector.scan_plate()
                self.inspector.save_results(components, filename="standard_plate.json")
                self.root.after(0, self.stop_scan_standard) # ссылка на функцию
            
            Thread(target=run_scan, daemon=True).start() # отдельный поток 
    
    def stop_scan_standard(self):
        self.mode = None
        self.btn_standard.config(text="Эталон", relief=tk.RAISED)
        self.btn_manual.config(state=tk.NORMAL)
        self.btn_plate.config(state=tk.NORMAL)
        self.status.config(text="Сканирование эталона завершено", fg="green")

    # сканировать плату
    def start_scan_plate(self):
        if self.mode == "plate":
            self.stop_scan_plate()
        else:
            self.inspector.active_defect = None
            self.mode = 'plate'
            self.inspector.hide_detections = False
            self.btn_plate.config(text="Проверить плату", relief=tk.SUNKEN)
            self.btn_manual.config(state=tk.DISABLED)
            self.btn_standard.config(state=tk.DISABLED)
            self.status.config(text="Проверка платы...", fg="orange")

            def run_scan():
                try:
                    current_components = self.inspector.scan_plate()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.inspector.save_results(current_components, filename=f"inspection_{timestamp}.json")

                    # сравнение с эталоном
                    standard_components = self.load_standard_components() # получить компоненты эталона из файла

                    self.defects_list = self.inspector.compare_with_standard(
                        standard_components=standard_components,
                        current_components=current_components,
                        match_distance_mm=4.0, # расстояние между центрами пар для поиска
                        shift_distance_mm=2.0  # допустимое смещение
                    )

                    self.defects_list.sort(key=lambda d: (d.get('current_comp') or d.get('standard_comp'))['crop_origin_mm']) # сортировка дефектов по кропам

                # не создан эталон
                except FileNotFoundError as e:
                    self.defects_list = []
                    self.root.after(0, lambda err=e: self.stop_scan_plate(error=str(e)))
                    return
                except Exception as e:
                    self.defects_list = []
                    self.root.after(0, lambda err=e: self.stop_scan_plate(error=f"Ошибка {e}"))
                    return

                self.root.after(0, lambda: self.stop_scan_plate(defects=self.defects_list))

            Thread(target=run_scan, daemon=True).start()

    def stop_scan_plate(self, defects=None, error=None):
        self.mode = None
        self.btn_plate.config(text="Проверить плату", relief=tk.RAISED)
        self.btn_manual.config(state=tk.NORMAL)
        self.btn_standard.config(state=tk.NORMAL)
        self.status.config(text="Проверка платы завершена", fg="green")

        if error:
            self.status.config(text=error, fg="red")
            self.btn_next_defect.config(state=tk.DISABLED)
            self.inspector.hide_detections = False
        elif defects is not None:
            if len(defects) == 0:
                self.status.config(text="Ошибок не найдено!", fg="green")
                self.btn_next_defect.config(state=tk.DISABLED)
                self.inspector.hide_detections = False
            else:
                self.status.config(text=f"Найдено дефектов: {len(defects)}", fg="red")
                self.btn_next_defect.config(state=tk.NORMAL, text='Показать дефекты')
                self.current_defect_idx = -1
                self.inspector.hide_detections = False # включаем режим скрытия контуров элементов если в режиме ошибок
        else:
            self.status.config(text="Проверка платы прервана", fg="orange")
            self.btn_next_defect.config(state=tk.DISABLED)
            self.inspector.hide_detections = False


    # кадр OpenCV -> изображение Tkinter
    def convert_frame_for_tkinter(self, opencv_frame):
        rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        return tk_image
    
    # постоянное обновление изображения из inspector.current_frame
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

    def apply_plate_size(self):
        try:
            width = float(self.plate_width_entry.get())
            height = float(self.plate_height_entry.get())

            if width <= 0 or height <= 0:
                raise ValueError("Размеры платы не могут быть отрицательными!")
            
            self.inspector.PLATE_WIDTH = width
            self.inspector.PLATE_HEIGHT = height
            self.status.config(text=f"Размер платы: {width} x {height} мм", fg="green")

        except ValueError as e:
            self.status.config(text=f"Ошибка: неверный ввод!", fg="red")

    # прочитать файл эталона
    def load_standard_components(self):
        filepath = Path(__file__).parent.parent / "results" / self.inspector.STANDARD_FILENAME

        if not filepath.exists():
            raise FileNotFoundError(f"Файл эталона не найден: {filepath}.")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data['components']
    
    # перейти к следующему найденному дефекту
    def show_next_defect(self):
        if not self.defects_list:
            self.status.config(text="Список дефектов пуст", fg="green")
            return

        start_search_idx = (self.current_defect_idx + 1) % len(self.defects_list)
        found_idx = -1

        for i in range(len(self.defects_list)):
            check_idx = (start_search_idx + i) % len(self.defects_list)
            defect = self.defects_list[check_idx]

            if defect.get('status', 'pending') != 'resolved':
                found_idx = check_idx
                break
            
        # ошибок нет
        if found_idx == -1:
            self.status.config(text='Все ошибки устранены!', fg='green')
            self.inspector.active_defect = None
            self.btn_next_defect.config(state=tk.DISABLED)
            return

        self.current_defect_idx = found_idx
        defect = self.defects_list[self.current_defect_idx]
        self.btn_next_defect.config(text='Следующий дефект')
        self.inspector.hide_detections = True
        
        target_comp = defect.get('current_comp') or defect.get('standard_comp') # missing - эталон, другие - плата
        crop_x, crop_y = target_comp['crop_origin_mm']

        display_num = self.current_defect_idx + 1 # индекс для отображения в статусе
        total_defects = len(self.defects_list)

        self.btn_next_defect.config(state=tk.DISABLED) # кнопка заблокирована во время движения
        self.status.config(text=f"Двигаюсь к ошибке {display_num}/{total_defects}...", fg="orange")

        def move_and_show():
            self.inspector.active_defect = None
            self.inspector.motion.move_absolute(crop_x, crop_y, feedrate=2000)
            self.inspector.motion.wait_for_stop()
            time.sleep(0.5)

            self.inspector.active_defect = defect
            
            # обновить статус
            defect_type_translate = {
                'MISSING': 'Отсутствует',
                'EXTRA': 'Лишний',
                'WRONG_SIZE': 'Размер не соответствует/Поворот',
                'WRONG_CLASS': 'Класс не соответствует',
                'SHIFTED': 'Сдвинут'
            }
            type_str = defect_type_translate.get(defect['type'], defect['type'])

            self.root.after(0, lambda: self.btn_next_defect.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status.config(text=f"Ошибка {display_num}/{total_defects}: {type_str} ({target_comp['class_name']})", fg="red"))

        Thread(target=move_and_show, daemon=True).start()

    # пометить дефект в списке чтобы не возвращаться к нему
    def resolve_defect(self):
        if self.current_defect_idx < 0 or self.current_defect_idx >= len(self.defects_list):
            return
        
        self.defects_list[self.current_defect_idx]['status'] = 'resolved' # пометка что ошибка обработана
        self.inspector.active_defect = None
        self.show_next_defect()

    def stop_scanning(self):
        if self.mode in ['standard', 'plate']:
            self.inspector.scan_aborted = True
            self.status.config(text='Остановка сканирования...', fg='orange')
            self.btn_plate.config(state=tk.NORMAL)
            self.btn_standard.config(state=tk.NORMAL)
    
    def run(self):
        self.root.mainloop()