<a href="./README.md">English</a> &nbsp;|&nbsp;
<a href="./README.ru.md"><b>Русский</b></a>

# YOLOv8 PCB Detection

Детектирование и классификация электронных компонентов на печатных платах в реальном времени с помощью YOLOv8, оптимизировано для ARM64-устройств (Raspberry Pi, Orange Pi) с ускорением RKNN.

## Стек технологий

- **Фреймворк:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Язык:** Python 3.10
- **Глубокое обучение:** PyTorch
- **Edge-ускорение:** RKNN Toolkit 2 (для инференса на NPU Rockchip SoC)
- **Контейнеризация:** Docker & Docker Compose

## Результаты обучения & Интерфейс

### Метрики
![Результаты обучения](runs/detect/one-board-dataset/results.png)

### Интерфейс
![Интерфейс](assets/gui-demo.png)

## Быстрый старт

Требования: [Docker](https://docs.docker.com/get-docker/) и [Docker Compose](https://docs.docker.com/compose/install/).

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/olegofriendz/yolov8-pcb-detection.git
   cd yolov8-pcb-detection
   ```

2. Запустите приложение:
   ```bash
   ./run.sh
   # Или вручную:
   # docker compose up --build
   ```
Контейнер запустится, смонтирует локальный код для горячей перезагрузки и откроет интерфейс детектирования.

## Ручная установка

> **Примечание:** Для Orange Pi требуется Python 3.10 для совместимости с предсобранными ARM64-зависимостями.

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/olegofriendz/yolov8-pcb-detection.git
   cd yolov8-pcb-detection
   ```

2. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. **Установки пакеты зависимостей (RKNN Toolkit):**
   ```bash
   pip install requirements/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```

4. **Установите оставшиеся пакеты и переведите проект в режим редактирования:**
   ```bash
   pip install -e .
   ```

5. **Запуск:**
   ```bash
   python app/main.py
   ```

## Процесс конвертации моделей

Для развёртывания моделей на NPU выполните следующую цепочку преобразований:

1. **Скачайте датасет**: Для установки датасета из Roboflow используйте команды `pcb-download` или `python src/data/download.py`.
2. **Обрезка датасета**: Для приведения датасета к нужному разрешению используйте команды `pcb-tile` или `python src/data/tile.py` (по умолчанию: 640px, перекрытие 20%):
   
   ```bash
   pcb-tile --size 640 --overlap 0.2
   ```
4. **Обучение**: Используйте `pcb-train` или `python src/training/train.py`.
5. **Конвертация в формат ONNX**:
   
   ```bash
   python app/utils/converters/pt_to_onnx.py --pt runs/detect/train/weights/best.pt
   ```
7. **Конвертация в формат RKNN**:
   
   ```bash
   python app/utils/converters/onnx_to_rknn.py --onnx runs/detect/train/weights/best.onnx
   ```

## Конфигурация

Переменные окружения можно изменить в `docker-compose.yml`:
- `DISPLAY`: Для проброса X11 (поддержка GUI).
- Проброс устройств (`/dev/video0`, `/dev/dri`) обеспечивает доступ к камере и NPU/GPU.
