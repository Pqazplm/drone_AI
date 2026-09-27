# drone_AI

Обнаружение объектов на Raspberry Pi Zero 2W (Picamera2 + YOLOv8 INT8 TFLite)
с наложенными bounding boxes, которые выводятся в framebuffer и передаются
на композитный (PAL) видеовыход, маршрутизируемый в SpeedyBee F4 / аналоговый VTX.

**ОПЦИОНАЛЬНО**: Визуальное наведение по маркеру на Betaflight через USB MSP.

---

## Структура проекта

```
drone_AI/
├── drone_ai/                 # Основной пакет
│   ├── __init__.py
│   ├── __main__.py           # Точка входа: python -m drone_ai
│   ├── cli.py                # Разбор CLI и главный цикл
│   ├── detector.py           # Загрузка модели, инференс, NMS, постобработка
│   ├── tracker.py            # SimpleIoUTracker (устранение мерцания)
│   ├── drawing.py            # Отрисовка bounding boxes
│   ├── framebuffer.py        # Запись в PAL composite framebuffer (/dev/fb0)
│   ├── camera.py             # Обёртка над Picamera2
│   └── guidance/             # Подмодуль наведения по маркеру
│       ├── __init__.py
│       ├── config.py         # GuidanceConfig, Observation dataclasses
│       ├── msp.py            # MSPv1 транспорт по serial
│       ├── worker.py         # Guidance worker thread
│       └── logging_setup.py  # Настройка логгера guidance
├── config/                   # Модель и лейблы
│   ├── yolov8n_int8.tflite   # Модель YOLOv8n INT8 TFLite
│   └── labels.txt            # Названия классов (по одному в строке)
├── logs/                     # Логи работы
│   ├── detector.log          # Основной пайплайн детектора (модель, FPS, треки)
│   └── marker_guidance.log   # Модуль наведения (MSP, AUX1, машина состояний)
├── requirements.txt
└── README.md
```

## Требования к оборудованию

- **Raspberry Pi Zero 2W** (или совместимая плата)
- **Picamera**, подключённая к CSI-порту
- **Композитный (PAL) выход** на тестовых контактах TVDAC+/TVDAC- для маршрутизации видео
- **Serial-подключение** к полётному контроллеру Betaflight (для режима наведения)

## Установка

### 1. Установка системных зависимостей

```bash
sudo apt update
sudo apt install python3-numpy python3-picamera2 python3-serial
```

### 2. Установка Python-зависимостей

```bash
pip install -r requirements.txt
```

Или вручную:

```bash
# Рекомендуемый: ai-edge-litert от Google
pip install ai-edge-litert

# Альтернатива: tflite_runtime
# pip install tflite-runtime
```

### 3. Настройка композитного (PAL) вывода

Отредактируйте `/boot/firmware/config.txt` (или `/boot/config.txt` на старых образах):

```
enable_tvout=1
sdtv_mode=2          # 2 = PAL, 0 = NTSC
sdtv_aspect=1        # 4:3 (используйте 3 для 16:9)
```

### 4. Размещение модели и лейблов

Скопируйте вашу модель YOLOv8 INT8 TFLite и файл лейблов в `config/`:

```
config/yolov8n_int8.tflite   # Ваша модель
config/labels.txt            # Названия классов, по одному в строке
```

## Использование

### Только детектор (без наведения)

```bash
python -m drone_ai \
    --model config/yolov8n_int8.tflite \
    --labels config/labels.txt \
    --classes 0,2 \
    --conf-thresh 0.45 \
    --iou-thresh 0.60
```

### С наведением по маркеру 

```bash
    python3 -m drone_ai \
    --guidance \
    --guidance-port /dev/ttyACM0 \
    --guidance-marker-class 0 \
    --guidance-live \
    --guidance-i-verified-primary-override \
    --guidance-raw-order AETR \
    --guidance-hover-throttle 1350 \
    --guidance-throttle-min 1250 \
    --guidance-throttle-max 1450 \
    --guidance-desired-box-height 0.25 \
    --guidance-roll-sign 1 --guidance-pitch-sign 1 \
    --guidance-yaw-sign 1 --guidance-throttle-sign -1
```

### Полный список опций командной строки

```
python -m drone_ai --help
```

Основные опции:

| Опция | По умолчанию | Описание |
|-------|-------------|----------|
| `--model` | `config/yolov8n_int8.tflite` | Путь к модели INT8 TFLite |
| `--labels` | `config/labels.txt` | Путь к файлу с названиями классов |
| `--input-size` | `320` | Размер входа модели (квадрат) |
| `--infer-every` | `3` | Инференс каждые N кадров |
| `--conf-thresh` | `0.45` | Порог уверенности |
| `--iou-thresh` | `0.60` | Порог IoU для NMS |
| `--classes` | `0,2,7` | Список COCO class IDs через запятую |
| `--track-iou-thresh` | `0.05` | Порог IoU для сопоставления треков |
| `--smooth-alpha` | `0.3` | Коэффициент EMA-сглаживания (0=нет сглаживания, 1=мгновенно) |
| `--track-life` | `20` | Жизненный трекa (циклы инференса) |
| `--num-threads` | `4` | Потоки интерпретатора TFLite |
| `--cam-width` | `720` | Ширина захвата камеры |
| `--cam-height` | `576` | Высота захвата камеры |
| `--no-fb` | — | Отключить framebuffer (отладочное окно) |
| `--main-log` | `logs/detector.log` | Путь к файлу лога детектора |
| `--guidance` | — | Включить наведение по маркеру |
| `--guidance-port` | — | Serial-порт к полётному контроллеру |
| `--guidance-marker-class` | — | Class ID маркера в вашей модели |
| `--guidance-live` | — | Отключить dry-run (требует подтверждения) |

## Предупреждение о безопасности наведения

**Подключение по MSP само по себе не обеспечивает безопасное переключение RC-управления.**
Модуль наведения требует отдельно настроенной и проверенной конфигурации Betaflight:

- Физический приёмник остаётся подключённым
- Физический AUX1 включает/отключает MSP override
- Переопределяются ТОЛЬКО roll, pitch, yaw и throttle
- AUX1 и все остальные aux-каналы остаются под управлением приёмника
- ARM НЕ назначен на AUX1
- Остановка обновлений MSP обеспечивает проверенное поведение failsafe приёмника

**НЕ настраивайте MSP как единственный приёмник.** Программные флаги не могут
проверить эти требования прошивки.

Начинайте со **снятыми пропеллерами** и `--guidance-dry-run`. Проверьте порядок каналов,
направление, отпускание override, потерю приёмника и USB. Это
**не протестировано на вашей конкретной прошивке и не является готовым автопилотом.**

Размер бокса с одной камеры НЕ является измерением дальности/высоты. Размер бокса —
это только относительный cue для приближения к ОДНОМУ и ТОМУ ЖЕ маркеру.

## Архитектура

### Пайплайн

```
Камера (Picamera2)
    ↓ RGB кадр
    ├─ Каждый N-й кадр: Инференс (YOLOv8 TFLite)
    │       ↓
    │   Постобработка (NMS, фильтрация)
    │       ↓
    │   Трекинг (SimpleIoUTracker)
    │       ↓
    │   Наведение (если включено, через MSP serial)
    │
    └─ Каждый кадр: Отрисовка обнаружений + Запись в framebuffer
```

### Компоненты

- **Detector** — Загружает TFLite модель, выполняет инференс, применяет NMS и фильтрацию классов
- **Tracker** — Greedy-сопоставление по IoU между циклами инференса для устранения мерцания
- **Drawing** — Отрисовка bounding boxes с лейблами и индикаторами coasting
- **Framebuffer** — Запись BGR-кадров в `/dev/fb0` с конвертацией формата
- **Camera** — Обёртка над Picamera2 для упрощённого захвата и отображения
- **Guidance** — Трекинг маркера и генерация RC-каналов MSP (опционально)

## Логирование

Логи пишутся в два файла в папке `logs/`:

- **`logs/detector.log`** — инициализация модели, FPS,confidence, треки, shutdown
- **`logs/marker_guidance.log`** — AUX1, состояния guidance, MSP каналы, ошибки связи

Каждый файл — ротационный, 5MB × 4 бэкапа. Оба дублируются в stdout.

## Исходный проект

https://github.com/Pqazplm/droneAI.git

## Лицензия

Приватная / Внутренняя


