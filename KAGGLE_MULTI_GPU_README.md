# Wan2GP Multi-GPU для Kaggle (2x T4)

## Быстрый старт

### 1. Клонирование репозитория в Kaggle Notebook

```python
!git clone https://github.com/sanek1989/Wan2GP_MultiGPU.git
%cd Wan2GP_MultiGPU
```

### 2. Исправление окружения Kaggle (ОБЯЗАТЕЛЬНО!)

Kaggle использует NumPy 2.0, который несовместим с зависимостями Wan2GP:

```python
!python fix_kaggle_env.py
```

Этот скрипт автоматически:
- Понизит NumPy до версии < 2.0
- Установит совместимые версии matplotlib и scikit-learn
- Установит GPUtil для мониторинга GPU

### 3. Установка зависимостей

```python
!pip install -r requirements.txt
```

### 4. Запуск с поддержкой Multi-GPU

```python
!python wgp.py --multi-gpu --gpu-devices 0,1 --share
```

## Параметры командной строки

- `--multi-gpu` - включить поддержку нескольких GPU
- `--gpu-devices 0,1` - использовать GPU 0 и 1 (обе T4 в Kaggle)
- `--share` - создать публичную ссылку Gradio (для доступа вне Kaggle)

## Важные замечания

### Конфликт с dataclasses.py
Файл `dataclasses.py` в корне проекта был переименован в `dataclasses_wangp.py`, чтобы избежать конфликта со стандартным модулем Python.

### Порядок загрузки моделей
Multi-GPU оборачивание применяется ПОСЛЕ конфигурации dtype моделей, чтобы обеспечить правильную инициализацию параметров.

### Мониторинг GPU
Логи будут показывать использование обоих GPU:
```
Multi-GPU enabled. Devices: [0, 1], primary: cuda:0
GPU 0: Tesla T4, Memory: 15.0 GB
GPU 1: Tesla T4, Memory: 15.0 GB
```

## Устранение неполадок

### Ошибка "cannot import name 'asdict' from 'dataclasses'"
Запустите `fix_kaggle_env.py` еще раз или перезапустите kernel.

### Ошибка "CUDA out of memory"
- Уменьшите batch size
- Уменьшите разрешение видео
- Проверьте, что используются обе GPU (должны быть логи "Wrapping self.model with DataParallel")

### Модель использует только одну GPU
Убедитесь, что:
1. Передан параметр `--multi-gpu`
2. В логах есть сообщение "Wrapping ... with DataParallel"
3. Переменные окружения установлены: `WANGP_MULTI_GPU_ENABLED=1`

## Производительность

С 2x T4 (по 16 GB каждая) вы получите:
- ~2x ускорение inference для больших моделей
- Возможность обрабатывать более высокие разрешения
- Более стабильную генерацию без OOM

## Дополнительная информация

- [Основной README](README.md)
- [Документация по установке](docs/INSTALLATION.md)
- [Устранение неполадок](docs/TROUBLESHOOTING.md)
