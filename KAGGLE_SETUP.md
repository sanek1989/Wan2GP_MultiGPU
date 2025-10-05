# Wan2GP_MultiGPU - Kaggle Setup Guide

## Проблема
Kaggle использует NumPy 2.0, который несовместим с matplotlib и scikit-learn в текущих версиях.

## Решение

### Вариант 1: Автоматическое исправление
```bash
python fix_kaggle_env.py
```

### Вариант 2: Ручное исправление
```bash
# Откатить NumPy на совместимую версию
pip install 'numpy<2.0' --force-reinstall

# Откатить matplotlib
pip install 'matplotlib<3.8.0' --force-reinstall

# Откатить scikit-learn  
pip install 'scikit-learn<1.4.0' --force-reinstall

# Установить GPUtil для мониторинга GPU
pip install GPUtil
```

### Вариант 3: Полная переустановка зависимостей
```bash
pip install -r requirements.txt --force-reinstall
```

## Запуск с Multi-GPU

### Включить GPU в Kaggle
1. Settings → Accelerator → GPU T4 x2

### Запустить с поддержкой двух GPU
```bash
python wgp.py --multi-gpu --gpu-devices 0,1
```

### Мониторинг GPU
Во время генерации в логах будет выводиться:
- Статус загрузки обоих GPU
- Использование памяти
- Температура

### Пример вывода
```
INFO:__main__:Startup GPU Status:
INFO:__main__:  gpu_0: Load 45.2%, Memory 67.3%, Temp 67°C
INFO:__main__:  gpu_1: Load 43.8%, Memory 65.1%, Temp 65°C
```

## Проверка установки
```bash
python -c "from multi_gpu_utils import test_multi_gpu_setup; test_multi_gpu_setup()"
```

## Troubleshooting

### Если все еще ошибки NumPy
```bash
pip uninstall numpy matplotlib scikit-learn -y
pip install 'numpy==1.24.3' 'matplotlib==3.7.2' 'scikit-learn==1.3.0'
```

### Если не видит GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Если DataParallel не работает
Проверьте переменные окружения:
```bash
python -c "import os; print('Multi-GPU enabled:', os.environ.get('WANGP_MULTI_GPU_ENABLED', '0')); print('GPU devices:', os.environ.get('WANGP_GPU_DEVICES', 'none'))"
```
