# 🚀 Wan2GP_MultiGPU - Quick Start для Kaggle

## Установка в одну строку

### Вариант 1: Полная установка
```python
exec(open('install_kaggle.py').read())
```

### Вариант 2: Быстрая установка всех зависимостей
```python
exec(open('install_all_deps.py').read())
```

### Вариант 3: Минимальная установка
```python
import subprocess, os; subprocess.run("pip install 'numpy<2.0' 'matplotlib<3.8.0' 'scikit-learn<1.4.0' transformers==4.53.1 optimum-quanto mmgp==3.6.2 GPUtil --force-reinstall", shell=True); import torch; os.environ.update({'WANGP_MULTI_GPU_ENABLED': '1', 'WANGP_GPU_DEVICES': '0,1'}) if torch.cuda.device_count() >= 2 else None; print(f"✓ GPUs: {torch.cuda.device_count()}, Ready: {torch.cuda.device_count() >= 2}")
```

### Вариант 4: Если mmgp не найден
```python
!pip install mmgp==3.6.2 --force-reinstall
```

## Запуск генерации

### 1. Включить GPU в Kaggle
- Settings → Accelerator → **GPU T4 x2**

### 2. Запустить с двумя GPU
```python
!python wgp.py --multi-gpu --gpu-devices 0,1
```

## Что происходит автоматически

1. **Исправляет NumPy 2.0** → откат на совместимую версию
2. **Устанавливает GPUtil** → мониторинг GPU
3. **Настраивает DataParallel** → распределение на 2 GPU
4. **Проверяет GPU** → показывает статус карт
5. **Устанавливает переменные** → WANGP_MULTI_GPU_ENABLED=1

## Мониторинг GPU

Во время генерации в логах:
```
INFO: gpu_0: Load 45.2%, Memory 67.3%, Temp 67°C
INFO: gpu_1: Load 43.8%, Memory 65.1%, Temp 65°C
```

## Troubleshooting

### Если ошибка NumPy
```python
!pip install 'numpy==1.24.3' --force-reinstall
```

### Если не видит GPU
```python
import torch; print(f"GPUs: {torch.cuda.device_count()}")
```

### Перезапуск ядра
- Runtime → Restart Runtime
- Повторить установку

## Полный пример для копирования

```python
# Установка
exec(open('install_kaggle.py').read())

# Запуск
!python wgp.py --multi-gpu --gpu-devices 0,1
```

**Готово!** Теперь Wan2GP использует оба T4 GPU для ускоренной генерации.
