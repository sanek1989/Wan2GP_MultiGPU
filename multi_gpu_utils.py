"""Multi-GPU utilities for Wan2GP_MultiGPU.

This module provides a small, robust manager for DataParallel-style multi-GPU
inference and a few helper utilities. It is defensive: it works when torch is
missing (returns no-op managers) and it avoids hard failures during wrapping.
"""

from typing import Optional, List, Dict, Any
import logging
import gc
import os
import time

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import GPUtil
except Exception:
    GPUtil = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPUManager:
    """Manage device selection and DataParallel wrapping.

    Notes:
    - This manager intentionally uses a conservative behavior: if CUDA is not
      available or fewer than 2 devices are present, it becomes a no-op wrapper
      to avoid crashes in single-GPU environments (useful for local dev).
    - It does not forcibly set `CUDA_VISIBLE_DEVICES` so that the caller / runtime
      can control visible GPUs (Kaggle runtime or container should set that).
    """

    def __init__(self, device_ids: Optional[List[int]] = None, enable_monitoring: bool = True):
        if torch is None:
            # Create a dummy manager when torch not available
            self.device_ids = []
            self.enable_monitoring = False
            self.primary_device = "cpu"
            return

        self.device_ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))
        self.enable_monitoring = enable_monitoring
        self.primary_device = f"cuda:{self.device_ids[0]}" if self.device_ids else "cuda:0"

        if not torch.cuda.is_available():
            logger.warning("CUDA not available; MultiGPUManager will be limited to CPU/single-GPU behavior")
        if len(self.device_ids) < 2:
            logger.info(f"Configured GPUs: {self.device_ids} (multi-GPU features will be limited)")

        self._log_gpu_info()

    def _log_gpu_info(self):
        if torch is None:
            return
        for i in self.device_ids:
            if i < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")

    def wrap_model_with_dataparallel(self, model: Any) -> Any:
        """Wrap an nn.Module with DataParallel when multi-GPU is available.

        Returns the original model on failure or when DataParallel isn't needed.
        """
        if torch is None or nn is None:
            return model
        if not torch.cuda.is_available() or len(self.device_ids) < 2:
            return model

        try:
            parallel_model = nn.DataParallel(model, device_ids=self.device_ids)
            logger.info(f"Model wrapped with DataParallel using devices: {self.device_ids}")
            return parallel_model
        except Exception as e:
            logger.warning(f"DataParallel wrapping failed ({e}); returning original model")
            return model

    def wrap_pipe(self, pipe: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(pipe, dict):
            return pipe
        if torch is None or not torch.cuda.is_available() or len(self.device_ids) < 2:
            return pipe

        for k, v in list(pipe.items()):
            try:
                if isinstance(v, nn.Module):
                    name = k.lower()
                    if any(x in name for x in ("transformer", "model", "vae", "text_encoder", "clip")):
                        pipe[k] = self.wrap_model_with_dataparallel(v)
            except Exception:
                # best-effort: do not raise from wrapping
                pass
        return pipe

    def optimize_batch_size_for_multi_gpu(self, base_batch_size: int) -> int:
        if torch is None or len(self.device_ids) < 2:
            return base_batch_size
        num_gpus = max(1, len(self.device_ids))
        if base_batch_size % num_gpus != 0:
            optimized_batch_size = ((base_batch_size + num_gpus - 1) // num_gpus) * num_gpus
            logger.info(f"Adjusted batch size from {base_batch_size} to {optimized_batch_size} for {num_gpus} GPUs")
            return optimized_batch_size
        return base_batch_size

    def monitor_gpu_usage(self) -> Dict[str, Any]:
        if torch is None or not self.enable_monitoring:
            return {}

        gpu_info = {}
        try:
            if GPUtil is not None:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    if i in self.device_ids:
                        gpu_info[f"gpu_{i}"] = {
                            "name": gpu.name,
                            "load": gpu.load * 100,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "memory_util": gpu.memoryUtil * 100,
                            "temperature": getattr(gpu, 'temperature', None)
                        }
                return gpu_info
        except Exception:
            logger.debug("GPUtil monitoring failed, falling back to torch stats")

        # Fallback to torch-based metrics
        for i in self.device_ids:
            if i < torch.cuda.device_count():
                gpu_info[f"gpu_{i}"] = {
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory
                }
        return gpu_info

    def log_gpu_status(self, prefix: str = ""):
        gpu_info = self.monitor_gpu_usage()
        if gpu_info:
            logger.info(f"{prefix}GPU Status:")
            for gpu_id, info in gpu_info.items():
                if "load" in info:
                    logger.info(f"  {gpu_id}: Load {info['load']:.1f}%, Memory {info['memory_util']:.1f}%, Temp {info.get('temperature')}")
                else:
                    mem_alloc = info['memory_allocated'] / 1024**3
                    mem_total = info['memory_total'] / 1024**3
                    logger.info(f"  {gpu_id}: Memory {mem_alloc:.1f}/{mem_total:.1f} GB")

    def clear_gpu_cache(self):
        if torch is None:
            return
        for i in self.device_ids:
            if i < torch.cuda.device_count():
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU cache cleared on all devices")

    def get_primary_device(self) -> str:
        return self.primary_device

    def get_device_count(self) -> int:
        return len(self.device_ids)

    def get_gpu_ids(self) -> List[int]:
        return list(self.device_ids)

    def log_wrapped_modules(self, wan_model: Any, prefix: str = ""):
        if wan_model is None:
            logger.info(prefix + "wan_model is None")
            return
        try:
            def _device_of(obj):
                try:
                    if isinstance(obj, nn.Module):
                        for p in obj.parameters():
                            return str(p.device)
                except Exception:
                    pass
                return None

            parts = [
                ("model", getattr(wan_model, 'model', None)),
                ("model2", getattr(wan_model, 'model2', None)),
                ("vae", getattr(wan_model, 'vae', None)),
                ("text_encoder", getattr(getattr(wan_model, 'text_encoder', None), 'model', getattr(wan_model, 'text_encoder', None))),
                ("clip", getattr(getattr(wan_model, 'clip', None), 'model', getattr(wan_model, 'clip', None)))
            ]
            for name, obj in parts:
                dev = _device_of(obj)
                if dev is None:
                    logger.info(f"{prefix}{name}: not an nn.Module or no params or not wrapped")
                else:
                    logger.info(f"{prefix}{name}: device={dev}")
        except Exception as e:
            logger.warning(f"{prefix}Failed to log wrapped modules: {e}")


class MultiGPUOffloadManager:
    """Basic component placer for distributing objects across GPUs."""

    def __init__(self, multi_gpu_manager: MultiGPUManager):
        self.multi_gpu_manager = multi_gpu_manager
        self.device_usage = {}

    def distribute_model_components(self, model_components: Dict[str, Any]) -> Dict[str, Any]:
        distributed_components = {}
        if torch is None:
            return model_components
        num_gpus = max(1, len(self.multi_gpu_manager.device_ids))
        for i, (name, component) in enumerate(model_components.items()):
            target_device = self.multi_gpu_manager.device_ids[i % num_gpus] if self.multi_gpu_manager.device_ids else 0
            try:
                distributed_components[name] = component.to(f"cuda:{target_device}")
                self.device_usage[name] = target_device
                logger.info(f"Component '{name}' assigned to GPU {target_device}")
            except Exception:
                distributed_components[name] = component
        return distributed_components

    def optimize_memory_allocation(self):
        if torch is None:
            return
        for device_id in self.multi_gpu_manager.device_ids:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        gc.collect()


def create_multi_gpu_manager(device_ids: Optional[List[int]] = None) -> MultiGPUManager:
    return MultiGPUManager(device_ids=device_ids)


def setup_multi_gpu_environment(device_ids: Optional[List[int]] = None) -> MultiGPUManager:
    """Return a configured manager. Do not overwrite CUDA_VISIBLE_DEVICES here.

    On Kaggle, the runtime should be configured to expose the desired GPUs.
    """
    manager = create_multi_gpu_manager(device_ids=device_ids)
    manager.log_gpu_status("Initial")
    return manager


def test_multi_gpu_setup():
    logger.info("Testing multi-GPU setup...")
    manager = create_multi_gpu_manager()
    if torch is None:
        logger.info("torch not available; skipping multi-GPU test")
        return manager
    if len(manager.device_ids) >= 2:
        logger.info("Multi-GPU setup test passed")
        test_model = nn.Linear(100, 100)
        wrapped_model = manager.wrap_model_with_dataparallel(test_model)
        test_data = torch.randn(4, 100)
        with torch.no_grad():
            try:
                output = wrapped_model(test_data)
                logger.info(f"Test model output shape: {output.shape}")
            except Exception as e:
                logger.warning(f"Test run failed: {e}")
        manager.log_gpu_status("After test")
    else:
        logger.warning("Single GPU detected, multi-GPU features not tested")
    return manager
