# Multi-GPU utilities for Wan2GP_MultiGPU
# Provides DataParallel support for T4 GPUs in Kaggle environment

import os
import logging
import gc
from typing import Optional, List, Dict, Any
import time

# safe torch import
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    class _NNStub:
        pass
    nn = _NNStub

try:
    import psutil
except Exception:
    psutil = None
try:
    import GPUtil
except Exception:
    GPUtil = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGPUManager:
    """Manages multi-GPU setup and monitoring for Wan2GP"""
    
    def __init__(self, device_ids: Optional[List[int]] = None, enable_monitoring: bool = True):
        """
        Initialize Multi-GPU Manager
        
        Args:
            device_ids: List of GPU device IDs to use. If None, uses all available GPUs.
            enable_monitoring: Whether to enable GPU monitoring
        """
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.enable_monitoring = enable_monitoring
        self.primary_device = f"cuda:{self.device_ids[0]}" if self.device_ids else "cuda:0"
        
        # Validate GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        if len(self.device_ids) < 2:
            logger.warning(f"Only {len(self.device_ids)} GPU(s) available. Multi-GPU benefits may be limited.")
        
        logger.info(f"Multi-GPU Manager initialized with devices: {self.device_ids}")
        self._log_gpu_info()
    
    def _log_gpu_info(self):
        """Log information about available GPUs"""
        for i in self.device_ids:
            if i < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
    
    def wrap_model_with_dataparallel(self, model: Any) -> Any:
        """
        Wrap model with DataParallel for multi-GPU inference
        
        Args:
            model: The model to wrap
            
        Returns:
            DataParallel wrapped model
        """
        if len(self.device_ids) < 2 or not torch.cuda.is_available():
            logger.info("Single GPU detected or CUDA unavailable, skipping DataParallel wrapper")
            try:
                return model.to(self.primary_device)
            except Exception:
                return model

        # Ensure model is on the primary device first (best-effort)
        try:
            model = model.to(self.primary_device)
        except Exception:
            pass

        # Wrap with DataParallel
        try:
            parallel_model = nn.DataParallel(model, device_ids=self.device_ids)
            logger.info(f"Model wrapped with DataParallel using devices: {self.device_ids}")
            return parallel_model
        except Exception:
            logger.warning("DataParallel wrapping failed, returning original model")
            return model

    def wrap_pipe(self, pipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap common nn.Modules inside a pipe mapping (e.g., transformer, vae, text_encoder)
        with DataParallel when multi-GPU is enabled.
        """
        if not isinstance(pipe, dict):
            return pipe
        if len(self.device_ids) < 2 or not torch.cuda.is_available():
            return pipe

        for k, v in list(pipe.items()):
            try:
                if isinstance(v, nn.Module):
                    name = k.lower()
                    if any(x in name for x in ("transformer", "model", "vae", "text_encoder", "clip")):
                        pipe[k] = self.wrap_model_with_dataparallel(v)
            except Exception:
                pass
        return pipe
    
    def optimize_batch_size_for_multi_gpu(self, base_batch_size: int) -> int:
        """
        Optimize batch size for multi-GPU processing
        
        Args:
            base_batch_size: Original batch size
            
        Returns:
            Optimized batch size
        """
        # Ensure batch size is divisible by number of GPUs
        num_gpus = len(self.device_ids)
        if base_batch_size % num_gpus != 0:
            optimized_batch_size = ((base_batch_size + num_gpus - 1) // num_gpus) * num_gpus
            logger.info(f"Adjusted batch size from {base_batch_size} to {optimized_batch_size} for {num_gpus} GPUs")
            return optimized_batch_size
        return base_batch_size
    
    def split_data_for_gpus(self, data: Any) -> Any:
        """
        Split data appropriately for multi-GPU processing
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor ready for multi-GPU processing
        """
        # DataParallel automatically handles data splitting
        # This method is kept for potential future custom splitting logic
        return data
    
    def monitor_gpu_usage(self) -> Dict[str, Any]:
        """
        Monitor GPU usage and memory
        
        Returns:
            Dictionary with GPU usage information
        """
        if not self.enable_monitoring:
            return {}
        
        gpu_info = {}
        try:
            # Use GPUtil for detailed GPU monitoring
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if i in self.device_ids:
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "load": gpu.load * 100,  # Convert to percentage
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_util": gpu.memoryUtil * 100,  # Convert to percentage
                        "temperature": gpu.temperature
                    }
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
            # Fallback to basic torch monitoring
            for i in self.device_ids:
                if i < torch.cuda.device_count():
                    gpu_info[f"gpu_{i}"] = {
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory
                    }
        
        return gpu_info
    
    def log_gpu_status(self, prefix: str = ""):
        """Log current GPU status"""
        gpu_info = self.monitor_gpu_usage()
        if gpu_info:
            logger.info(f"{prefix}GPU Status:")
            for gpu_id, info in gpu_info.items():
                if "load" in info:
                    logger.info(f"  {gpu_id}: Load {info['load']:.1f}%, Memory {info['memory_util']:.1f}%, Temp {info['temperature']}°C")
                else:
                    mem_alloc = info['memory_allocated'] / 1024**3
                    mem_total = info['memory_total'] / 1024**3
                    logger.info(f"  {gpu_id}: Memory {mem_alloc:.1f}/{mem_total:.1f} GB")
    
    def clear_gpu_cache(self):
        """Clear GPU cache on all devices"""
        for i in self.device_ids:
            if i < torch.cuda.device_count():
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU cache cleared on all devices")
    
    def get_primary_device(self) -> str:
        """Get the primary device string"""
        return self.primary_device
    
    def get_device_count(self) -> int:
        """Get number of available devices"""
        return len(self.device_ids)

    def get_gpu_ids(self) -> List[int]:
        """Return the list of device ids this manager controls."""
        return list(self.device_ids)

    def log_wrapped_modules(self, wan_model: Any, prefix: str = ""):
        """
        Log devices for key wrapped modules of a wan_model for debugging.
        """
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

    def wrap_wan_model(self, wan_model: Any) -> Any:
        """
        Safely wrap internal components of a wan_model in DataParallel when multi-GPU is enabled.

        This will attempt to wrap attributes commonly present on Wan models:
        - model, model2
        - vae
        - text_encoder (and text_encoder.model)
        - clip (and clip.model)

        The function returns the (possibly modified) wan_model.
        """
        if wan_model is None:
            return wan_model
        if len(self.device_ids) < 2 or not torch.cuda.is_available():
            return wan_model

        try:
            # helper to wrap a value if it's an nn.Module
            def _wrap_if_module(obj):
                try:
                    if isinstance(obj, nn.Module):
                        return self.wrap_model_with_dataparallel(obj)
                except Exception:
                    pass
                return obj

            # wrap main transformer(s)
            if hasattr(wan_model, 'model') and isinstance(getattr(wan_model, 'model'), nn.Module):
                wan_model.model = _wrap_if_module(wan_model.model)
            if hasattr(wan_model, 'model2') and isinstance(getattr(wan_model, 'model2'), nn.Module):
                wan_model.model2 = _wrap_if_module(wan_model.model2)

            # wrap vae if present
            if hasattr(wan_model, 'vae') and getattr(wan_model, 'vae') is not None:
                try:
                    # some VAEs expose .model
                    if isinstance(getattr(wan_model.vae, 'model', None), nn.Module):
                        wan_model.vae.model = _wrap_if_module(wan_model.vae.model)
                    else:
                        wan_model.vae = _wrap_if_module(wan_model.vae)
                except Exception:
                    pass

            # wrap text encoders
            if hasattr(wan_model, 'text_encoder'):
                te = getattr(wan_model, 'text_encoder')
                if te is not None:
                    try:
                        if isinstance(getattr(te, 'model', None), nn.Module):
                            te.model = _wrap_if_module(te.model)
                    except Exception:
                        pass

            # wrap clip if present
            if hasattr(wan_model, 'clip'):
                try:
                    clip = getattr(wan_model, 'clip')
                    if clip is not None and isinstance(getattr(clip, 'model', None), nn.Module):
                        clip.model = _wrap_if_module(clip.model)
                except Exception:
                    pass
        except Exception:
            # best-effort: don't break if wrapping fails
            pass
        return wan_model


class MultiGPUOffloadManager:
    """Enhanced offload manager for multi-GPU scenarios"""
    
    def __init__(self, multi_gpu_manager: MultiGPUManager):
        self.multi_gpu_manager = multi_gpu_manager
        self.device_usage = {}
        
    def distribute_model_components(self, model_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distribute model components across available GPUs
        
        Args:
            model_components: Dictionary of model components
            
        Returns:
            Distributed model components
        """
        distributed_components = {}
        num_gpus = len(self.multi_gpu_manager.device_ids)
        
        for i, (name, component) in enumerate(model_components.items()):
            # Round-robin distribution
            target_device = self.multi_gpu_manager.device_ids[i % num_gpus]
            distributed_components[name] = component.to(f"cuda:{target_device}")
            self.device_usage[name] = target_device
            logger.info(f"Component '{name}' assigned to GPU {target_device}")
        
        return distributed_components
    
    def optimize_memory_allocation(self):
        """Optimize memory allocation across GPUs"""
        for device_id in self.multi_gpu_manager.device_ids:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        gc.collect()


def create_multi_gpu_manager(device_ids: Optional[List[int]] = None) -> MultiGPUManager:
    """
    Factory function to create MultiGPUManager
    
    Args:
        device_ids: List of GPU device IDs to use
        
    Returns:
        Configured MultiGPUManager instance
    """
    return MultiGPUManager(device_ids=device_ids)


def setup_multi_gpu_environment():
    """
    Setup optimal environment for multi-GPU processing
    
    Returns:
        MultiGPUManager instance
    """
    # Set optimal environment variables
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    
    # Create multi-GPU manager
    manager = create_multi_gpu_manager()
    
    # Log initial status
    manager.log_gpu_status("Initial")
    
    return manager


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics for multi-GPU operations"""
    
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing an operation and record metric"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            logger.info(f"{operation} took {duration:.2f} seconds")
            del self.start_times[operation]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return self.metrics.copy()


# Utility functions for compatibility
def ensure_tensor_on_device(tensor, device: str):
    """Ensure tensor is on the specified device"""
    if torch is None:
        raise ImportError("torch не импортирован, функция ensure_tensor_on_device недоступна")
    if not hasattr(torch, 'Tensor'):
        raise ImportError("torch не содержит тип Tensor, проверьте установку")
    if tensor.device != torch.device(device):
        return tensor.to(device)
    return tensor
    return tensor


def get_optimal_device_for_tensor(tensor: torch.Tensor, available_devices: List[int]) -> str:
    """Get optimal device for a tensor based on size and available devices"""
    # Simple heuristic: use the device with least memory usage
    device_memory = {}
    for device_id in available_devices:
        if device_id < torch.cuda.device_count():
            device_memory[device_id] = torch.cuda.memory_allocated(device_id)
    
    if device_memory:
        optimal_device_id = min(device_memory, key=device_memory.get)
        return f"cuda:{optimal_device_id}"
    
    return "cuda:0"


# Example usage and testing functions
def test_multi_gpu_setup():
    """Test multi-GPU setup and functionality"""
    logger.info("Testing multi-GPU setup...")
    
    # Create manager
    manager = create_multi_gpu_manager()
    
    # Test basic functionality
    if len(manager.device_ids) >= 2:
        logger.info("Multi-GPU setup test passed")
        
        # Create a simple test model
        test_model = nn.Linear(100, 100)
        wrapped_model = manager.wrap_model_with_dataparallel(test_model)
        
        # Test with dummy data
        test_data = torch.randn(4, 100)
        with torch.no_grad():
            output = wrapped_model(test_data)
        
        logger.info(f"Test model output shape: {output.shape}")
        manager.log_gpu_status("After test")
        
    else:
        logger.warning("Single GPU detected, multi-GPU features not tested")
    
    return manager
