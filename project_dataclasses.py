from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Base configuration for AI models"""
    model_type: str
    model_path: str
    device: str = "cuda"
    precision: str = "fp16"
    enabled: bool = True


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU setup"""
    enabled: bool = False
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    memory_pooling: bool = True
    load_balancing: str = "auto"


@dataclass
class InferenceParams:
    """Parameters for model inference"""
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None


@dataclass
class VideoGenerationParams(InferenceParams):
    """Parameters for video generation"""
    frames: int = 24
    fps: int = 24
    duration: float = 1.0
    loop: bool = False
