#!/usr/bin/env python3
"""
Ultra-quick installer for Wan2GP_MultiGPU on Kaggle
Copy-paste this entire code block into a Kaggle notebook cell
"""

# One-liner for Kaggle Notebook:
exec(open('/kaggle/working/Wan2GP_MultiGPU/install_kaggle.py').read()) if os.path.exists('/kaggle/working/Wan2GP_MultiGPU/install_kaggle.py') else None

# Or use this complete code block:
import subprocess, sys, os
print("🚀 Wan2GP_MultiGPU Quick Setup")
subprocess.run("pip install 'numpy<2.0' 'matplotlib<3.8.0' 'scikit-learn<1.4.0' mmgp==3.6.2 GPUtil psutil --force-reinstall", shell=True)
import torch
print(f"✓ GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.device_count() >= 2:
    os.environ['WANGP_MULTI_GPU_ENABLED'] = '1'
    os.environ['WANGP_GPU_DEVICES'] = '0,1'
    print("✓ Multi-GPU ready! Run: python wgp.py --multi-gpu --gpu-devices 0,1")
else:
    print("⚠️ Single GPU detected")
