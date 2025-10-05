#!/usr/bin/env python3
"""
One-line installer for Wan2GP_MultiGPU on Kaggle
Usage in Kaggle Notebook:
    exec(open('install_kaggle.py').read())
"""

import subprocess
import sys
import os
import urllib.request
import shutil

def print_step(step, message):
    """Print formatted step message"""
    print(f"\n{'='*50}")
    print(f"STEP {step}: {message}")
    print(f"{'='*50}")

def run_command(cmd, description=""):
    """Run command and handle errors"""
    if description:
        print(f"  {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"    ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error: {e}")
        if e.stderr:
            print(f"    Stderr: {e.stderr}")
        return False

def install_wangp_multi_gpu():
    """Main installation function"""
    print("🚀 Wan2GP_MultiGPU Kaggle Installer")
    print("   Optimized for 2x T4 GPUs")
    
    # Step 1: Fix NumPy compatibility
    print_step(1, "Fixing NumPy 2.0 compatibility issues")
    packages_to_fix = [
        ("'numpy<2.0'", "Downgrading NumPy"),
        ("'matplotlib<3.8.0'", "Downgrading matplotlib"), 
        ("'scikit-learn<1.4.0'", "Downgrading scikit-learn")
    ]
    
    for package, desc in packages_to_fix:
        if not run_command(f"pip install {package} --force-reinstall", desc):
            print(f"    ⚠️  Warning: Failed to install {package}")
    
    # Step 2: Install core dependencies
    print_step(2, "Installing core dependencies")
    run_command("pip install mmgp==3.6.2", "Installing mmgp")
    run_command("pip install GPUtil", "Installing GPUtil")
    
    # Step 3: Install additional requirements
    print_step(3, "Installing additional requirements")
    run_command("pip install psutil", "Installing psutil")
    
    # Step 4: Verify GPU setup
    print_step(4, "Verifying GPU setup")
    
    try:
        import torch
        print(f"    ✓ PyTorch version: {torch.__version__}")
        print(f"    ✓ CUDA available: {torch.cuda.is_available()}")
        print(f"    ✓ GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"    ✓ GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        else:
            print("    ⚠️  No CUDA GPUs detected!")
            
    except ImportError:
        print("    ✗ PyTorch not available")
    
    # Step 5: Test multi-GPU utilities
    print_step(5, "Testing multi-GPU utilities")
    
    try:
        # Create a simple test
        test_code = """
import os
import torch
import torch.nn as nn

# Test DataParallel setup
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device_ids = list(range(min(2, torch.cuda.device_count())))
    print(f"    ✓ Multi-GPU setup ready with devices: {device_ids}")
    
    # Test DataParallel
    test_model = nn.Linear(100, 100)
    parallel_model = nn.DataParallel(test_model, device_ids=device_ids)
    print(f"    ✓ DataParallel wrapper working")
    
    # Set environment variables for Wan2GP
    os.environ['WANGP_MULTI_GPU_ENABLED'] = '1'
    os.environ['WANGP_GPU_DEVICES'] = ','.join(map(str, device_ids))
    print(f"    ✓ Environment variables set")
else:
    print(f"    ⚠️  Single GPU or no GPU detected")
"""
        exec(test_code)
        
    except Exception as e:
        print(f"    ✗ Multi-GPU test failed: {e}")
    
    # Step 6: Final instructions
    print_step(6, "Installation Complete!")
    
    print("""
🎉 Wan2GP_MultiGPU is ready for Kaggle!

📋 To start generation with 2x T4 GPUs:
   python wgp.py --multi-gpu --gpu-devices 0,1

📊 GPU monitoring will be shown in logs during generation.

🔧 If you encounter issues:
   1. Restart the notebook kernel
   2. Re-run this installer
   3. Check GPU availability: !nvidia-smi

💡 For faster setup next time, just run:
   exec(open('install_kaggle.py').read())
""")

if __name__ == "__main__":
    install_wangp_multi_gpu()
