#!/usr/bin/env python3
"""
Install all dependencies for Wan2GP_MultiGPU
Run this in Kaggle notebook to install everything needed
"""

import subprocess
import sys

def install_dependencies():
    """Install all required dependencies"""
    print("🚀 Installing Wan2GP_MultiGPU dependencies...")
    
    # Core dependencies that must be installed first
    core_deps = [
        "numpy<2.0",
        "matplotlib<3.8.0", 
        "scikit-learn<1.4.0",
        "mmgp==3.6.2",
        "GPUtil",
        "psutil"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "--force-reinstall"], 
                         check=True, capture_output=True)
            print(f"✓ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
    
    # Install from requirements.txt if it exists
    try:
        print("Installing from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ requirements.txt installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not install from requirements.txt: {e}")
    
    print("\n✅ Installation complete!")
    print("You can now run: python wgp.py --multi-gpu --gpu-devices 0,1")

if __name__ == "__main__":
    install_dependencies()
