#!/usr/bin/env python3
"""
Fix Kaggle environment for Wan2GP_MultiGPU compatibility
Resolves NumPy 2.0 compatibility issues with matplotlib and scikit-learn
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run shell command and print output"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def fix_kaggle_environment():
    """Fix Kaggle environment compatibility issues"""
    print("=== Fixing Kaggle Environment for Wan2GP_MultiGPU ===")
    
    # Downgrade numpy to compatible version
    print("\n1. Downgrading NumPy to compatible version...")
    if not run_command("pip install 'numpy<2.0' --force-reinstall"):
        print("Failed to downgrade NumPy")
        return False
    
    # Downgrade matplotlib
    print("\n2. Downgrading matplotlib...")
    if not run_command("pip install 'matplotlib<3.8.0' --force-reinstall"):
        print("Failed to downgrade matplotlib")
        return False
    
    # Downgrade scikit-learn
    print("\n3. Downgrading scikit-learn...")
    if not run_command("pip install 'scikit-learn<1.4.0' --force-reinstall"):
        print("Failed to downgrade scikit-learn")
        return False
    
    # Install missing GPUtil if not present
    print("\n4. Installing GPUtil for GPU monitoring...")
    run_command("pip install GPUtil")
    
    print("\n=== Environment fix completed ===")
    print("You can now run: python wgp.py --multi-gpu --gpu-devices 0,1")
    return True

if __name__ == "__main__":
    fix_kaggle_environment()
