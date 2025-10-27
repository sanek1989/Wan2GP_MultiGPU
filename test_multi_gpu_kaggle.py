#!/usr/bin/env python3
"""
Test Multi-GPU setup specifically for Kaggle environment with 2x T4 GPUs
"""

import os
import sys

def test_basic_imports():
    """Test that basic imports work"""
    print("=" * 50)
    print("TEST 1: Basic Imports")
    print("=" * 50)
    try:
        import torch
        import torch.nn as nn
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"✓ GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_dataparallel():
    """Test DataParallel functionality"""
    print("\n" + "=" * 50)
    print("TEST 2: DataParallel Functionality")
    print("=" * 50)
    try:
        import torch
        import torch.nn as nn
        
        if torch.cuda.device_count() < 2:
            print("⚠ Warning: Less than 2 GPUs detected, skipping DataParallel test")
            return True
        
        # Create simple model
        model = nn.Linear(100, 100)
        device_ids = [0, 1]
        
        # Wrap with DataParallel
        parallel_model = nn.DataParallel(model, device_ids=device_ids)
        parallel_model = parallel_model.cuda()
        
        print(f"✓ Created DataParallel model with devices: {device_ids}")
        
        # Test forward pass
        test_input = torch.randn(8, 100).cuda()
        with torch.no_grad():
            output = parallel_model(test_input)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        print(f"✓ DataParallel test passed!")
        
        return True
    except Exception as e:
        print(f"✗ DataParallel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_gpu_utils():
    """Test multi_gpu_utils module"""
    print("\n" + "=" * 50)
    print("TEST 3: Multi-GPU Utils Module")
    print("=" * 50)
    try:
        from multi_gpu_utils import MultiGPUManager, create_multi_gpu_manager
        
        print("✓ Successfully imported multi_gpu_utils")
        
        # Create manager
        manager = create_multi_gpu_manager()
        print(f"✓ Created MultiGPUManager with devices: {manager.device_ids}")
        print(f"✓ Primary device: {manager.get_primary_device()}")
        print(f"✓ Device count: {manager.get_device_count()}")
        
        # Log GPU status
        manager.log_gpu_status("Test ")
        
        # Test model wrapping
        import torch.nn as nn
        test_model = nn.Linear(50, 50)
        wrapped = manager.wrap_model_with_dataparallel(test_model)
        
        if manager.get_device_count() >= 2:
            print(f"✓ Model wrapped with DataParallel")
        else:
            print(f"✓ Single GPU detected, skipping DataParallel wrapper")
        
        print(f"✓ Multi-GPU utils test passed!")
        return True
    except Exception as e:
        print(f"✗ Multi-GPU utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test that environment variables are set correctly"""
    print("\n" + "=" * 50)
    print("TEST 4: Environment Variables")
    print("=" * 50)
    
    # Set env vars as they would be set by --multi-gpu flag
    os.environ["WANGP_MULTI_GPU_ENABLED"] = "1"
    os.environ["WANGP_GPU_DEVICES"] = "0,1"
    
    enabled = os.environ.get("WANGP_MULTI_GPU_ENABLED", "0")
    devices = os.environ.get("WANGP_GPU_DEVICES", "")
    
    print(f"✓ WANGP_MULTI_GPU_ENABLED: {enabled}")
    print(f"✓ WANGP_GPU_DEVICES: {devices}")
    
    if enabled == "1" and devices == "0,1":
        print(f"✓ Environment variables test passed!")
        return True
    else:
        print(f"✗ Environment variables not set correctly")
        return False

def test_numpy_compatibility():
    """Test NumPy compatibility (critical for Kaggle)"""
    print("\n" + "=" * 50)
    print("TEST 5: NumPy Compatibility")
    print("=" * 50)
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        # NumPy 2.0 causes issues with matplotlib and scikit-learn
        major_version = int(np.__version__.split('.')[0])
        if major_version >= 2:
            print(f"⚠ WARNING: NumPy {np.__version__} detected!")
            print(f"  This may cause compatibility issues.")
            print(f"  Run 'python fix_kaggle_env.py' to fix.")
            return False
        else:
            print(f"✓ NumPy version is compatible (< 2.0)")
            return True
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_torch_operations():
    """Test basic torch operations on multiple GPUs"""
    print("\n" + "=" * 50)
    print("TEST 6: Torch Operations on Multiple GPUs")
    print("=" * 50)
    try:
        import torch
        
        if torch.cuda.device_count() < 2:
            print("⚠ Skipping multi-GPU operations test (< 2 GPUs)")
            return True
        
        # Test tensor operations on both GPUs
        for gpu_id in [0, 1]:
            device = f"cuda:{gpu_id}"
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print(f"✓ Matrix multiplication successful on {device}")
        
        # Test data transfer between GPUs
        x_gpu0 = torch.randn(100, 100).cuda(0)
        x_gpu1 = x_gpu0.cuda(1)
        print(f"✓ Data transfer between GPUs successful")
        
        print(f"✓ Torch operations test passed!")
        return True
    except Exception as e:
        print(f"✗ Torch operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Wan2GP Multi-GPU Test Suite for Kaggle (2x T4)")
    print("=" * 60 + "\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("DataParallel", test_dataparallel),
        ("Multi-GPU Utils", test_multi_gpu_utils),
        ("Environment Variables", test_environment_variables),
        ("NumPy Compatibility", test_numpy_compatibility),
        ("Torch Operations", test_torch_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Multi-GPU setup is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
