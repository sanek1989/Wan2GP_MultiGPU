import traceback
import sys
try:
    from multi_gpu_utils import test_multi_gpu_setup
    import torch
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    try:
        mgr = test_multi_gpu_setup()
        print('Manager type:', type(mgr))
        print('Device ids:', getattr(mgr, 'device_ids', None))
    except Exception as e:
        print('test_multi_gpu_setup failed:', e)
        traceback.print_exc()
except Exception as e:
    print('Import failed:', e)
    traceback.print_exc()
    sys.exit(2)
print('OK')