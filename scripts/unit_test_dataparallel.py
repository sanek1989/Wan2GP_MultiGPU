import torch
from multi_gpu_utils import create_multi_gpu_manager

# Simple unit test: create a small model, wrap with manager and run a forward pass

def test_dataparallel():
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    try:
        mgr = create_multi_gpu_manager()
    except Exception as e:
        print('Could not create manager:', e)
        return
    print('Manager devices:', mgr.get_gpu_ids())
    model = torch.nn.Linear(10, 4)
    print('Model before wrap device of first param:', next(model.parameters()).device)
    wrapped = mgr.wrap_model_with_dataparallel(model)
    print('Wrapped type:', type(wrapped))
    try:
        x = torch.randn(2, 10)
        out = wrapped(x)
        print('Forward ok, out shape:', out.shape)
    except Exception as e:
        print('Forward failed:', e)

if __name__ == '__main__':
    test_dataparallel()
