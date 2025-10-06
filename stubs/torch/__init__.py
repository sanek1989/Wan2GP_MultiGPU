# Lightweight stub for torch
class _Tensor: 
    pass

def cuda_is_available():
    return False

def no_grad():
    class ctx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return ctx()

__all__ = ['no_grad', 'cuda_is_available', 'Tensor']
