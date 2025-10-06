# top-level torch stub for static analysis
class Tensor: pass
def no_grad():
    class ctx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return ctx()

def cuda_is_available():
    return False
