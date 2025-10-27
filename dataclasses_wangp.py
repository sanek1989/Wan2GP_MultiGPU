"""Compatibility shim that delegates to the real standard-library `dataclasses`.

When a repository includes a top-level `dataclasses.py`, it shadows the stdlib
module and breaks third-party libraries (for example: `huggingface_hub` which
imports `dataclasses.asdict`). We resolve that by locating the stdlib's
`dataclasses` implementation and loading it explicitly from the Python
installation, then re-exporting its key symbols.
"""

import importlib.util
import sys
import os
import sysconfig

# Attempt to locate stdlib 'dataclasses' by file path
_stdlib = None
try:
    stdlib_dir = sysconfig.get_paths().get('stdlib')
    if stdlib_dir:
        candidate = os.path.join(stdlib_dir, 'dataclasses.py')
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location('_stdlib_dataclasses', candidate)
            _stdlib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_stdlib)
except Exception:
    _stdlib = None

# Fallback: try normal import but guard against circular import
if _stdlib is None:
    try:
        # this may still pick up this shim if stdlib cannot be found; catch recursion
        import importlib
        _stdlib = importlib.import_module('dataclasses')
        if getattr(_stdlib, '__file__', None) and os.path.abspath(_stdlib.__file__) == os.path.abspath(__file__):
            # we appear to have imported this file again, reset _stdlib to None
            _stdlib = None
    except Exception:
        _stdlib = None

if _stdlib is None:
    raise ImportError('Could not locate the standard library dataclasses module; please remove or rename the repository top-level dataclasses.py')

# Re-export commonly used symbols
dataclass = _stdlib.dataclass
field = _stdlib.field
asdict = _stdlib.asdict
astuple = _stdlib.astuple
make_dataclass = _stdlib.make_dataclass

__all__ = ['dataclass', 'field', 'asdict', 'astuple', 'make_dataclass']