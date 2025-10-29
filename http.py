"""Compatibility shim which exposes the real standard-library `http` package.

Problems this fixes:
- If a top-level `http.py` exists in the repository it shadows the stdlib
  `http` package. Many third-party packages (urllib, httpx, gradio, etc.)
  expect `http` to be a package with submodules like `http.client`.

This shim locates the stdlib `http` package and loads it, then replaces
the current module entry in sys.modules so subsequent `import http.client`
works as expected.

If for some reason the stdlib package cannot be located, a very small
fallback is provided to avoid hard crashes, but functionality will be
limited.
"""

import importlib.util
import importlib
import sys
import sysconfig
import os

def _load_stdlib_http():
    try:
        stdlib_dir = sysconfig.get_paths().get('stdlib')
        if not stdlib_dir:
            return None
        candidate = os.path.join(stdlib_dir, 'http', '__init__.py')
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location('_stdlib_http', candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception:
        return None
    return None


_stdlib_http = _load_stdlib_http()
if _stdlib_http is None:
    # As a last resort, try a normal import, but avoid importing this file again.
    try:
        # importlib.import_module may return this same module if resolution finds
        # the repo-local file first; guard against that by checking __file__.
        mod = importlib.import_module('http')
        if getattr(mod, '__file__', '') == os.path.abspath(__file__):
            _stdlib_http = None
        else:
            _stdlib_http = mod
    except Exception:
        _stdlib_http = None

if _stdlib_http is not None:
    # Replace sys.modules entry so future imports (http.client) resolve to stdlib
    sys.modules['http'] = _stdlib_http
    # Also populate current module namespace with stdlib attributes for compatibility
    for k, v in vars(_stdlib_http).items():
        if not k.startswith('__'):
            globals()[k] = v
    __all__ = [k for k in vars(_stdlib_http) if not k.startswith('__')]
else:
    # Minimal fallback to avoid immediate ImportError; limited functionality.
    class HTTPStatus:
        OK = 200
        NOT_FOUND = 404

    __all__ = ['HTTPStatus']