"""Shim to expose the standard-library http.HTTPStatus.

This repository includes a lightweight `http.py` that shadows the stdlib
`http` package and can break third-party imports. We load the stdlib http
package explicitly from the Python installation and re-export HTTPStatus.
"""

import importlib.util
import sysconfig
import os
import importlib

_stdlib = None
try:
    stdlib_dir = sysconfig.get_paths().get('stdlib')
    if stdlib_dir:
        candidate = os.path.join(stdlib_dir, 'http', '__init__.py')
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location('_stdlib_http', candidate)
            _stdlib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_stdlib)
except Exception:
    _stdlib = None

if _stdlib is None:
    # Fallback to normal import; should import stdlib if available
    _stdlib = importlib.import_module('http')

# Re-export just HTTPStatus
try:
    HTTPStatus = _stdlib.HTTPStatus
except Exception:
    # As a last resort, provide a minimal fallback
    class HTTPStatus:
        OK = 200
        NOT_FOUND = 404

__all__ = ['HTTPStatus']