"""LLM-free bug-hunting harness (fork-only).

``HUNT_CODE=/path/to/checkout`` runs the harness against another LightRAG
tree (a candidate fix, a closed PR). It must win over pytest's rootdir
insertion and the editable install, so it is put first on ``sys.path`` here,
before anything imports ``lightrag``, and checked.
"""

import os
import sys

_code = os.environ.get("HUNT_CODE")
if _code:
    _code = os.path.realpath(_code)
    if "lightrag" in sys.modules:
        raise RuntimeError("HUNT_CODE set but lightrag was imported before hunt")
    sys.path.insert(0, _code)
    import lightrag as _lr

    if not os.path.realpath(_lr.__file__).startswith(_code + os.sep):
        raise RuntimeError(f"HUNT_CODE={_code} but lightrag loaded from {_lr.__file__}")
