"""Disable OGB 1.3.x's optional online version check during offline runs.

OGB imports `ogb.version`, which starts non-daemon HTTP update-check threads;
these can keep an otherwise finished training process alive indefinitely.
Use the installed distribution's actual version, without importing that module's
network side effects. No dataset, model, or training operation is replaced.
"""
from importlib.metadata import PackageNotFoundError, version
import os
import sys
from types import ModuleType


def configure_offline_runtime():
    if os.environ.get("HF_HUB_OFFLINE") != "1" or "ogb.version" in sys.modules:
        return False
    try:
        installed = version("ogb")
    except PackageNotFoundError:
        return False
    module = ModuleType("ogb.version", "Installed version; online check disabled for offline training")
    module.__version__ = installed
    sys.modules["ogb.version"] = module
    return True
