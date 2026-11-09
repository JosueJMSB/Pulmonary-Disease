"""Versiones de librerias, CPU, memoria y carga del sistema.

Nunca importa PyTorch: las versiones se leen con ``importlib.metadata``.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata

PACKAGES = (
    "numpy", "scipy", "pandas", "soundfile", "librosa", "scikit-learn",
    "onnxruntime", "onnx", "psutil", "torch",
)


def package_versions(packages=PACKAGES) -> dict:
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _total_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return None


def system_load() -> dict:
    load = {"loadavg_1_5_15": None, "cpu_percent": None, "available_memory_bytes": None}
    if hasattr(os, "getloadavg"):
        try:
            load["loadavg_1_5_15"] = [round(v, 3) for v in os.getloadavg()]
        except OSError:
            pass
    try:
        import psutil

        load["cpu_percent"] = psutil.cpu_percent(interval=1.0)
        load["available_memory_bytes"] = int(psutil.virtual_memory().available)
    except ImportError:
        pass
    return load


def collect() -> dict:
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": physical,
        "total_memory_bytes": _total_memory_bytes(),
        "packages": package_versions(),
        "torch_imported_in_this_process": "torch" in sys.modules,
    }
