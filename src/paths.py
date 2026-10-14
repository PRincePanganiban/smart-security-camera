"""
paths.py
--------
Finds files both when running as a normal Python script and when packaged
into a Windows .exe with PyInstaller.

  resource_path() -> read-only files bundled with the app (e.g. the Haar cascade)
  app_dir()       -> folder where the app writes results (snapshots, logs, screenshots)
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_frozen():
    """True when running from the packaged .exe."""
    return getattr(sys, "frozen", False)


def resource_path(relative):
    """Path to a bundled read-only resource."""
    base = getattr(sys, "_MEIPASS", _PROJECT_ROOT)
    return os.path.join(base, relative)


def app_dir():
    """Folder next to the .exe (or the project folder) for output files."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return _PROJECT_ROOT
