"""PyXDM: Python package for XDM multipole moment calculations."""

import importlib.metadata

__version__ = importlib.metadata.version(__name__)

from .core import XDMCalculator, XDMSession
from .grids import CustomGrid, load_mesh
from .logger import logger

__all__ = ["XDMSession", "XDMCalculator", "CustomGrid", "load_mesh", "logger"]
