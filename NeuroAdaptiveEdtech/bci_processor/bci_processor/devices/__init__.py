"""
Device interfaces for various BCI hardware and data sources.
"""
from .base_device import BaseDevice
from .matlab_device import MatlabDevice
from .muse_device import MuseDevice

__all__ = ['BaseDevice', 'MatlabDevice', 'MuseDevice']
