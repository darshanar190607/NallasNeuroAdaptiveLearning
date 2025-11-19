"""
BCI Processor - A Python package for processing BCI data from various sources.
"""
from .core import BCIProcessor
from .devices import MatlabDevice, MuseDevice  # Add other devices as needed

__version__ = '0.1.0'
__all__ = ['BCIProcessor', 'MatlabDevice', 'MuseDevice']
