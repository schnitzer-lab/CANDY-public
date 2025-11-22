"""
CANDY: A neural dynamics modeling framework

This package provides tools for modeling neural dynamics and behavior decoding.
"""

__version__ = "0.1.0"
__author__ = "CANDY Team"

# Import main modules for easy access
from . import dynamics
from . import data_loader
from . import decoding
from . import runner
from . import plotting

__all__ = [
    "dynamics",
    "data_loader", 
    "decoding",
    "runner",
    "plotting",
]
