"""
Visualization utilities - Bridge module for backward compatibility.

This module re-exports visualization tools from the tools package.
Please update your imports to use the tools package directly:
    from tools.visualization import DetectionVisualizer
    from tools.gif_generator import FrameToGifConverter
    from tools.inference_pipeline import InferencePipeline
"""

import warnings

# Import from tools package
from tools.visualization import (
    DetectionVisualizer,
    load_kitti_calib
)

from tools.gif_generator import FrameToGifConverter

# InferencePipeline requires torch, so import it optionally
try:
    from tools.inference_pipeline import InferencePipeline
    _has_inference_pipeline = True
except ImportError:
    _has_inference_pipeline = False
    InferencePipeline = None

# Show deprecation warning
warnings.warn(
    "Importing from lib.helpers.visualization_utils is deprecated. "
    "Please use 'from tools.visualization import DetectionVisualizer' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'DetectionVisualizer',
    'FrameToGifConverter', 
    'load_kitti_calib'
]

if _has_inference_pipeline:
    __all__.append('InferencePipeline')
