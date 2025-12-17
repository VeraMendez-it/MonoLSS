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
from tools.inference_pipeline import InferencePipeline

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
    'InferencePipeline',
    'load_kitti_calib'
]
