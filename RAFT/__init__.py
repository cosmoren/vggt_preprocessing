"""
RAFT Optical Flow Preprocessing Module

This module provides tools for:
- Optical flow estimation using RAFT
- Rigid/residual flow decomposition using RANSAC
- Batch processing of datasets
"""

from .preprocessing import (
    ImagePreprocessor,
    RAFTFlowEstimator,
    RANSACAffineEstimator,
    FlowProcessor,
    DatasetLoader
)

from .dataset_loaders import (
    DynamicReplicaLoader,
    CustomDatasetLoader
)

from .data_preprocessors import (
    DynamicReplicaProcessor
)

__all__ = [
    'ImagePreprocessor',
    'RAFTFlowEstimator',
    'RANSACAffineEstimator',
    'FlowProcessor',
    'DatasetLoader',
    'DynamicReplicaLoader',
    'CustomDatasetLoader',
    'DynamicReplicaProcessor',
]

