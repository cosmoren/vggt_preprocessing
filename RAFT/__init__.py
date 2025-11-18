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
    ImagePairLoader,
    SequentialFrameLoader,
    DynamicReplicaLoader,
    CustomDatasetLoader
)

__all__ = [
    'ImagePreprocessor',
    'RAFTFlowEstimator',
    'RANSACAffineEstimator',
    'FlowProcessor',
    'DatasetLoader',
    'ImagePairLoader',
    'SequentialFrameLoader',
    'DynamicReplicaLoader',
    'CustomDatasetLoader',
]

