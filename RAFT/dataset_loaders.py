"""
Example dataset loaders for different dataset structures.

You can extend these or create your own by inheriting from DatasetLoader.
"""

import torch
from torchvision.io import read_image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
try:
    from .preprocessing import DatasetLoader
except ImportError:
    from preprocessing import DatasetLoader


def load_image_rgb(image_path: str) -> torch.Tensor:
    """
    Load an image and convert to RGB format (3 channels).
    
    Handles both RGB (3 channels) and RGBA (4 channels) images.
    For RGBA images, composites alpha channel onto white background.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image tensor (C, H, W) with 3 channels, values in [0, 1]
    """
    img = read_image(str(image_path))  # (C, H, W), uint8
    
    # Convert to float and normalize to [0, 1]
    img = img.float() / 255.0
    
    # Handle RGBA images (4 channels)
    if img.shape[0] == 4:
        # Composite RGBA onto white background
        rgb = img[:3, :, :]  # (3, H, W)
        alpha = img[3:4, :, :]  # (1, H, W)
        # Composite: RGB * alpha + white * (1 - alpha)
        img = rgb * alpha + (1.0 - alpha)
    elif img.shape[0] == 3:
        # Already RGB, no conversion needed
        pass
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[0]}. Expected 3 (RGB) or 4 (RGBA).")
    
    return img


class ImagePairLoader(DatasetLoader):
    """Simple loader for image pairs from a directory."""
    
    def __init__(
        self,
        img1_dir: str,
        img2_dir: str,
        img1_pattern: str = "*.jpg",
        img2_pattern: str = "*.jpg",
        sort: bool = True
    ):
        """
        Args:
            img1_dir: Directory containing first images
            img2_dir: Directory containing second images
            img1_pattern: Glob pattern for first images
            img2_pattern: Glob pattern for second images
            sort: Whether to sort image paths
        """
        self.img1_dir = Path(img1_dir)
        self.img2_dir = Path(img2_dir)
        
        img1_paths = sorted(self.img1_dir.glob(img1_pattern)) if sort else list(self.img1_dir.glob(img1_pattern))
        img2_paths = sorted(self.img2_dir.glob(img2_pattern)) if sort else list(self.img2_dir.glob(img2_pattern))
        
        if len(img1_paths) != len(img2_paths):
            raise ValueError(f"Mismatch: {len(img1_paths)} img1 files vs {len(img2_paths)} img2 files")
        
        self.pairs = list(zip(img1_paths, img2_paths))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load a batch of image pairs."""
        img1_list = []
        img2_list = []
        metadata = {'img1_paths': [], 'img2_paths': []}
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
            img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
            
            img1_list.append(img1)
            img2_list.append(img2)
            metadata['img1_paths'].append(str(img1_path))
            metadata['img2_paths'].append(str(img2_path))
        
        # Stack into batches: (B, C, H, W)
        img1_batch = torch.stack(img1_list, dim=0)
        img2_batch = torch.stack(img2_list, dim=0)
        
        return img1_batch, img2_batch, metadata
    
    def get_all_pairs(self) -> List[Tuple[Path, Path]]:
        return self.pairs


class SequentialFrameLoader(DatasetLoader):
    """Loader for sequential frames from a single directory."""
    
    def __init__(
        self,
        frame_dir: str,
        frame_pattern: str = "frame_*.jpg",
        sort: bool = True
    ):
        """
        Args:
            frame_dir: Directory containing frames
            frame_pattern: Glob pattern for frames (should match frame_0.jpg, frame_1.jpg, etc.)
            sort: Whether to sort frame paths
        """
        self.frame_dir = Path(frame_dir)
        frame_paths = sorted(self.frame_dir.glob(frame_pattern)) if sort else list(self.frame_dir.glob(frame_pattern))
        
        if len(frame_paths) < 2:
            raise ValueError(f"Need at least 2 frames, got {len(frame_paths)}")
        
        self.frame_paths = frame_paths
        # Create pairs: (frame_0, frame_1), (frame_1, frame_2), ...
        self.pairs = list(zip(self.frame_paths[:-1], self.frame_paths[1:]))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load a batch of sequential frame pairs."""
        img1_list = []
        img2_list = []
        metadata = {'frame1_paths': [], 'frame2_paths': [], 'frame_indices': []}
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
            img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
            
            img1_list.append(img1)
            img2_list.append(img2)
            metadata['frame1_paths'].append(str(img1_path))
            metadata['frame2_paths'].append(str(img2_path))
            metadata['frame_indices'].append((idx, idx + 1))
        
        img1_batch = torch.stack(img1_list, dim=0)
        img2_batch = torch.stack(img2_list, dim=0)
        
        return img1_batch, img2_batch, metadata
    
    def get_all_pairs(self) -> List[Tuple[Path, Path]]:
        return self.pairs


class DynamicReplicaLoader(DatasetLoader):
    """
    Loader for DynamicReplica dataset.
    
    Each scene has images with naming pattern:
    - {scene_name}_left-{04d}.png
    - {scene_name}_right-{04d}.png
    
    Creates consecutive pairs separately for left and right images.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        camera_type: str = "left",  # "left", "right", or "both"
        scene_pattern: str = "*",
        sort: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing scene folders
            image_subdir: Subdirectory name containing images (default: "images")
            camera_type: Which camera to process ("left", "right", or "both")
            scene_pattern: Glob pattern for scene folders (default: "*")
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.camera_type = camera_type
        self.sort = sort
        
        # Find all scene directories
        scene_dirs = sorted(self.root_dir.glob(scene_pattern)) if sort else list(self.root_dir.glob(scene_pattern))
        scene_dirs = [d for d in scene_dirs if d.is_dir()]
        
        if len(scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {root_dir} with pattern {scene_pattern}")
        
        # Build pairs for each scene
        self.pairs = []
        self.scene_info = []  # Track which scene each pair belongs to
        
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            images_dir = scene_dir / image_subdir
            
            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                continue
            
            # Process left and/or right images
            if camera_type in ["left", "both"]:
                left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "left")
                self.pairs.extend(left_pairs)
                self.scene_info.extend([(scene_name, "left")] * len(left_pairs))
            
            if camera_type in ["right", "both"]:
                right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "right")
                self.pairs.extend(right_pairs)
                self.scene_info.extend([(scene_name, "right")] * len(right_pairs))
        
        print(f"DynamicReplicaLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
    def _get_consecutive_pairs(
        self,
        images_dir: Path,
        scene_name: str,
        camera: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get consecutive pairs for a specific camera type.
        
        Args:
            images_dir: Directory containing images
            scene_name: Name of the scene
            camera: "left" or "right"
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        # Pattern: {scene_name}_{camera}-{04d}.png
        pattern = f"{scene_name}_{camera}-*.png"
        image_paths = sorted(images_dir.glob(pattern)) if self.sort else list(images_dir.glob(pattern))
        
        if len(image_paths) < 2:
            print(f"Warning: Scene {scene_name} {camera} has only {len(image_paths)} images, need at least 2")
            return []
        
        # Create consecutive pairs: (0-1, 1-2, ..., n-2-n-1)
        pairs = list(zip(image_paths[:-1], image_paths[1:]))
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load a batch of consecutive image pairs."""
        img1_list = []
        img2_list = []
        metadata = {
            'img1_paths': [],
            'img2_paths': [],
            'scene_names': [],
            'camera_types': [],
            'pair_indices': []
        }
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            scene_name, camera_type = self.scene_info[idx]
            
            try:
                img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
                img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
                
                img1_list.append(img1)
                img2_list.append(img2)
                metadata['img1_paths'].append(str(img1_path))
                metadata['img2_paths'].append(str(img2_path))
                metadata['scene_names'].append(scene_name)
                metadata['camera_types'].append(camera_type)
                metadata['pair_indices'].append(idx)
            except Exception as e:
                print(f"Error loading pair {idx} ({img1_path}, {img2_path}): {e}")
                raise
        
        # Stack into batches: (B, C, H, W)
        img1_batch = torch.stack(img1_list, dim=0)
        img2_batch = torch.stack(img2_list, dim=0)
        
        return img1_batch, img2_batch, metadata
    
    def get_all_pairs(self) -> List[Tuple[Path, Path]]:
        return self.pairs
    
    def get_scene_info(self) -> List[Tuple[str, str]]:
        """Get scene and camera type for each pair."""
        return self.scene_info


class CustomDatasetLoader(DatasetLoader):
    """
    Template for creating custom dataset loaders.
    
    Inherit from this class and implement the required methods.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize your dataset here."""
        # Your initialization code
        pass
    
    def __len__(self) -> int:
        """Return the number of image pairs."""
        raise NotImplementedError
    
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load a batch of image pairs.
        
        Returns:
            (img1_batch, img2_batch, metadata)
            - img1_batch: (B, C, H, W) tensor with values in [0, 1]
            - img2_batch: (B, C, H, W) tensor with values in [0, 1]
            - metadata: Dict with any additional information
        """
        raise NotImplementedError
    
    def get_all_pairs(self) -> List[Tuple[Any, Any]]:
        """Return list of all image pairs."""
        raise NotImplementedError

