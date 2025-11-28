"""
Example dataset loaders for different dataset structures.

You can extend these or create your own by inheriting from DatasetLoader.
"""

import torch
import torch.nn.functional as F
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

class UnrealStereo4KLoader(DatasetLoader):
    """
    Loader for UnrealStereo4K dataset.
    
    Each scene has images with naming pattern:
    - {05d}_cam0.png
    - {05d}_cam1.png
    
    Creates consecutive pairs separately for left and right images.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        scene_pattern: Optional[str] = None,
        scene_name: Optional[str] = None,
        sort: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing scene folders
            image_subdir: Subdirectory name containing images (default: "images")
            camera_type: Which camera to process ("left", "right", or "both")
            scene_pattern: Glob pattern for scene folders (default: "*", ignored if scene_name is provided)
            scene_name: Direct scene name for efficient single-scene loading (takes precedence over scene_pattern)
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        
        # Build pairs for each scene
        self.pairs = []
        self.scene_info = []  # Track which scene each pair belongs to
        
        # If scene_name is provided, use direct path (more efficient)
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Process left and right images
            left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "0")
            self.pairs.extend(left_pairs)
            self.scene_info.extend([(scene_name, "0")] * len(left_pairs))
            

            right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "1")
            self.pairs.extend(right_pairs)
            self.scene_info.extend([(scene_name, "1")] * len(right_pairs))
            
            print(f"UnrealStereo4KLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Use scene_pattern (backward compatibility)
            if scene_pattern is None:
                scene_pattern = "*"
            
            # Find all scene directories
            scene_dirs = sorted(self.root_dir.glob(scene_pattern)) if sort else list(self.root_dir.glob(scene_pattern))
            scene_dirs = [d for d in scene_dirs if d.is_dir()]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No scene directories found in {root_dir} with pattern {scene_pattern}")
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Process left and/or right images
                #if camera_type in ["left", "both"]:
                #    left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "left")
                #    self.pairs.extend(left_pairs)
                #    self.scene_info.extend([(scene_name, "left")] * len(left_pairs))
                
                #if camera_type in ["right", "both"]:
                #    right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "right")
                #    self.pairs.extend(right_pairs)
                #    self.scene_info.extend([(scene_name, "right")] * len(right_pairs))
            
            print(f"SpringLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")

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
        pattern = f"*_cam{camera}.png"
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


class SpringLoader(DatasetLoader):
    """
    Loader for DynamicReplica dataset.
    
    Each scene has images with naming pattern:
    - frame_left_{04d}.png
    - frame_right_{04d}.png
    
    Creates consecutive pairs separately for left and right images.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        scene_pattern: Optional[str] = None,
        scene_name: Optional[str] = None,
        sort: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing scene folders
            image_subdir: Subdirectory name containing images (default: "images")
            camera_type: Which camera to process ("left", "right", or "both")
            scene_pattern: Glob pattern for scene folders (default: "*", ignored if scene_name is provided)
            scene_name: Direct scene name for efficient single-scene loading (takes precedence over scene_pattern)
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        
        # Build pairs for each scene
        self.pairs = []
        self.scene_info = []  # Track which scene each pair belongs to
        
        # If scene_name is provided, use direct path (more efficient)
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Process left and right images
            left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "left")
            self.pairs.extend(left_pairs)
            self.scene_info.extend([(scene_name, "left")] * len(left_pairs))
            

            right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "right")
            self.pairs.extend(right_pairs)
            self.scene_info.extend([(scene_name, "right")] * len(right_pairs))
            
            print(f"SpringLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Use scene_pattern (backward compatibility)
            if scene_pattern is None:
                scene_pattern = "*"
            
            # Find all scene directories
            scene_dirs = sorted(self.root_dir.glob(scene_pattern)) if sort else list(self.root_dir.glob(scene_pattern))
            scene_dirs = [d for d in scene_dirs if d.is_dir()]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No scene directories found in {root_dir} with pattern {scene_pattern}")
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Process left and/or right images
                #if camera_type in ["left", "both"]:
                #    left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "left")
                #    self.pairs.extend(left_pairs)
                #    self.scene_info.extend([(scene_name, "left")] * len(left_pairs))
                
                #if camera_type in ["right", "both"]:
                #    right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "right")
                #    self.pairs.extend(right_pairs)
                #    self.scene_info.extend([(scene_name, "right")] * len(right_pairs))
            
            print(f"SpringLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")

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
        pattern = f"frame_{camera}_*.png"
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
        scene_pattern: Optional[str] = None,
        scene_name: Optional[str] = None,
        sort: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing scene folders
            image_subdir: Subdirectory name containing images (default: "images")
            camera_type: Which camera to process ("left", "right", or "both")
            scene_pattern: Glob pattern for scene folders (default: "*", ignored if scene_name is provided)
            scene_name: Direct scene name for efficient single-scene loading (takes precedence over scene_pattern)
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.camera_type = camera_type
        self.sort = sort
        
        # Build pairs for each scene
        self.pairs = []
        self.scene_info = []  # Track which scene each pair belongs to
        
        # If scene_name is provided, use direct path (more efficient)
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Process left and/or right images
            if camera_type in ["left", "both"]:
                left_pairs = self._get_consecutive_pairs(images_dir, scene_name, "left")
                self.pairs.extend(left_pairs)
                self.scene_info.extend([(scene_name, "left")] * len(left_pairs))
            
            if camera_type in ["right", "both"]:
                right_pairs = self._get_consecutive_pairs(images_dir, scene_name, "right")
                self.pairs.extend(right_pairs)
                self.scene_info.extend([(scene_name, "right")] * len(right_pairs))
            
            # print(f"DynamicReplicaLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Use scene_pattern (backward compatibility)
            if scene_pattern is None:
                scene_pattern = "*"
            
            # Find all scene directories
            scene_dirs = sorted(self.root_dir.glob(scene_pattern)) if sort else list(self.root_dir.glob(scene_pattern))
            scene_dirs = [d for d in scene_dirs if d.is_dir()]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No scene directories found in {root_dir} with pattern {scene_pattern}")
            
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
            
            # print(f"DynamicReplicaLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
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


class MVSSynthLoader(DatasetLoader):
    """
    Loader for MVS_Synth dataset.
    
    Each scene has images with naming pattern:
    - {04d}/images/{04d}.png
    
    Creates consecutive pairs for images within each scene.
    No left/right camera distinction.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        sort: bool = True,
        scene_name: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing scene folders (e.g., /mnt/nfs/SpatialAI/wai_datasets/mvs_synth)
            image_subdir: Subdirectory name containing images (default: "images")
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        self.pairs = []
        self.scene_info = []
        
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Get consecutive pairs for this scene
            scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
            self.pairs.extend(scene_pairs)
            self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"MVSSynthLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Find all scene directories (4-digit folders)
            # Use * pattern and filter for 4-digit directories
            all_dirs = sorted(self.root_dir.glob("*")) if sort else list(self.root_dir.glob("*"))
            scene_dirs = [d for d in all_dirs if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No 4-digit scene directories found in {root_dir}")
            
            # Build pairs for each scene
            self.pairs = []
            self.scene_info = []  # Track which scene each pair belongs to
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Get consecutive pairs for this scene
                scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
                self.pairs.extend(scene_pairs)
                self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"MVSSynthLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
    def _get_consecutive_pairs(
        self,
        images_dir: Path,
        scene_name: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get consecutive pairs for a scene.
        
        Args:
            images_dir: Directory containing images
            scene_name: Name of the scene
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        # Pattern: {04d}.png (4-digit number with .png extension)
        # Use * pattern and filter for 4-digit .png files
        all_files = sorted(images_dir.glob("*.png")) if self.sort else list(images_dir.glob("*.png"))
        image_paths = [f for f in all_files if f.stem.isdigit() and len(f.stem) == 4]
        
        if len(image_paths) < 2:
            print(f"Warning: Scene {scene_name} has only {len(image_paths)} images, need at least 2")
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
            'pair_indices': []
        }
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            scene_name = self.scene_info[idx]
            
            try:
                img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
                img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
                
                img1_list.append(img1)
                img2_list.append(img2)
                metadata['img1_paths'].append(str(img1_path))
                metadata['img2_paths'].append(str(img2_path))
                metadata['scene_names'].append(scene_name)
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
    
    def get_scene_info(self) -> List[str]:
        """Get scene name for each pair."""
        return self.scene_info



class Eth3DLoader(DatasetLoader):
    """
    Loader for ETH3D dataset.
    
    Each scene has images with naming pattern:
    - {scene_name}/images/DSC_{04d}.png
    
    Creates consecutive pairs for images within each scene.
    No left/right camera distinction.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        sort: bool = True,
        scene_name: Optional[str] = None,
        downsample_factor: int = 4
    ):
        """
        Args:
            root_dir: Root directory containing scene folders (e.g., /mnt/nfs/SpatialAI/wai_datasets/mvs_synth)
            image_subdir: Subdirectory name containing images (default: "images")
            sort: Whether to sort scenes and images
            scene_name: Optional scene name to load
            downsample_factor: Downsampling factor (e.g., 4 for 1/4, 16 for 1/16 resolution). Default: 1 (no downsampling)
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        self.downsample_factor = downsample_factor
        self.pairs = []
        self.scene_info = []
        
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Get consecutive pairs for this scene
            scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
            self.pairs.extend(scene_pairs)
            self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Eth3DLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Find all scene directories (4-digit folders)
            # Use * pattern and filter for 4-digit directories
            all_dirs = sorted(self.root_dir.glob("*")) if sort else list(self.root_dir.glob("*"))
            scene_dirs = [d for d in all_dirs if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No 4-digit scene directories found in {root_dir}")
            
            # Build pairs for each scene
            self.pairs = []
            self.scene_info = []  # Track which scene each pair belongs to
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Get consecutive pairs for this scene
                scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
                self.pairs.extend(scene_pairs)
                self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Eth3DLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
    def _get_consecutive_pairs(
        self,
        images_dir: Path,
        scene_name: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get consecutive pairs for a scene.
        
        Args:
            images_dir: Directory containing images
            scene_name: Name of the scene
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        # Pattern: DSC_{04d}.png (4-digit number with .png extension)
        # Use * pattern and filter for 4-digit .png files
        all_files = sorted(images_dir.glob("*.png")) if self.sort else list(images_dir.glob("*.png"))
        image_paths = [f for f in all_files]
        
        
        if len(image_paths) < 2:
            print(f"Warning: Scene {scene_name} has only {len(image_paths)} images, need at least 2")
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
            'pair_indices': []
        }
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            scene_name = self.scene_info[idx]
            
            try:
                img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
                img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
                
                # Downsample individual images if factor > 1
                if self.downsample_factor > 1:
                    C, H, W = img1.shape
                    new_H = max(1, H // self.downsample_factor)
                    new_W = max(1, W // self.downsample_factor)
                    
                    if new_H < H or new_W < W:
                        # Add batch dimension for interpolation: (1, C, H, W)
                        print(f"debug: img1.shape: {img1.shape}")
                        print(f"debug: unsqueeze(0).shape: {img1.unsqueeze(0).shape}")
                        img1 = F.interpolate(
                            img1.unsqueeze(0), size=(new_H, new_W),
                            mode='area',
                            #antialias=True
                        ).squeeze(0)  # Remove batch dimension: (C, H, W)
                        img2 = F.interpolate(
                            img2.unsqueeze(0), size=(new_H, new_W),
                            mode='area',
                            #antialias=True
                        ).squeeze(0)  # Remove batch dimension: (C, H, W)
                
                img1_list.append(img1)
                img2_list.append(img2)
                metadata['img1_paths'].append(str(img1_path))
                metadata['img2_paths'].append(str(img2_path))
                metadata['scene_names'].append(scene_name)
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
    
    def get_scene_info(self) -> List[str]:
        """Get scene name for each pair."""
        return self.scene_info

class ScannetppV2Loader(DatasetLoader):
    """
    Loader for ScannetppV2 dataset.
    
    Each scene has images with naming pattern:
    - {scene_name}/images/DSC{05d}.jpg
    
    Creates consecutive pairs for images within each scene.
    No left/right camera distinction.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        sort: bool = True,
        scene_name: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing scene folders (e.g., /mnt/nfs/SpatialAI/wai_datasets/mvs_synth)
            image_subdir: Subdirectory name containing images (default: "images")
            sort: Whether to sort scenes and images
            scene_name: Optional scene name to load
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        self.pairs = []
        self.scene_info = []
        
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Get consecutive pairs for this scene
            scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
            self.pairs.extend(scene_pairs)
            self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Eth3DLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Find all scene directories (4-digit folders)
            # Use * pattern and filter for 4-digit directories
            all_dirs = sorted(self.root_dir.glob("*")) if sort else list(self.root_dir.glob("*"))
            scene_dirs = [d for d in all_dirs if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No 4-digit scene directories found in {root_dir}")
            
            # Build pairs for each scene
            self.pairs = []
            self.scene_info = []  # Track which scene each pair belongs to
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Get consecutive pairs for this scene
                scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
                self.pairs.extend(scene_pairs)
                self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Eth3DLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
    def _get_consecutive_pairs(
        self,
        images_dir: Path,
        scene_name: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get consecutive pairs for a scene.
        
        Args:
            images_dir: Directory containing images
            scene_name: Name of the scene
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        # Pattern: DSC_{04d}.jpg (4-digit number with .jpg extension)
        # Use * pattern and filter for 4-digit .jpg files
        all_files = sorted(images_dir.glob("*.jpg")) if self.sort else list(images_dir.glob("*.jpg"))
        image_paths = [f for f in all_files]
        
        
        if len(image_paths) < 2:
            print(f"Warning: Scene {scene_name} has only {len(image_paths)} images, need at least 2")
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
            'pair_indices': []
        }
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            scene_name = self.scene_info[idx]
            
            try:
                img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
                img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
                
                img1_list.append(img1)
                img2_list.append(img2)
                metadata['img1_paths'].append(str(img1_path))
                metadata['img2_paths'].append(str(img2_path))
                metadata['scene_names'].append(scene_name)
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
    
    def get_scene_info(self) -> List[str]:
        """Get scene name for each pair."""
        return self.scene_info




class Sailvos3dLoader(DatasetLoader):
    """
    Loader for Sailvos3d dataset.
    
    Each scene has images with naming pattern:
    -scene_name/images/{06d}.png
    
    Creates consecutive pairs for images within each scene.
    No left/right camera distinction.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_subdir: str = "images",
        sort: bool = True,
        scene_name: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing scene folders (e.g., /mnt/nfs/SpatialAI/wai_datasets/sailivos3d)
            image_subdir: Subdirectory name containing images (default: "images")
            sort: Whether to sort scenes and images
        """
        self.root_dir = Path(root_dir)
        self.image_subdir = image_subdir
        self.sort = sort
        self.pairs = []
        self.scene_info = []
        
        if scene_name is not None:
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")
            
            images_dir = scene_dir / image_subdir
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Get consecutive pairs for this scene
            scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
            self.pairs.extend(scene_pairs)
            self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Sailvos3dLoader: Found {len(self.pairs)} image pairs for scene {scene_name}")
        else:
            # Find all scene directories (4-digit folders)
            # Use * pattern and filter for 4-digit directories
            all_dirs = sorted(self.root_dir.glob("*")) if sort else list(self.root_dir.glob("*"))
            scene_dirs = [d for d in all_dirs if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
            
            if len(scene_dirs) == 0:
                raise ValueError(f"No 4-digit scene directories found in {root_dir}")
            
            # Build pairs for each scene
            self.pairs = []
            self.scene_info = []  # Track which scene each pair belongs to
            
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                images_dir = scene_dir / image_subdir
                
                if not images_dir.exists():
                    print(f"Warning: {images_dir} does not exist, skipping scene {scene_name}")
                    continue
                
                # Get consecutive pairs for this scene
                scene_pairs = self._get_consecutive_pairs(images_dir, scene_name)
                self.pairs.extend(scene_pairs)
                self.scene_info.extend([scene_name] * len(scene_pairs))
            
            print(f"Sailvos3dLoader: Found {len(self.pairs)} image pairs across {len(scene_dirs)} scenes")
    
    def _get_consecutive_pairs(
        self,
        images_dir: Path,
        scene_name: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get consecutive pairs for a scene.
        
        Args:
            images_dir: Directory containing images
            scene_name: Name of the scene
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        # Pattern: {04d}.png (4-digit number with .png extension)
        # Use * pattern and filter for 4-digit .png files
        all_files = sorted(images_dir.glob("*.png")) if self.sort else list(images_dir.glob("*.png"))
        image_paths = [f for f in all_files if f.stem.isdigit() and len(f.stem) == 6]
        
        if len(image_paths) < 2:
            print(f"Warning: Scene {scene_name} has only {len(image_paths)} images, need at least 2")
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
            'pair_indices': []
        }
        
        for idx in indices:
            img1_path, img2_path = self.pairs[idx]
            scene_name = self.scene_info[idx]
            
            try:
                img1 = load_image_rgb(str(img1_path))  # (3, H, W), float [0, 1]
                img2 = load_image_rgb(str(img2_path))  # (3, H, W), float [0, 1]
                
                img1_list.append(img1)
                img2_list.append(img2)
                metadata['img1_paths'].append(str(img1_path))
                metadata['img2_paths'].append(str(img2_path))
                metadata['scene_names'].append(scene_name)
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
    
    def get_scene_info(self) -> List[str]:
        """Get scene name for each pair."""
        return self.scene_info