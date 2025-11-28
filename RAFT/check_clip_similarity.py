"""
Script to check CLIP similarity for processed optical flow pairs.

For each .pt file in the output directory, this script:
1. Reconstructs the original image pair paths
2. Computes CLIP similarity between the two images
3. Logs pairs with low similarity to clip_low_conf.txt
"""

import torch
import clip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

# Import dataset loaders to understand naming patterns
try:
    from dataset_loaders import (
        DynamicReplicaLoader, MVSSynthLoader, UnrealStereo4KLoader,
        SpringLoader, Sailvos3dLoader
    )
except ImportError:
    from .dataset_loaders import (
        DynamicReplicaLoader, MVSSynthLoader, UnrealStereo4KLoader,
        SpringLoader, Sailvos3dLoader
    )


def get_image_pair_paths(
    pt_file: Path,
    dataset_type: str,
    original_dataset_root: str,
    image_subdir: str = "images"
) -> Optional[Tuple[Path, Path]]:
    """
    Reconstruct the original image pair paths from a .pt file path.
    
    Args:
        pt_file: Path to the .pt file (e.g., /path/to/output/scene_name/img1_stem.pt)
        dataset_type: Type of dataset (dynamicreplica, mvs_synth, unrealstereo4k, spring, sailvos3d)
        original_dataset_root: Root directory of the original dataset
        image_subdir: Subdirectory containing images (default: "images")
    
    Returns:
        Tuple of (img1_path, img2_path) or None if reconstruction fails
    """
    original_root = Path(original_dataset_root)
    
    # Extract scene name and image1 stem from .pt file path
    # pt_file structure: output_dir/scene_name/img1_stem.pt
    scene_name = pt_file.parent.name
    img1_stem = pt_file.stem  # filename without .pt extension
    
    scene_dir = original_root / dataset_type / scene_name
    images_dir = scene_dir / image_subdir
    
    if not images_dir.exists():
        return None
    
    # Reconstruct image paths based on dataset type
    if dataset_type == "dynamicreplica":
        # Pattern: {scene_name}_left-{04d}.png or {scene_name}_right-{04d}.png
        # img1_stem is the full stem like "009850-3_obj_source_left-0000"
        # The .pt filename is exactly img1_path.stem + '.pt'
        
        # Directly construct img1_path from stem
        img1_path = images_dir / f"{img1_stem}.png"
        
        if not img1_path.exists():
            # Try with different extensions
            for ext in [".png", ".jpg", ".PNG", ".JPG"]:
                test_path = images_dir / f"{img1_stem}{ext}"
                if test_path.exists():
                    img1_path = test_path
                    break
            else:
                return None
        
        # Extract frame number from the end of the stem
        # Format: ..._left-0000 or ..._right-0000
        # The last part after the last '-' should be the 4-digit frame number
        parts = img1_stem.split('-')
        if len(parts) < 2:
            return None
        
        try:
            # Last part should be the frame number (4 digits)
            frame_num_str = parts[-1]
            if len(frame_num_str) == 4 and frame_num_str.isdigit():
                frame_num = int(frame_num_str)
                next_frame_num = frame_num + 1
                
                # Reconstruct stem for next frame
                base_parts = parts[:-1]
                next_stem = '-'.join(base_parts) + f'-{next_frame_num:04d}'
                img2_path = images_dir / f"{next_stem}.png"
                
                if not img2_path.exists():
                    # Try with same extension as img1
                    img2_path = images_dir / f"{next_stem}{img1_path.suffix}"
                
                if img2_path.exists():
                    return img1_path, img2_path
        except (ValueError, IndexError):
            pass
        
        return None
    
    elif dataset_type == "mvs_synth":
        # Pattern: {04d}.png (4-digit number)
        try:
            frame_num = int(img1_stem)
            next_frame_num = frame_num + 1
            
            img1_path = images_dir / f"{frame_num:04d}.png"
            img2_path = images_dir / f"{next_frame_num:04d}.png"
            
            if not img1_path.exists() or not img2_path.exists():
                return None
            
            return img1_path, img2_path
        except ValueError:
            return None
    
    elif dataset_type == "unrealstereo4k":
        # Pattern: {05d}_cam0.png or {05d}_cam1.png
        # Extract frame number and camera from stem
        # img1_stem should be like "00001_cam0"
        parts = img1_stem.split('_cam')
        if len(parts) != 2:
            return None
        
        try:
            frame_num = int(parts[0])
            camera = parts[1]
            next_frame_num = frame_num + 1
            
            img1_path = images_dir / f"{frame_num:05d}_cam{camera}.png"
            img2_path = images_dir / f"{next_frame_num:05d}_cam{camera}.png"
            
            if not img1_path.exists() or not img2_path.exists():
                return None
            
            return img1_path, img2_path
        except (ValueError, IndexError):
            return None
    
    elif dataset_type == "spring":
        # Pattern: frame_left_{04d}.png or frame_right_{04d}.png
        # img1_stem should be like "frame_left_0000"
        if not img1_stem.startswith("frame_"):
            return None
        
        parts = img1_stem.split('_')
        if len(parts) < 3:
            return None
        
        camera = parts[1]  # "left" or "right"
        try:
            frame_num = int(parts[2])
            next_frame_num = frame_num + 1
            
            img1_path = images_dir / f"frame_{camera}_{frame_num:04d}.png"
            img2_path = images_dir / f"frame_{camera}_{next_frame_num:04d}.png"
            
            if not img1_path.exists() or not img2_path.exists():
                return None
            
            return img1_path, img2_path
        except (ValueError, IndexError):
            return None
    
    elif dataset_type == "sailvos3d":
        # Pattern: {06d}.png (6-digit number)
        try:
            frame_num = int(img1_stem)
            next_frame_num = frame_num + 1
            
            img1_path = images_dir / f"{frame_num:06d}.png"
            img2_path = images_dir / f"{next_frame_num:06d}.png"
            
            if not img1_path.exists() or not img2_path.exists():
                return None
            
            return img1_path, img2_path
        except ValueError:
            return None
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return None


def compute_clip_similarity(
    img1_path: Path,
    img2_path: Path,
    model,
    preprocess,
    device: str = "cuda"
) -> float:
    """
    Compute CLIP similarity between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device to run on
    
    Returns:
        Similarity score (cosine similarity) between -1 and 1
    """
    try:
        # Load and preprocess images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            img1_features = model.encode_image(img1_tensor)
            img2_features = model.encode_image(img2_tensor)
            
            # Normalize features
            img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
            img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (img1_features * img2_features).sum(dim=-1).item()
        
        return similarity
    except Exception as e:
        print(f"Error computing similarity for {img1_path} and {img2_path}: {e}")
        return None


def detect_dataset_type_from_path(output_dir: str, original_dataset_root: str) -> Optional[str]:
    """
    Try to detect dataset type from directory structure.
    
    Args:
        output_dir: Output directory containing .pt files
        original_dataset_root: Original dataset root directory
    
    Returns:
        Detected dataset type or None
    """
    output_path = Path(output_dir)
    original_root = Path(original_dataset_root)
    
    # Get a sample scene directory
    scene_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    if not scene_dirs:
        return None
    
    sample_scene = scene_dirs[0].name
    scene_path = original_root / sample_scene / "images"
    
    if not scene_path.exists():
        return None
    
    # Check for dataset-specific patterns
    files = list(scene_path.glob("*.png"))
    if not files:
        return None
    
    sample_file = files[0].name
    
    # Check patterns
    if "_left-" in sample_file or "_right-" in sample_file:
        return "dynamicreplica"
    elif "_cam0" in sample_file or "_cam1" in sample_file:
        return "unrealstereo4k"
    elif sample_file.startswith("frame_left_") or sample_file.startswith("frame_right_"):
        return "spring"
    elif len(sample_file) == 8 and sample_file[:4].isdigit():  # 4-digit .png
        return "mvs_synth"
    elif len(sample_file) == 10 and sample_file[:6].isdigit():  # 6-digit .png
        return "sailvos3d"
    
    return None


def check_clip_similarity_for_dataset(
    dataset_type: Optional[str] = None,
    similarity_threshold: float = 0.85,
    image_subdir: str = "images",
    device: str = "cuda"
):
    """
    Check CLIP similarity for all .pt files in the output directory.
    
    Args:
        output_dir: Directory containing processed .pt files (e.g., /mnt/nfs/SpatialAI/sq/optical_flow_datavggt_ft_optical_flow/<dataset>)
        dataset_type: Type of dataset (dynamicreplica, mvs_synth, unrealstereo4k, spring, sailvos3d).
                     If None, will try to auto-detect.
        original_dataset_root: Root directory of the original dataset.
                              If None, will try to infer from output_dir.
        similarity_threshold: Minimum similarity threshold (default: 0.85)
        image_subdir: Subdirectory containing images (default: "images")
        device: Device to run CLIP on (default: "cuda")
    """
    output_dir = f"/mnt/nfs/SpatialAI/sq/optical_flow_datavggt_ft_optical_flow/{dataset_type}"
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Auto-detect dataset type if not provided
    if dataset_type is None:
        if original_dataset_root is None:
            raise ValueError("Either dataset_type or original_dataset_root must be provided")
        dataset_type = detect_dataset_type_from_path(output_dir, original_dataset_root)
        if dataset_type is None:
            raise ValueError("Could not auto-detect dataset type. Please specify --dataset_type")
        print(f"Auto-detected dataset type: {dataset_type}")
    
    original_dataset_root = "/mnt/nfs/SpatialAI/wai_datasets"
    original_root = Path(original_dataset_root)
    if not original_root.exists():
        raise ValueError(f"Original dataset root does not exist: {original_dataset_root}")
    
    # Load CLIP model
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Find all .pt files
    pt_files = list(output_path.rglob("*.pt"))
    print(f"Found {len(pt_files)} .pt files to process")
    
    # Output file for low-confidence pairs
    log_file = output_path / "clip_low_conf.txt"
    
    low_conf_count = 0
    processed_count = 0
    failed_count = 0
    
    with open(log_file, 'w') as f:
        f.write(f"# CLIP similarity check results (threshold: {similarity_threshold})\n")
        f.write(f"# Format: similarity_score img1_path img2_path\n\n")
        
        for pt_file in tqdm(pt_files, desc="Processing .pt files"):
            # Reconstruct image pair paths
            pair_paths = get_image_pair_paths(
                pt_file, dataset_type, original_dataset_root, image_subdir
            )
            
            if pair_paths is None:
                failed_count += 1
                continue
            
            img1_path, img2_path = pair_paths
            
            # Compute CLIP similarity
            similarity = compute_clip_similarity(
                img1_path, img2_path, model, preprocess, device
            )
            
            if similarity is None:
                failed_count += 1
                continue
            
            processed_count += 1
            
            # Log if similarity is below threshold
            if similarity < similarity_threshold:
                low_conf_count += 1
                f.write(f"{similarity:.4f} {img1_path} {img2_path}\n")
                f.flush()  # Ensure immediate write
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count}/{len(pt_files)}")
    print(f"  Low confidence (similarity < {similarity_threshold}): {low_conf_count}")
    print(f"  Failed to process: {failed_count}")
    print(f"  Results saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Check CLIP similarity for processed optical flow pairs"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["dynamicreplica", "mvs_synth", "unrealstereo4k", "spring", "sailvos3d"],
        help="Type of dataset (will auto-detect if not provided)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Minimum similarity threshold (default: 0.85)"
    )
    parser.add_argument(
        "--image_subdir",
        type=str,
        default="images",
        help="Subdirectory containing images (default: images)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run CLIP on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    check_clip_similarity_for_dataset(
        dataset_type=args.dataset_type,
        similarity_threshold=args.similarity_threshold,
        image_subdir=args.image_subdir,
        device=args.device
    )


if __name__ == "__main__":
    main()

