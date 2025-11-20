"""
Dataset-specific preprocessing processors.

Each dataset type has its own processor class that handles dataset-specific
logic like scene organization, multi-camera handling, etc.
"""

from preprocessing import FlowProcessor
from dataset_loaders import DynamicReplicaLoader, MVSSynthLoader, Sailvos3dLoader
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import torch
import multiprocessing as mp
import os


class DynamicReplicaProcessor:
    """Processor for DynamicReplica dataset with scene-based organization."""
    
    @staticmethod
    def get_all_scenes(root_dir: str) -> List[str]:
        """
        Get list of all scene names in the dataset.
        
        Args:
            root_dir: Root directory containing scene folders
        
        Returns:
            List of scene names
        """
        root_path = Path(root_dir)
        scene_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        return [d.name for d in scene_dirs]
    
    @staticmethod
    def process_single_scene(
        scene_name: str,
        root_dir: str,
        output_dir: str,
        gpu_id: int,
        batch_size: int = 50,
        resize_k: int = 8,
        ransac_config: Optional[dict] = None,
        vis_dir: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Process a single scene on a specific GPU.
        
        This function is designed to be called by multiprocessing workers.
        Each worker processes one scene at a time. Always processes both left and right cameras.
        
        Args:
            scene_name: Name of the scene to process
            root_dir: Root directory of DynamicReplica dataset
            output_dir: Output directory for .pt files
            gpu_id: GPU device ID to use
            batch_size: Batch size for processing
            resize_k: Resize to nearest multiple of k
            ransac_config: RANSAC configuration dictionary
            vis_dir: Directory to save visualization images (optional)
        
        Returns:
            Tuple of (success: bool, scene_name: str)
        """
        # Set the GPU device for this process
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        
        # Default RANSAC config
        if ransac_config is None:
            ransac_config = {
                'num_iters': 1000,
                'sample_size': 200,
                'inlier_thresh': 120.0,
                'min_inlier_ratio': 0.2,
                'verbose': False
            }
        
        try:
            print(f"[GPU {gpu_id}] Processing scene: {scene_name}")
            
            # Initialize processor on this GPU
            processor = FlowProcessor(
                device=device,
                resize_k=resize_k,
                ransac_config=ransac_config
            )
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            vis_path = Path(vis_dir) if vis_dir else None
            if vis_path:
                vis_path.mkdir(parents=True, exist_ok=True)

            # Process left camera images
            loader_left = DynamicReplicaLoader(
                root_dir=root_dir,
                image_subdir="images",
                camera_type="left",
                scene_name=scene_name,
                sort=True
            )
            
            if len(loader_left) > 0:
                print(f"[GPU {gpu_id}] Scene {scene_name}: Processing {len(loader_left)} left pairs")
                processor.process_dataset(
                    dataset_loader=loader_left,
                    batch_size=batch_size,
                    save_dir=output_path,
                    vis_dir=vis_path
                )
            del loader_left
            torch.cuda.empty_cache()
            
            # Process right camera images
            loader_right = DynamicReplicaLoader(
                root_dir=root_dir,
                image_subdir="images",
                camera_type="right",
                scene_name=scene_name,
                sort=True
            )
            
            if len(loader_right) > 0:
                print(f"[GPU {gpu_id}] Scene {scene_name}: Processing {len(loader_right)} right pairs")
                processor.process_dataset(
                    dataset_loader=loader_right,
                    batch_size=batch_size,
                    save_dir=output_path,
                    vis_dir=vis_path
                )
            del loader_right
            torch.cuda.empty_cache()
            
            print(f"[GPU {gpu_id}] Completed scene: {scene_name}")
            del processor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True, scene_name
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            return False, scene_name
    
    @staticmethod
    def process_multi_gpu(
        root_dir: str,
        batch_size: int = 50,
        resize_k: int = 8,
        output_dir: str = "./output_dynamicreplica",
        gpu_ids: Optional[List[int]] = None,
        ransac_config: Optional[dict] = None,
        vis_dir: Optional[str] = None
    ) -> List[Tuple[bool, str]]:
        """
        Process DynamicReplica dataset across multiple GPUs.
        
        Each GPU processes one scene at a time. Scenes are distributed across
        available GPUs for parallel processing. Always processes both left and right cameras.
        
        Args:
            root_dir: Root directory of DynamicReplica dataset
            batch_size: Batch size for processing (default: 50)
            resize_k: Resize to nearest multiple of k
            output_dir: Output directory for .pt files
            gpu_ids: List of GPU IDs to use (default: all available GPUs)
            ransac_config: RANSAC configuration dictionary
            vis_dir: Directory to save visualization images
        
        Returns:
            List of (success, scene_name) tuples
        """
        # Get all scenes
        scenes = DynamicReplicaProcessor.get_all_scenes(root_dir)
        print(f"Found {len(scenes)} scenes to process")
        
        # Initialize output directory and RANSAC failure log
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ransac_failure_log = output_path / 'ransac_failures.txt'
        if ransac_failure_log.exists():
            ransac_failure_log.unlink()  # Clear previous log
        
        # Determine available GPUs
        if gpu_ids is None:
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                raise RuntimeError("No CUDA devices available")
        
        print(f"Using GPUs: {gpu_ids}")
        print(f"Total scenes: {len(scenes)}")
        
        # Distribute scenes across GPUs (round-robin)
        scene_gpu_pairs = []
        for i, scene in enumerate(scenes):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            scene_gpu_pairs.append((
                scene,  # scene_name (position 0)
                root_dir,  # root_dir (position 1)
                output_dir,  # output_dir (position 2)
                gpu_id,  # gpu_id (position 3)
                batch_size,  # batch_size (position 4)
                resize_k,  # resize_k (position 5)
                ransac_config,  # ransac_config (position 6)
                vis_dir  # vis_dir (position 7)
            ))
        
        # Process scenes in parallel
        print("\n" + "="*60)
        print("Starting multi-GPU processing...")
        print("="*60)
        
        with mp.Pool(processes=len(gpu_ids)) as pool:
            results = pool.starmap(DynamicReplicaProcessor.process_single_scene, scene_gpu_pairs)
        
        # Report results
        successful = [r[1] for r in results if r[0]]
        failed = [r[1] for r in results if not r[0]]
        
        print("\n" + "="*60)
        print(f"Processing complete!")
        print(f"Successfully processed: {len(successful)} scenes")
        if failed:
            print(f"Failed scenes: {len(failed)}")
            for scene in failed:
                print(f"  - {scene}")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
        return results
    
    @staticmethod
    def process_single_gpu(
        root_dir: str,
        batch_size: int = 50,
        device: str = 'cuda',
        resize_k: int = 8,
        output_dir: str = "./output_dynamicreplica",
        vis_dir: Optional[str] = None,
        ransac_config: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Process DynamicReplica dataset on a single GPU.
        
        Processes one scene at a time, using one dataloader per scene.
        Always processes both left and right cameras for each scene.
        
        Args:
            root_dir: Root directory of DynamicReplica dataset
            batch_size: Batch size for processing (default: 50)
            device: Device to run on
            resize_k: Resize to nearest multiple of k
            output_dir: Output directory for .pt files
            vis_dir: Directory to save visualization images (optional)
            ransac_config: RANSAC configuration dictionary
        
        Returns:
            List of result dictionaries
        """
        # Default RANSAC config
        if ransac_config is None:
            ransac_config = {
                'num_iters': 1000,
                'sample_size': 200,
                'inlier_thresh': 120.0,
                'min_inlier_ratio': 0.2,
                'verbose': True
            }
        
        # Get all scenes
        scenes = DynamicReplicaProcessor.get_all_scenes(root_dir)
        print(f"Found {len(scenes)} scenes to process on {device}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize RANSAC failure log file (clear previous log)
        ransac_failure_log = output_path / 'ransac_failures.txt'
        if ransac_failure_log.exists():
            ransac_failure_log.unlink()  # Clear previous log
        
        vis_path = Path(vis_dir) if vis_dir else None
        if vis_path:
            vis_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor (reused across scenes)
        processor = FlowProcessor(
            device=device,
            resize_k=resize_k,
            ransac_config=ransac_config
        )
        
        
        # Process each scene separately
        for scene_idx, scene_name in enumerate(scenes):
            if scene_name == "dynamicreplica":
                print("DDDDDDDDDDD, encounter dynamicreplica!")
                continue
            print(f"\n[{device}] Processing scene {scene_idx + 1}/{len(scenes)}: {scene_name}")
            
            try:
                # Process left camera images
                loader_left = DynamicReplicaLoader(
                    root_dir=root_dir,
                    image_subdir="images",
                    camera_type="left",
                    scene_name=scene_name,
                    sort=True
                )
                
                if len(loader_left) > 0:
                    print(f"[{device}] Scene {scene_name}: Processing {len(loader_left)} left pairs")
                    processor.process_dataset(
                        dataset_loader=loader_left,
                        batch_size=batch_size,
                        save_dir=output_path,
                        vis_dir=vis_path
                    )
                    
                del loader_left
                torch.cuda.empty_cache()
                
                # Process right camera images
                loader_right = DynamicReplicaLoader(
                    root_dir=root_dir,
                    image_subdir="images",
                    camera_type="right",
                    scene_name=scene_name,
                    sort=True
                )
                
                if len(loader_right) > 0:
                    print(f"[{device}] Scene {scene_name}: Processing {len(loader_right)} right pairs")
                    processor.process_dataset(
                        dataset_loader=loader_right,
                        batch_size=batch_size,
                        save_dir=output_path,
                        vis_dir=vis_path
                    )
                    
                del loader_right
                torch.cuda.empty_cache()
                
                print(f"[{device}] Completed scene: {scene_name}")
                
            except Exception as e:
                print(f"[{device}] Error processing scene {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        del processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n[{device}] Single GPU processing complete. Processed {len(scenes)} scenes.")

class MVSSynthProcessor:
    """Processor for MVS Synth dataset with scene-based organization."""

    @staticmethod
    def get_all_scenes(root_dir: str) -> List[str]:
        """
        Get list of all scene names in the dataset.
        
        Args:
            root_dir: Root directory containing scene folders
        
        Returns:
            List of scene names
        """
        root_path = Path(root_dir)
        scene_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        return [d.name for d in scene_dirs]
    
    @staticmethod
    def process_single_gpu(
        root_dir: str,
        batch_size: int = 50,
        device: str = 'cuda',
        resize_k: int = 8,
        output_dir: str = "./output_mvs_synth",
        vis_dir: Optional[str] = None,
        ransac_config: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Process mvssynth dataset on a single GPU.
        
        Processes one scene at a time, using one dataloader per scene.
        Always processes both left and right cameras for each scene.
        
        Args:
            root_dir: Root directory of mvssynth dataset
            batch_size: Batch size for processing (default: 50)
            device: Device to run on
            resize_k: Resize to nearest multiple of k
            output_dir: Output directory for .pt files
            vis_dir: Directory to save visualization images (optional)
            ransac_config: RANSAC configuration dictionary
        
        Returns:
            List of result dictionaries
        """
        # Default RANSAC config
        if ransac_config is None:
            ransac_config = {
                'num_iters': 1000,
                'sample_size': 200,
                'inlier_thresh': 120.0,
                'min_inlier_ratio': 0.2,
                'verbose': True
            }
        
        # Get all scenes
        scenes = MVSSynthProcessor.get_all_scenes(root_dir)
        print(f"Found {len(scenes)} scenes to process on {device}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize RANSAC failure log file (clear previous log)
        ransac_failure_log = output_path / 'ransac_failures.txt'
        if ransac_failure_log.exists():
            ransac_failure_log.unlink()  # Clear previous log
        
        vis_path = Path(vis_dir) if vis_dir else None
        if vis_path:
            vis_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor (reused across scenes)
        processor = FlowProcessor(
            device=device,
            resize_k=resize_k,
            ransac_config=ransac_config
        )
        
        
        # Process each scene separately
        for scene_idx, scene_name in enumerate(scenes):
            print(f"\n[{device}] Processing scene {scene_idx + 1}/{len(scenes)}: {scene_name}")
            
            try:
                # Process images
                loader = MVSSynthLoader(
                    root_dir=root_dir,
                    image_subdir="images",
                    sort=True,
                    scene_name=scene_name,
                )
                
                if len(loader) > 0:
                    print(f"[{device}] Scene {scene_name}: Processing {len(loader)} left pairs")
                    processor.process_dataset(
                        dataset_loader=loader,
                        batch_size=batch_size,
                        save_dir=output_path,
                        vis_dir=vis_path
                    )
                    
                del loader
                torch.cuda.empty_cache()
                
                
                print(f"[{device}] Completed scene: {scene_name}")
                
            except Exception as e:
                print(f"[{device}] Error processing scene {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        del processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n[{device}] Single GPU processing complete. Processed {len(scenes)} scenes.")
    
class Sailvos3dProcessor:
    """Processor for Sailvos3d dataset with scene-based organization."""

    @staticmethod
    def get_all_scenes(root_dir: str) -> List[str]:
        """
        Get list of all scene names in the dataset.
        
        Args:
            root_dir: Root directory containing scene folders
        
        Returns:
            List of scene names
        """
        root_path = Path(root_dir)
        scene_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and os.path.exists(d / "images")])
        return [d.name for d in scene_dirs]
    
    @staticmethod
    def process_single_gpu(
        root_dir: str,
        batch_size: int = 50,
        device: str = 'cuda',
        resize_k: int = 8,
        output_dir: str = "./output_sailvos3d",
        vis_dir: Optional[str] = None,
        ransac_config: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Process sailvos3d dataset on a single GPU.
        
        Processes one scene at a time, using one dataloader per scene.
        Always processes both left and right cameras for each scene.
        
        Args:
            root_dir: Root directory of sailvos3d dataset
            batch_size: Batch size for processing (default: 50)
            device: Device to run on
            resize_k: Resize to nearest multiple of k
            output_dir: Output directory for .pt files
            vis_dir: Directory to save visualization images (optional)
            ransac_config: RANSAC configuration dictionary
        
        Returns:
            List of result dictionaries
        """
        # Default RANSAC config
        if ransac_config is None:
            ransac_config = {
                'num_iters': 1000,
                'sample_size': 200,
                'inlier_thresh': 120.0,
                'min_inlier_ratio': 0.2,
                'verbose': True
            }
        
        # Get all scenes
        scenes = Sailvos3dProcessor.get_all_scenes(root_dir)
        print(f"Found {len(scenes)} scenes to process on {device}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize RANSAC failure log file (clear previous log)
        ransac_failure_log = output_path / 'ransac_failures.txt'
        if ransac_failure_log.exists():
            ransac_failure_log.unlink()  # Clear previous log
        
        vis_path = Path(vis_dir) if vis_dir else None
        if vis_path:
            vis_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor (reused across scenes)
        processor = FlowProcessor(
            device=device,
            resize_k=resize_k,
            ransac_config=ransac_config
        )
        
        
        # Process each scene separately
        for scene_idx, scene_name in enumerate(scenes):
            print(f"\n[{device}] Processing scene {scene_idx + 1}/{len(scenes)}: {scene_name}")
            
            try:
                # Process images
                loader = Sailvos3dLoader(
                    root_dir=root_dir,
                    image_subdir="images",
                    sort=True,
                    scene_name=scene_name,
                )
                
                if len(loader) > 0:
                    print(f"[{device}] Scene {scene_name}: Processing {len(loader)} left pairs")
                    processor.process_dataset(
                        dataset_loader=loader,
                        batch_size=batch_size,
                        save_dir=output_path,
                        vis_dir=vis_path
                    )
                    
                del loader
                torch.cuda.empty_cache()
                
                
                print(f"[{device}] Completed scene: {scene_name}")
                
            except Exception as e:
                print(f"[{device}] Error processing scene {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        del processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n[{device}] Single GPU processing complete. Processed {len(scenes)} scenes.")