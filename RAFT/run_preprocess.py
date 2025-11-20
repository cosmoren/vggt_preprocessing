"""
Universal preprocessing script for optical flow estimation.

This script can be used to process different datasets by importing
the appropriate dataset processor.
"""

from preprocessing import FlowProcessor
from data_preprocessors import DynamicReplicaProcessor, MVSSynthProcessor, Sailvos3dProcessor
from dataset_loaders import DynamicReplicaLoader
from pathlib import Path
import torch
from torchvision.utils import flow_to_image, save_image
import multiprocessing as mp
from typing import Optional
import argparse


def test_first_pair(
    dataset_type: str = "dynamicreplica",
    root_dir: str = "/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
    device: str = 'cuda',
    resize_k: int = 8,
    output_dir: str = "./test_output",
    **kwargs
):
    """
    Load and process the first pair from the dataloader for testing.
    
    Args:
        dataset_type: Type of dataset ("dynamicreplica", "sequential", "imagepair")
        root_dir: Root directory of dataset
        device: Device to run on
        resize_k: Resize to nearest multiple of k
        output_dir: Output directory for test results
        **kwargs: Additional dataset-specific arguments
    """
    print("\n" + "="*60)
    print(f"Testing first pair from {dataset_type} dataset...")
    print("="*60)
    
    # Initialize processor
    processor = FlowProcessor(
        device=device,
        resize_k=resize_k,
        ransac_config={
            'num_iters': 1000,
            'sample_size': 200,
            'inlier_thresh': 10.0,
            'min_inlier_ratio': 0.2,
            'verbose': False
        }
    )
    
    # Create dataset loader based on type
    if dataset_type == "dynamicreplica":
        camera_type = kwargs.get('camera_type', 'left')
        loader = DynamicReplicaLoader(
            root_dir=root_dir,
            image_subdir="images",
            camera_type=camera_type,
            sort=True
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if len(loader) == 0:
        print(f"Error: No image pairs found")
        return None
    
    print(f"Found {len(loader)} image pairs")
    print(f"Loading first pair (index 0)...")
    
    # Load first pair
    img1_batch, img2_batch, metadata = loader.get_batch([0])
    
    print(f"Image 1 shape: {img1_batch.shape}")
    print(f"Image 2 shape: {img2_batch.shape}")
    if 'img1_paths' in metadata:
        print(f"Image 1 path: {metadata['img1_paths'][0]}")
        print(f"Image 2 path: {metadata['img2_paths'][0]}")
    
    # Process the pair
    print("\nProcessing flow...")
    results = processor.process_batch(
        img1_batch,
        img2_batch,
        compute_rigid=True,
        compute_dynamic=True
    )
    
    print(f"Flow shape: {results['flow'].shape}")
    if 'rigid_flow' in results:
        print(f"Rigid flow shape: {results['rigid_flow'].shape}")
        print(f"Residual flow shape: {results['flow_res'].shape}")
    if 'dynamic_flow' in results:
        print(f"Dynamic flow shape: {results['dynamic_flow'].shape}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}...")
    
    # Save flow tensors
    torch.save(results['flow'], output_path / 'flow.pt')
    print(f"Saved: {output_path / 'flow.pt'}")
    
    if 'rigid_flow' in results:
        torch.save(results['rigid_flow'], output_path / 'rigid_flow.pt')
        torch.save(results['flow_res'], output_path / 'flow_res.pt')
        torch.save(results['inlier_masks'], output_path / 'inlier_masks.pt')
        print(f"Saved: {output_path / 'rigid_flow.pt'}")
        print(f"Saved: {output_path / 'flow_res.pt'}")
        print(f"Saved: {output_path / 'inlier_masks.pt'}")
        
        if results['affine_matrices'][0] is not None:
            torch.save(results['affine_matrices'][0], output_path / 'affine_matrix.pt')
            print(f"Saved: {output_path / 'affine_matrix.pt'}")
    
    # Save visualization images
    if 'flow' in results:
        flow_img = flow_to_image(results['flow'])
        save_image(flow_img.float() / 255.0, output_path / 'flow_visualization.jpg')
        print(f"Saved: {output_path / 'flow_visualization.jpg'}")
    
    if 'rigid_flow' in results:
        rigid_flow_img = flow_to_image(results['rigid_flow'])
        save_image(rigid_flow_img.float() / 255.0, output_path / 'rigid_flow_visualization.jpg')
        print(f"Saved: {output_path / 'rigid_flow_visualization.jpg'}")
        
        flow_res_img = flow_to_image(results['flow_res'])
        save_image(flow_res_img.float() / 255.0, output_path / 'flow_res_visualization.jpg')
        print(f"Saved: {output_path / 'flow_res_visualization.jpg'}")
    
    if 'dynamic_flow' in results:
        dynamic_flow_img = flow_to_image(results['dynamic_flow'])
        save_image(dynamic_flow_img.float() / 255.0, output_path / 'dynamic_flow_visualization.jpg')
        print(f"Saved: {output_path / 'dynamic_flow_visualization.jpg'}")
    
    # Save metadata
    import json
    metadata_to_save = {}
    if 'img1_paths' in metadata:
        metadata_to_save['img1_path'] = metadata['img1_paths'][0]
        metadata_to_save['img2_path'] = metadata['img2_paths'][0]
    if 'scene_names' in metadata:
        metadata_to_save['scene_name'] = metadata['scene_names'][0]
    if 'camera_types' in metadata:
        metadata_to_save['camera_type'] = metadata['camera_types'][0]
    if 'pair_indices' in metadata:
        metadata_to_save['pair_index'] = metadata['pair_indices'][0]
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    print(f"Saved: {output_path / 'metadata.json'}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    return results


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(description='Universal optical flow preprocessing')
    parser.add_argument('--dataset', type=str, default='dynamicreplica',
                       choices=['dynamicreplica', 'mvs_synth', 'sailvos3d'],
                       help='Dataset type to process')
    parser.add_argument('--mode', type=str, default='single_gpu',
                       choices=['single_gpu', 'multi_gpu', 'test'],
                       help='Processing mode')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for processing')
    parser.add_argument('--resize_k', type=int, default=8,
                       help='Resize to nearest multiple of k')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2],
                       help='GPU IDs to use (default: all available)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for single GPU mode')
    parser.add_argument('--inlier_thresh', type=float, default=3.0,
                       help='Inlier threshold for RANSAC')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method (required for CUDA)
    mp.set_start_method('spawn', force=True)
    
    # Default RANSAC config
    ransac_config = {
        'num_iters': 1000,
        'sample_size': 200,
        'inlier_thresh': args.inlier_thresh, #3 for dynamicreplica, 20 for mvs_synth, 10 for sailvos3d
        'min_inlier_ratio': 0.1,
        'verbose': False
    }
    
    if args.mode == 'test':
        # Test mode: process first pair
        test_first_pair(
            dataset_type=args.dataset,
            root_dir=args.root_dir,
            device=args.device,
            resize_k=args.resize_k,
            output_dir=args.output_dir,
            camera_type=args.camera_type
        )
    
    elif args.mode == 'multi_gpu':
        # Multi-GPU processing (always processes both left and right cameras)
        if args.dataset == 'dynamicreplica':
            DynamicReplicaProcessor.process_multi_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
                batch_size=args.batch_size,
                resize_k=args.resize_k,
                output_dir=f"/work/Datasets/vggt_ft_optical_flow/{args.dataset}",
                gpu_ids=args.gpu_ids,
                ransac_config=ransac_config,
                vis_dir=f"/work/Datasets/vggt_ft_flow_vis/{args.dataset}"
            )
        else:
            raise ValueError(f"Unsupported dataset type for multi_gpu mode: {args.dataset}")
    
    else:  # single_gpu
        # Single GPU processing (always processes both left and right cameras)
        if args.dataset == 'dynamicreplica':
            DynamicReplicaProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/work/Datasets/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/work/Datasets/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'mvs_synth':
            MVSSynthProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/mvs_synth",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/work/Datasets/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/work/Datasets/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'sailvos3d':
            Sailvos3dProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/sailvos3d",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/work/Datasets/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/work/Datasets/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        else:
            raise ValueError(f"Unsupported dataset type for single_gpu mode: {args.dataset}")


if __name__ == '__main__':
   
    main()
