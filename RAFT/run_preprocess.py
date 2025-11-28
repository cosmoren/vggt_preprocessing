"""
Universal preprocessing script for optical flow estimation.

This script can be used to process different datasets by importing
the appropriate dataset processor.
"""

from preprocessing import FlowProcessor
from data_preprocessors import DynamicReplicaProcessor, MVSSynthProcessor, Sailvos3dProcessor, SpringProcessor, UnrealStereo4KProcessor, Eth3DProcessor, ScannetppV2Processor
from dataset_loaders import DynamicReplicaLoader
from pathlib import Path
import torch
from torchvision.utils import flow_to_image, save_image
import multiprocessing as mp
from typing import Optional
import argparse


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(description='Universal optical flow preprocessing')
    parser.add_argument('--dataset', type=str, default='dynamicreplica',
                       choices=['dynamicreplica', 'mvs_synth', 'sailvos3d', 'spring', 'unrealstereo4k', 'eth3d', 'scannetppv2'],
                       help='Dataset type to process')
    parser.add_argument('--mode', type=str, default='single_gpu',
                       choices=['single_gpu', 'multi_gpu', 'test'],
                       help='Processing mode')
    parser.add_argument('--batch_size', type=int, default=10,  
                       help='Batch size for processing') #12 for dynamicreplica, 2 for mvs_synth, 10 for sailvos3d, 3 for spring, 1 for eth3d
    parser.add_argument('--resize_k', type=int, default=8,
                       help='Resize to nearest multiple of k')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2],
                       help='GPU IDs to use (default: all available)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for single GPU mode')
    parser.add_argument('--inlier_thresh', type=float, default=10,
                       help='Inlier threshold for RANSAC') # 8 for dynamicreplica, 10 for mvs_synth, 10 for sailvos3d, 10 for spring, 10 for unrealstereo4k, 10 for eth3d, 10 for scannetppv2
    
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
        # test_first_pair(
        #     dataset_type=args.dataset,
        #     root_dir=args.root_dir,
        #     device=args.device,
        #     resize_k=args.resize_k,
        #     output_dir=args.output_dir,
        #     camera_type=args.camera_type
        # )
        print("no test function")
    
    elif args.mode == 'multi_gpu':
        # Multi-GPU processing (always processes both left and right cameras)
        if args.dataset == 'dynamicreplica':
            DynamicReplicaProcessor.process_multi_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
                batch_size=args.batch_size,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                gpu_ids=args.gpu_ids,
                ransac_config=ransac_config,
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}"
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
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'mvs_synth':
            MVSSynthProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/mvs_synth",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'sailvos3d':
            Sailvos3dProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/sailvos3d",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'spring':
            SpringProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/spring",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'unrealstereo4k':
            UnrealStereo4KProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/unrealstereo4k",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'eth3d':
            Eth3DProcessor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/eth3d",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        elif args.dataset == 'scannetppv2':
            ScannetppV2Processor.process_single_gpu(
                root_dir="/mnt/nfs/SpatialAI/wai_datasets/scannetppv2",
                batch_size=args.batch_size,
                device=args.device,
                resize_k=args.resize_k,
                output_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_optical_flow/{args.dataset}",
                vis_dir=f"/mnt/glusterfs/SpatialAI/Datasets/sq/optical_flow_data/vggt_ft_flow_vis/{args.dataset}",
                ransac_config=ransac_config
            )
        else:
            raise ValueError(f"Unsupported dataset type for single_gpu mode: {args.dataset}")


if __name__ == '__main__':
   
    main()
