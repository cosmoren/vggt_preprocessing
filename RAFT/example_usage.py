"""
Example usage of the preprocessing module.

This demonstrates how to use the FlowProcessor with different dataset loaders.
"""

from preprocessing import FlowProcessor
from dataset_loaders import DynamicReplicaLoader
from pathlib import Path
import torch
from torchvision.utils import flow_to_image, save_image


def process_dynamicreplica(
    root_dir: str = "/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
    batch_size: int = 50,
    device: str = 'cuda',
    resize_k: int = 8,
    output_dir: str = "./output_dynamicreplica",
    process_left: bool = True,
    process_right: bool = True
):
    """
    Process DynamicReplica dataset.
    
    Args:
        root_dir: Root directory of DynamicReplica dataset
        batch_size: Batch size for processing (default: 50)
        device: Device to run on
        resize_k: Resize to nearest multiple of k
        output_dir: Output directory for results
        process_left: Whether to process left camera images
        process_right: Whether to process right camera images
    """
    # Initialize processor
    processor = FlowProcessor(
        device=device,
        resize_k=resize_k,
        ransac_config={
            'num_iters': 1000,
            'sample_size': 200,
            'inlier_thresh': 120.0,
            'min_inlier_ratio': 0.2,
            'verbose': True
        }
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Process left camera images
    if process_left:
        print("\n" + "="*60)
        print("Processing LEFT camera images...")
        print("="*60)
        
        loader_left = DynamicReplicaLoader(
            root_dir=root_dir,
            image_subdir="images",
            camera_type="left",
            sort=True
        )
        
        print(f"Found {len(loader_left)} left image pairs")
        print(f"Will process in {len(loader_left) // batch_size + (1 if len(loader_left) % batch_size > 0 else 0)} batches")
        
        left_results = processor.process_dataset(
            dataset_loader=loader_left,
            batch_size=batch_size,
            save_dir=output_path  # Save to same parent directory
        )
        
        all_results.extend(left_results)
        print(f"Completed processing {len(left_results)} left batches")
    
    # Process right camera images
    if process_right:
        print("\n" + "="*60)
        print("Processing RIGHT camera images...")
        print("="*60)
        
        loader_right = DynamicReplicaLoader(
            root_dir=root_dir,
            image_subdir="images",
            camera_type="right",
            sort=True
        )
        
        print(f"Found {len(loader_right)} right image pairs")
        print(f"Will process in {len(loader_right) // batch_size + (1 if len(loader_right) % batch_size > 0 else 0)} batches")
        
        right_results = processor.process_dataset(
            dataset_loader=loader_right,
            batch_size=batch_size,
            save_dir=output_path  # Save to same parent directory
        )
        
        all_results.extend(right_results)
        print(f"Completed processing {len(right_results)} right batches")
    
    print("\n" + "="*60)
    print(f"Processing complete! Total batches: {len(all_results)}")
    print(f"Results saved to: {output_path}")
    print("="*60)
    
    return all_results


def test_first_pair(
    root_dir: str = "/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
    device: str = 'cuda',
    resize_k: int = 8,
    output_dir: str = "./test_output",
    camera_type: str = "left"
):
    """
    Load and process the first pair from the dataloader for testing.
    
    Args:
        root_dir: Root directory of DynamicReplica dataset
        device: Device to run on
        resize_k: Resize to nearest multiple of k
        output_dir: Output directory for test results
        camera_type: Which camera to test ("left" or "right")
    """
    print("\n" + "="*60)
    print(f"Testing first pair from {camera_type} camera...")
    print("="*60)
    
    # Initialize processor
    processor = FlowProcessor(
        device=device,
        resize_k=resize_k,
        ransac_config={
            'num_iters': 1000,
            'sample_size': 200,
            'inlier_thresh': 1.0,
            'min_inlier_ratio': 0.2,
            'verbose': False
        }
    )
    
    # Create dataset loader
    loader = DynamicReplicaLoader(
        root_dir=root_dir,
        image_subdir="images",
        camera_type=camera_type,
        sort=True
    )
    
    if len(loader) == 0:
        print(f"Error: No image pairs found in {camera_type} camera")
        return None
    
    print(f"Found {len(loader)} image pairs")
    print(f"Loading first pair (index 0)...")
    
    # Load first pair
    img1_batch, img2_batch, metadata = loader.get_batch([0])
    
    print(f"Image 1 shape: {img1_batch.shape}")
    print(f"Image 2 shape: {img2_batch.shape}")
    print(f"Image 1 path: {metadata['img1_paths'][0]}")
    print(f"Image 2 path: {metadata['img2_paths'][0]}")
    print(f"Scene: {metadata['scene_names'][0]}")
    print(f"Camera: {metadata['camera_types'][0]}")
    
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
    metadata_to_save = {
        'img1_path': metadata['img1_paths'][0],
        'img2_path': metadata['img2_paths'][0],
        'scene_name': metadata['scene_names'][0],
        'camera_type': metadata['camera_types'][0],
        'pair_index': metadata['pair_indices'][0]
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    print(f"Saved: {output_path / 'metadata.json'}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    return results


if __name__ == '__main__':
    # Test first pair
    # test_first_pair(
    #     root_dir="/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
    #     device='cuda',
    #     resize_k=8,
    #     output_dir="./test_output",
    #     camera_type="left"
    # )
    
    # Process DynamicReplica dataset (commented out - uncomment to run full processing)
    process_dynamicreplica(
        root_dir="/mnt/nfs/SpatialAI/wai_datasets/dynamicreplica",
        batch_size=5, #50,
        device='cuda',
        resize_k=8,
        output_dir="/work/Datasets/ddad_train_val/output_dynamicreplica",
        process_left=True,
        process_right=True
    )

