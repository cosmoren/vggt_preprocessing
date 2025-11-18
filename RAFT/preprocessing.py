"""
Optical Flow Preprocessing Module

This module provides classes for preprocessing images, estimating optical flow
using RAFT, and decomposing flow into rigid and residual components using RANSAC.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image, save_image
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path


class ImagePreprocessor:
    """Handles image preprocessing and resizing operations."""
    
    def __init__(self, normalize_mean: float = 0.5, normalize_std: float = 0.5):
        """
        Args:
            normalize_mean: Mean for normalization (default: 0.5 to map [0,1] to [-1,1])
            normalize_std: Std for normalization (default: 0.5)
        """
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.transforms = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    
    def preprocess(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image batch.
        
        Args:
            batch: Image tensor of shape (B, C, H, W) with values in [0, 1]
        
        Returns:
            Preprocessed tensor with values in [-1, 1]
        """
        return self.transforms(batch)
    
    def resize_to_nearest_multiple(
        self, 
        x: torch.Tensor, 
        k: int = 32, 
        mode: str = 'bilinear'
    ) -> torch.Tensor:
        """
        Resize tensor to nearest multiple of k.
        
        Args:
            x: Tensor of shape [B, C, H, W]
            k: Desired multiple (e.g., 8, 16, 32)
            mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')
        
        Returns:
            Resized tensor
        """
        B, C, H, W = x.shape
        new_H = round(H / k) * k
        new_W = round(W / k) * k
        
        new_H = max(new_H, k)
        new_W = max(new_W, k)
        
        if new_H == H and new_W == W:
            return x
        
        align_corners = False if mode in ['bilinear', 'bicubic'] else None
        x_resized = F.interpolate(
            x, size=(new_H, new_W),
            mode=mode,
            align_corners=align_corners
        )
        return x_resized


class RAFTFlowEstimator:
    """Handles RAFT model loading and optical flow estimation."""
    
    def __init__(
        self, 
        device: str = 'cuda',
        model_type: str = 'large',
        pretrained: bool = True
    ):
        """
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            model_type: Model type ('large' or 'small')
            pretrained: Whether to use pretrained weights
        """
        self.device = device
        self.model = None
        self._load_model(model_type, pretrained)
    
    def _load_model(self, model_type: str, pretrained: bool):
        """Load RAFT model."""
        if model_type == 'large':
            self.model = raft_large(pretrained=pretrained, progress=False)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        self.model = self.model.to(self.device).eval()
    
    def estimate_flow(
        self, 
        img1_batch: torch.Tensor, 
        img2_batch: torch.Tensor,
        return_all_iterations: bool = False
    ) -> torch.Tensor:
        """
        Estimate optical flow between two image batches.
        
        Args:
            img1_batch: First image batch (B, C, H, W)
            img2_batch: Second image batch (B, C, H, W)
            return_all_iterations: If True, return all iterations; else return final flow
        
        Returns:
            Optical flow tensor (B, 2, H, W) or list of flows if return_all_iterations=True
        """
        with torch.no_grad():
            list_of_flows = self.model(img1_batch, img2_batch)
        
        if return_all_iterations:
            return list_of_flows
        else:
            return list_of_flows[-1]  # Return final flow


class RANSACAffineEstimator:
    """Estimates affine transformation from optical flow using RANSAC."""
    
    def __init__(
        self,
        num_iters: int = 1000,
        sample_size: int = 3,
        inlier_thresh: float = 1.0,
        min_inlier_ratio: float = 0.3,
        verbose: bool = False
    ):
        """
        Args:
            num_iters: Number of RANSAC iterations
            sample_size: Number of points to sample for each iteration
            inlier_thresh: Threshold for inlier classification (in pixels)
            min_inlier_ratio: Minimum ratio of inliers to accept result
            verbose: Whether to print debug information
        """
        self.num_iters = num_iters
        self.sample_size = sample_size
        self.inlier_thresh = inlier_thresh
        self.min_inlier_ratio = min_inlier_ratio
        self.verbose = verbose
    
    def estimate_single(
        self,
        flow_1img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Estimate affine transformation for a single flow image.
        
        Args:
            flow_1img: Flow tensor (2, H, W), flow[y,x] = (du, dv)
        
        Returns:
            Tuple of (rigid_flow, flow_res, M_best, inlier_mask):
                - rigid_flow: (2, H, W) rigid component
                - flow_res: (2, H, W) residual component
                - M_best: (2, 3) affine matrix or None
                - inlier_mask: (H, W) bool mask
        """
        assert flow_1img.dim() == 3 and flow_1img.size(0) == 2
        device = flow_1img.device
        _, H, W = flow_1img.shape
        
        # Pixel coordinate grid
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        src_pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).float()  # (N, 2)
        N = src_pts.size(0)
        
        # Destination points = source + flow
        flow_flat = flow_1img.permute(1, 2, 0).reshape(-1, 2)  # (N, 2)
        dst_pts = src_pts + flow_flat  # (N, 2)
        
        # Homogeneous coordinates
        ones = torch.ones((N, 1), device=device, dtype=src_pts.dtype)
        src_h = torch.cat([src_pts, ones], dim=1)  # (N, 3)
        
        best_num_inliers = 0
        best_inlier_mask = None
        best_M = None
        
        # RANSAC loop
        for it in range(self.num_iters):
            # Random sample
            idx = torch.randint(0, N, (self.sample_size,), device=device)
            A_sample = src_h[idx]  # (k, 3)
            b_sample = dst_pts[idx]  # (k, 2)
            
            # Least squares fit
            try:
                sol = torch.linalg.lstsq(A_sample, b_sample)
                M_T = sol.solution  # (3, 2)
            except AttributeError:
                M_T = torch.pinverse(A_sample) @ b_sample  # (3, 2)
            
            M = M_T.transpose(0, 1)  # (2, 3)
            
            # Compute reprojection errors
            dst_pred = (src_h @ M.T)  # (N, 2)
            errors = torch.norm(dst_pred - dst_pts, dim=1)  # (N,)
            
            inlier_mask = errors < self.inlier_thresh
            num_inliers = int(inlier_mask.sum().item())
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inlier_mask = inlier_mask
                best_M = M
        
        if self.verbose:
            print(f"RANSAC: {best_num_inliers} inliers, threshold: {self.min_inlier_ratio * N}")
        
        # Check if result is acceptable
        print(f"inlier_ratio: {best_num_inliers / N * 1.0}")
        if best_M is None or best_num_inliers < self.min_inlier_ratio * N:
            # Fallback: no rigid flow
            print("Too few inliers, fallback to no rigid flow")
            rigid_flow = torch.zeros_like(flow_1img)
            flow_res = flow_1img.clone()
            inlier_mask_map = torch.zeros((H, W), dtype=torch.bool, device=device)
            return rigid_flow, flow_res, None, inlier_mask_map
        
        # Refit with all inliers
        src_in = src_h[best_inlier_mask]  # (n_in, 3)
        dst_in = dst_pts[best_inlier_mask]  # (n_in, 2)
        
        try:
            sol_final = torch.linalg.lstsq(src_in, dst_in)
            M_T_final = sol_final.solution  # (3, 2)
        except AttributeError:
            M_T_final = torch.pinverse(src_in) @ dst_in
        
        M_best = M_T_final.transpose(0, 1)  # (2, 3)
        
        # Compute rigid flow
        dst_rigid = (src_h @ M_best.T)  # (N, 2)
        rigid_flow_flat = dst_rigid - src_pts  # (N, 2)
        rigid_flow = rigid_flow_flat.view(H, W, 2).permute(2, 0, 1)  # (2, H, W)
        
        # Residual = full - rigid
        flow_res = flow_1img - rigid_flow
        
        inlier_mask_map = best_inlier_mask.view(H, W)  # (H, W)
        
        return rigid_flow, flow_res, M_best, inlier_mask_map
    
    def estimate_batch(
        self,
        flow_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor]:
        """
        Estimate affine transformation for a batch of flows.
        
        Args:
            flow_batch: Flow tensor (B, 2, H, W)
        
        Returns:
            Tuple of (rigid_flow, flow_res, M_list, inlier_masks):
                - rigid_flow: (B, 2, H, W)
                - flow_res: (B, 2, H, W)
                - M_list: List of (2, 3) affine matrices or None
                - inlier_masks: (B, H, W) bool
        """
        assert flow_batch.dim() == 4 and flow_batch.size(1) == 2
        B, _, H, W = flow_batch.shape
        device = flow_batch.device
        dtype = flow_batch.dtype
        
        rigid_flow_all = torch.zeros_like(flow_batch)
        flow_res_all = torch.zeros_like(flow_batch)
        inlier_masks = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        M_list = []
        
        for b in range(B):
            flow_1img = flow_batch[b]  # (2, H, W)
            rigid_b, res_b, M_b, mask_b = self.estimate_single(flow_1img)
            rigid_flow_all[b] = rigid_b.to(dtype)
            flow_res_all[b] = res_b.to(dtype)
            inlier_masks[b] = mask_b
            M_list.append(M_b)
        
        return rigid_flow_all, flow_res_all, M_list, inlier_masks


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        pass
    
    @abstractmethod
    def get_batch(
        self, 
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load a batch of image pairs.
        
        Args:
            indices: List of indices to load
        
        Returns:
            Tuple of (img1_batch, img2_batch, metadata):
                - img1_batch: (B, C, H, W) first images
                - img2_batch: (B, C, H, W) second images
                - metadata: Dict with additional information (paths, etc.)
        """
        pass
    
    @abstractmethod
    def get_all_pairs(self) -> List[Tuple[Any, Any]]:
        """
        Get all image pairs in the dataset.
        
        Returns:
            List of tuples (img1_path, img2_path) or similar
        """
        pass


class FlowProcessor:
    """Main processor that orchestrates the entire flow estimation pipeline."""
    
    def __init__(
        self,
        device: str = 'cuda',
        resize_k: int = 8,
        ransac_config: Optional[Dict[str, Any]] = None,
        flow_model_type: str = 'large'
    ):
        """
        Args:
            device: Device to run on
            resize_k: Resize to nearest multiple of k
            ransac_config: Dict with RANSAC parameters (num_iters, sample_size, etc.)
            flow_model_type: RAFT model type
        """
        self.device = device
        self.resize_k = resize_k
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.flow_estimator = RAFTFlowEstimator(device=device, model_type=flow_model_type)
        
        # Default RANSAC config
        default_ransac = {
            'num_iters': 1000,
            'sample_size': 200,
            'inlier_thresh': 0.1,
            'min_inlier_ratio': 0.2,
            'verbose': False
        }
        if ransac_config:
            default_ransac.update(ransac_config)
        
        self.ransac_estimator = RANSACAffineEstimator(**default_ransac)
    
    def process_batch(
        self,
        img1_batch: torch.Tensor,
        img2_batch: torch.Tensor,
        compute_rigid: bool = True,
        compute_dynamic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of image pairs through the full pipeline.
        
        Args:
            img1_batch: First image batch (B, C, H, W) in [0, 1]
            img2_batch: Second image batch (B, C, H, W) in [0, 1]
            compute_rigid: Whether to compute rigid/residual decomposition
            compute_dynamic: Whether to compute dynamic flow (mean-subtracted)
        
        Returns:
            Dict with keys:
                - 'flow': Full optical flow (B, 2, H, W)
                - 'rigid_flow': Rigid component (if compute_rigid=True)
                - 'flow_res': Residual component (if compute_rigid=True)
                - 'dynamic_flow': Mean-subtracted flow (if compute_dynamic=True)
                - 'inlier_masks': Inlier masks (if compute_rigid=True)
                - 'affine_matrices': List of affine matrices (if compute_rigid=True)
        """
        # Preprocess
        img1_processed = self.preprocessor.preprocess(img1_batch).to(self.device)
        img2_processed = self.preprocessor.preprocess(img2_batch).to(self.device)
        
        # Resize
        img1_resized = self.preprocessor.resize_to_nearest_multiple(
            img1_processed, k=self.resize_k
        )
        img2_resized = self.preprocessor.resize_to_nearest_multiple(
            img2_processed, k=self.resize_k
        )
        img1_resized = img1_resized.contiguous()
        img2_resized = img2_resized.contiguous()
        # Estimate flow
        flow = self.flow_estimator.estimate_flow(img1_resized, img2_resized)
        
        results = {'flow': flow}
        
        # RANSAC decomposition
        if compute_rigid:
            rigid_flow, flow_res, M_list, inlier_masks = self.ransac_estimator.estimate_batch(flow)
            results.update({
                'rigid_flow': rigid_flow,
                'flow_res': flow_res,
                'inlier_masks': inlier_masks,
                'affine_matrices': M_list
            })
        
        # Dynamic flow (mean-subtracted)
        if compute_dynamic:
            mean_flow = torch.mean(flow, dim=(2, 3), keepdim=True)  # (B, 2, 1, 1)
            dynamic_flow = flow - mean_flow
            results['dynamic_flow'] = dynamic_flow
        
        return results
    
    def process_dataset(
        self,
        dataset_loader: DatasetLoader,
        batch_size: int = 1,
        save_dir: Optional[Path] = None
    ):
        """
        Process an entire dataset in batches.
        
        For each pair, saves results in individual folders:
        - pair_XXXXXX/scene_flow.pt: [H, W, 4] tensor (rigid_flow + flow_res)
        - pair_XXXXXX/flow_visualization.jpg: Full flow visualization
        - pair_XXXXXX/flow_res_visualization.jpg: Residual flow visualization
        
        Args:
            dataset_loader: DatasetLoader instance
            batch_size: Batch size for processing
            save_dir: Directory to save results (optional)
        """
        num_pairs = len(dataset_loader)
        all_results = []
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_idx in range(0, num_pairs, batch_size):
            end_idx = min(batch_idx + batch_size, num_pairs)
            indices = list(range(batch_idx, end_idx))
            
            # Load batch
            img1_batch, img2_batch, metadata = dataset_loader.get_batch(indices)
            
            # Process
            results = self.process_batch(
                img1_batch, 
                img2_batch,
                compute_rigid=True,
                compute_dynamic=True
            )
            
            # Add metadata
            results['metadata'] = metadata
            results['indices'] = indices
            
            all_results.append(results)
            
            # Save if requested
            if save_dir:
                self._save_pair_results(results, save_dir, indices)
        
        return all_results
    
    def _save_pair_results(
        self,
        results: Dict[str, Any],
        save_dir: Path,
        indices: List[int]
    ):
        """
        Save results for each pair individually.
        
        For each pair, creates a folder named after the first image (without extension)
        and organizes by scene to match input dataset structure.
        
        Saves:
        - scene_flow.pt: [H, W, 4] tensor with rigid_flow (channels 0-1) and flow_res (channels 2-3)
        - flow_visualization.jpg: Visualization of full flow
        - flow_res_visualization.jpg: Visualization of residual flow
        """
        B = results['flow'].shape[0]
        metadata = results['metadata']
        
        # Extract individual pairs from batch
        for i in range(B):
            # Get image paths from metadata
            img1_path = Path(metadata['img1_paths'][i])
            scene_name = metadata['scene_names'][i]
            
            # Get filename without extension
            img1_name = img1_path.stem  # filename without extension
            
            # Create directory structure: output_dir/scene_name/img1_name/
            pair_dir = save_dir / scene_name / img1_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Get individual pair results
            flow = results['flow'][i]  # (2, H, W)
            rigid_flow = results['rigid_flow'][i]  # (2, H, W)
            flow_res = results['flow_res'][i]  # (2, H, W)
            
            # Convert to (H, W, 2) format
            flow_hw = flow.permute(1, 2, 0)  # (H, W, 2)
            rigid_flow_hw = rigid_flow.permute(1, 2, 0)  # (H, W, 2)
            flow_res_hw = flow_res.permute(1, 2, 0)  # (H, W, 2)
            
            # Concatenate to create scene_flow: [H, W, 4]
            # Channels 0-1: rigid_flow, Channels 2-3: flow_res
            scene_flow = torch.cat([rigid_flow_hw, flow_res_hw], dim=2)  # (H, W, 4)
            scene_flow = scene_flow.to(torch.float16)
            
            # Save scene_flow.pt
            torch.save(scene_flow, pair_dir / 'scene_flow.pt')
            
            # Save flow visualization (full flow)
            flow_img = flow_to_image(flow.unsqueeze(0))  # (1, 3, H, W)
            save_image(flow_img.float() / 255.0, pair_dir / 'flow_visualization.jpg')
            
            # Save flow_res visualization
            flow_res_img = flow_to_image(flow_res.unsqueeze(0))  # (1, 3, H, W)
            save_image(flow_res_img.float() / 255.0, pair_dir / 'flow_res_visualization.jpg')

