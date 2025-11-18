import torch
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import flow_to_image, save_image

def resize_to_nearest_multiple(x, k=32, mode='bilinear'):
    """
    x: [B, C, H, W]
    k: 想要的倍数，比如 8/16/32
    """
    B, C, H, W = x.shape
    new_H = round(H / k) * k
    new_W = round(W / k) * k

    new_H = max(new_H, k)
    new_W = max(new_W, k)

    if new_H == H and new_W == W:
        return x

    x_resized = F.interpolate(
        x, size=(new_H, new_W),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    return x_resized

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


def estimate_affine_ransac_single(
    flow_1img: torch.Tensor,
    num_iters: int = 1000,
    sample_size: int = 3,
    inlier_thresh: float = 1.0,
    min_inlier_ratio: float = 0.3,
):
    """
    对单张光流 (2,H,W) 做全局仿射 + RANSAC：
    输入:
        flow_1img: (2, H, W), flow[y,x] = (du, dv)
    输出:
        rigid_flow:   (2, H, W)
        flow_res:     (2, H, W)
        M_best:       (2, 3) 仿射矩阵, [ [a11,a12,tx],
                                        [a21,a22,ty] ]
        inlier_mask:  (H, W) bool, True = 被认为是刚体背景
    """
    assert flow_1img.dim() == 3 and flow_1img.size(0) == 2
    device = flow_1img.device
    _, H, W = flow_1img.shape

    # 像素坐标网格
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    # src_pts: (N, 2) = (x, y)
    src_pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).float()  # (N,2)
    N = src_pts.size(0)

    # 目标点 = 原始坐标 + 光流
    flow_flat = flow_1img.permute(1, 2, 0).reshape(-1, 2)  # (N,2)
    dst_pts = src_pts + flow_flat                           # (N,2)

    # 齐次 src_h: (N,3) = [x y 1]
    ones = torch.ones((N, 1), device=device, dtype=src_pts.dtype)
    src_h = torch.cat([src_pts, ones], dim=1)  # (N,3)

    best_num_inliers = 0
    best_inlier_mask = None
    best_M = None

    for it in range(num_iters):
        # 随机采样 sample_size 个点拟合仿射
        idx = torch.randint(0, N, (sample_size,), device=device)
        A_sample = src_h[idx]         # (k,3)
        b_sample = dst_pts[idx]       # (k,2)

        # 最小二乘拟合 M: (2,3)，通过 lstsq 解 A_sample @ M^T ≈ b_sample
        # 解的是 M_T: (3,2)
        try:
            # torch.linalg.lstsq 在新版本 PyTorch 里可用
            sol = torch.linalg.lstsq(A_sample, b_sample)
            M_T = sol.solution  # (3,2)
        except AttributeError:
            # 老版本 PyTorch 没有 linalg.lstsq，用 pinv 替代
            M_T = torch.pinverse(A_sample) @ b_sample  # (3,2)

        M = M_T.transpose(0, 1)  # (2,3)

        # 对所有点计算重投影误差
        dst_pred = (src_h @ M.T)         # (N,2)
        errors = torch.norm(dst_pred - dst_pts, dim=1)  # (N,)

        inlier_mask = errors < inlier_thresh
        num_inliers = int(inlier_mask.sum().item())
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_M = M

    print(best_num_inliers, min_inlier_ratio, N, min_inlier_ratio*N )

    # 检查 RANSAC 结果是否可信
    if best_M is None or best_num_inliers < min_inlier_ratio * N:
        # 动态太多 or 光流太吵 → fallback：刚体流=0，残差=full flow
        rigid_flow = torch.zeros_like(flow_1img)
        flow_res = flow_1img.clone()
        inlier_mask_map = torch.zeros((H, W), dtype=torch.bool, device=device)
        return rigid_flow, flow_res, None, inlier_mask_map

    # 用所有内点重新精拟合一次
    src_in = src_h[best_inlier_mask]   # (n_in,3)
    dst_in = dst_pts[best_inlier_mask] # (n_in,2)

    try:
        sol_final = torch.linalg.lstsq(src_in, dst_in)
        M_T_final = sol_final.solution  # (3,2)
    except AttributeError:
        M_T_final = torch.pinverse(src_in) @ dst_in

    M_best = M_T_final.transpose(0, 1)  # (2,3)

    # 用最终 M_best 算所有像素的刚体对应位置
    dst_rigid = (src_h @ M_best.T)       # (N,2)
    rigid_flow_flat = dst_rigid - src_pts  # (N,2)
    rigid_flow = rigid_flow_flat.view(H, W, 2).permute(2, 0, 1)  # (2,H,W)

    # 残差 = full - rigid
    flow_res = flow_1img - rigid_flow

    inlier_mask_map = best_inlier_mask.view(H, W)  # (H,W)

    return rigid_flow, flow_res, M_best, inlier_mask_map


def estimate_affine_ransac_batch(
    flow_batch: torch.Tensor,
    num_iters: int = 1000,
    sample_size: int = 3,
    inlier_thresh: float = 1.0,
    min_inlier_ratio: float = 0.3,
):
    """
    flow_batch: (B, 2, H, W)
    返回:
        rigid_flow:  (B, 2, H, W)
        flow_res:    (B, 2, H, W)
        M_list:      长度为 B 的仿射矩阵列表，每个元素 (2,3) 或 None
        inlier_masks:(B, H, W) bool
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
        flow_1img = flow_batch[b]  # (2,H,W)
        rigid_b, res_b, M_b, mask_b = estimate_affine_ransac_single(
            flow_1img,
            num_iters=num_iters,
            sample_size=sample_size,
            inlier_thresh=inlier_thresh,
            min_inlier_ratio=min_inlier_ratio,
        )
        rigid_flow_all[b] = rigid_b.to(dtype)
        flow_res_all[b] = res_b.to(dtype)
        inlier_masks[b] = mask_b
        M_list.append(M_b)

    return rigid_flow_all, flow_res_all, M_list, inlier_masks


device = 'cuda'

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

img1_batch = preprocess( read_image('/work/yuan/zero-msf/bus/frame_0.jpg')[None,:,:,:] ).to(device)
img2_batch = preprocess( read_image('/work/yuan/zero-msf/bus/frame_1.jpg')[None,:,:,:] ).to(device)

img1_batch = resize_to_nearest_multiple(img1_batch, k=8)
img2_batch = resize_to_nearest_multiple(img2_batch, k=8)

list_of_flows = model(img1_batch, img2_batch)
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1] # optical flow 
print(predicted_flows.shape)

rigid_flow, flow_res, M_list, inlier_masks = estimate_affine_ransac_batch(
    predicted_flows,
    num_iters=1000,
    sample_size=200,
    inlier_thresh=120.0,      # 可以根据分辨率调，例如 0.5~2 像素
    min_inlier_ratio=0.2,
)

print(M_list)

mean_flow = torch.mean(predicted_flows, dim=(2,3))
dyn_flow = torch.zeros_like(predicted_flows)
dyn_flow[:,0,:,:] = predicted_flows[:,0,:,:] - mean_flow[0,0]
dyn_flow[:,1,:,:] = predicted_flows[:,1,:,:] - mean_flow[0,1]

flow_imgs = flow_to_image(dyn_flow)
save_image(flow_imgs.float()/255, 'aaa.jpg')