import torch
import torch.nn.functional as F
from typing import Tuple, Optional

@torch.no_grad()
def mismatch_filter_lidar_vs_pred(
    z_pred: torch.Tensor,          # [B,1,H,W]  预测深度 (相对或近似绝对均可)
    z_lidar: torch.Tensor,         # [B,1,H,W]  稀疏 LiDAR 深度（无处为0或任意值）
    m_lidar: torch.Tensor,         # [B,1,H,W]  LiDAR 有效掩码 (bool 或 0/1)
    ksize: int = 7,                # 窗口大小（奇数）
    sigma: Optional[float] = None, # 高斯权重的标准差；None 则用 ksize/3
    tau_rel: float = 0.03,         # 残差相对阈值系数：tau = tau_rel * z_center + tau_abs
    tau_abs: float = 0.02,         # 残差绝对阈值（米等尺度单位）
    min_pts: int = 4,              # 每个窗口内用于回归的最小 LiDAR 点数
    robust: bool = False,          # 是否输出鲁棒权重（Huber）
    huber_delta: float = 0.1       # Huber 损失阈值（和深度单位一致）
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    返回:
      inlier_mask: [B,1,H,W]  True 表示 LiDAR 与预测局部可对齐（保留）
      residual:    [B,1,H,W]  中心像素残差 |a*zhat + b - zL|
      w_robust:    [B,1,H,W]  (可选) 鲁棒权重 (Huber 导出的权重)，robust=False 时为 None
    说明:
      - 采用图像域 k×k 小窗，把窗口内所有“有 LiDAR 的像素”参与回归，求 a,b。
      - 在 LiDAR 稀疏处（邻域内有效点不足 min_pts）退化为简单差分阈值。
      - 全流程张量化，适合放到 GPU 上批量跑。
    """
    assert z_pred.ndim == z_lidar.ndim == m_lidar.ndim == 4 and z_pred.shape == z_lidar.shape == m_lidar.shape
    B, _, H, W = z_pred.shape
    device = z_pred.device
    eps = 1e-8

    # 准备 Unfold
    pad = ksize // 2
    unfold = torch.nn.Unfold(kernel_size=ksize, padding=pad)

    # 展平为 [B, K, L]，K=ksize*ksize, L=H*W
    Zp   = unfold(z_pred)                      # 预测深度补丁
    Zl   = unfold(z_lidar)                     # LiDAR 深度补丁
    Ml   = unfold(m_lidar.float())             # LiDAR 掩码补丁 (0/1)

    B_, K, L = Zp.shape
    assert B_ == B

    # 高斯空间权重（仅按窗口坐标，不依赖内容）
    if sigma is None:
        sigma = max(ksize / 3.0, 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(ksize, device=device, dtype=torch.float32),
        torch.arange(ksize, device=device, dtype=torch.float32),
        indexing='ij'
    )
    cy = (ksize - 1) / 2.0
    cx = (ksize - 1) / 2.0
    g = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / (2.0 * sigma**2))  # [K_y, K_x]
    g = (g / (g.sum() + eps)).reshape(1, K, 1)                        # [1,K,1] 归一化
    Wgt = g * Ml                                                      # 只给有 LiDAR 的位置权重

    # 计算带权统计量（正规方程）
    # 目标: min Σ w (a*x + b - y)^2,  x=Zp, y=Zl, w=Wgt
    Sw  = (Wgt).sum(dim=1, keepdim=True)                     # [B,1,L]
    Sx  = (Wgt * Zp).sum(dim=1, keepdim=True)                # [B,1,L]
    Sy  = (Wgt * Zl).sum(dim=1, keepdim=True)                # [B,1,L]
    Sxx = (Wgt * Zp * Zp).sum(dim=1, keepdim=True)           # [B,1,L]
    Sxy = (Wgt * Zp * Zl).sum(dim=1, keepdim=True)           # [B,1,L]

    denom = (Sw * Sxx - Sx * Sx).clamp_min(eps)              # [B,1,L]
    a = (Sw * Sxy - Sx * Sy) / denom                         # [B,1,L]
    b = (Sy - a * Sx) / (Sw.clamp_min(eps))                  # [B,1,L]

    # 取中心像素（便捷：直接用原图展平）
    zc_hat = z_pred.view(B, 1, -1)                           # [B,1,L]
    zc_lid = z_lidar.view(B, 1, -1)                          # [B,1,L]
    mc_lid = m_lidar.view(B, 1, -1) > 0.5                    # [B,1,L] bool

    # 窗口有效点计数
    cnt = Ml.sum(dim=1, keepdim=True)                        # [B,1,L]

    # 残差（中心像素）：r = |a*zhat_c + b - zL_c|
    r = (a * zc_hat + b - zc_lid).abs()                      # [B,1,L]

    # 阈值（相对 + 绝对）
    tau = tau_rel * zc_lid.abs() + tau_abs                   # [B,1,L]

    # 回归方案仅在 “邻域内有效点数>=min_pts 且 中心有 LiDAR” 时启用
    use_reg = (cnt >= float(min_pts)) & mc_lid               # [B,1,L]

    # 对退化情况（邻域点太少），回退为简单阈值：|zhat_c - zL_c|
    r_fallback = (zc_hat - zc_lid).abs()
    r = torch.where(use_reg, r, r_fallback)

    # 最终 inlier 判定
    inlier = (r <= tau) & mc_lid                             # [B,1,L]
    inlier = inlier.view(B, 1, H, W)

    residual = r.view(B, 1, H, W)

    # (可选) 鲁棒权重：Huber 权重 w = dρ/dr / r
    w_robust = None
    if robust:
        d = huber_delta
        # ρ(r) = 0.5 r^2 (|r|<=d) ; ρ(r) = d(|r| - 0.5 d) (|r|>d)
        # 对应权重: w = 1                 (|r|<=d)
        #          = d/|r|               (|r|>d)
        r_safe = residual.clamp_min(1e-12)
        w_robust = torch.where(residual <= d, torch.ones_like(residual), (d / r_safe))
        # 若希望考虑深度随距离放宽，可把 d 也设为随深度增长的函数

    return inlier, residual, w_robust


# --- 辅助：灰度与边缘权重 ---
def rgb_to_gray(I):  # I: (B,3,H,W) in [0,1]
    w = torch.tensor([0.299, 0.587, 0.114], device=I.device, dtype=I.dtype)
    return (I * w[None,:,None,None]).sum(1, keepdim=True)

def edge_weights(I, sigma=0.1):
    """
    计算图像的边缘权重 w(x)=exp(-||∇I||²/sigma²)
    I: (B,1,H,W) 灰度 or (B,3,H,W) RGB，范围[0,1]
    """
    if I.size(1) == 3:  # RGB
        Ig = 0.299 * I[:,0:1] + 0.587 * I[:,1:2] + 0.114 * I[:,2:3]
    else:
        Ig = I

    # Sobel kernel (1,1,3,3)
    kx = torch.tensor([[[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]]], dtype=I.dtype, device=I.device)
    ky = torch.tensor([[[[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]]]], dtype=I.dtype, device=I.device)

    # 卷积计算梯度
    gx = F.conv2d(Ig, kx, padding=1)
    gy = F.conv2d(Ig, ky, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # (B,1,H,W)

    # 边缘权重
    w = torch.exp(- (grad / (sigma + 1e-8)) ** 2)
    w = torch.clamp(w, 1e-3, 1.0)
    return w

# --- 鲁棒全局(alpha,beta)作为种子 ---
def solve_scale_shift_huber(P, D, M, delta=0.1, iters=5):
    B = P.shape[0]
    alpha = torch.ones(B,1,1,1, device=P.device, dtype=P.dtype)
    beta  = torch.zeros_like(alpha)
    for _ in range(iters):
        r = (alpha*P + beta - D) * M
        w = torch.where(r.abs() <= delta, torch.ones_like(r), (delta/(r.abs()+1e-8)))
        WP  = (w*M*P).sum((2,3))
        W1  = (w*M).sum((2,3))
        WP2 = (w*M*P*P).sum((2,3))
        WD  = (w*M*D).sum((2,3))
        WPD = (w*M*P*D).sum((2,3))
        A11, A12 = WP2, WP
        A21, A22 = WP,  W1
        b1,  b2  = WPD, WD
        det = A11*A22 - A12*A21 + 1e-8
        alpha = ((b1*A22 - b2*A12) / det).view(B,1,1,1)
        beta  = ((A11*b2 - A21*b1) / det).view(B,1,1,1)
    return alpha, beta

# --- 稀疏场 -> 稠密场（各向异性扩散 + 屏蔽数据项） ---
def densify_field(I, field_raw, M, seed, lam=200., mu=2., gamma=8., sigma=0.1, iters=400):
    """
    I:   (B,3,H,W) 引导图
    field_raw: (B,1,H,W) 稀疏观测(仅在M=1有值)
    M:   (B,1,H,W) 0/1
    seed:(B,1,H,W) 全局种子场（常数或平滑先验）
    返回：field_dense (B,1,H,W)
    """
    B,_,H,W = field_raw.shape
    z = seed.clone()  # 以种子初始化，保证全图有值
    w = edge_weights(I, sigma)      # (B,1,H,W)

    def shift(x, dy, dx):
        return F.pad(x, (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0)), mode='replicate') \
               [:,:, max(-dy,0):x.shape[2]+max(-dy,0), max(-dx,0):x.shape[3]+max(-dx,0)]

    for _ in range(iters):
        z_up, z_down = shift(z,-1,0), shift(z,1,0)
        z_left, z_right = shift(z,0,-1), shift(z,0,1)
        # 平滑项
        smooth_num = w*(z_up + z_down + z_left + z_right)
        smooth_den = w*4.0
        denom = lam*M + mu + gamma*smooth_den + 1e-8
        numer = lam*M*field_raw + mu*seed + gamma*smooth_num
        z = numer / denom
    return z

# --- 主流程：s/b 双场稠密化 + 重建稠密深度 z ---
def align_with_spatial_scale_bias_dense(I_rgb, P, D, M,
                                        lam=250., mu=2., gamma=10., sigma=0.1,
                                        iters=400, clip=(2,98)):
    """
    I_rgb: (B,3,H,W) [0,1]
    P    : (B,1,H,W) 单目深度(任意尺度)
    D    : (B,1,H,W) 稀疏LiDAR深度
    M    : (B,1,H,W) 0/1 mask
    返回：z(稠密), s_dense, b_dense, alpha, beta
    """
    eps_div = 1e-6
    P_safe = torch.where(P.abs() < eps_div, torch.full_like(P, eps_div), P)
    s_raw = torch.zeros_like(P); b_raw = torch.zeros_like(P)
    s_raw[M.bool()] = (D[M.bool()] / P_safe[M.bool()])
    b_raw[M.bool()] = (D[M.bool()] - s_raw[M.bool()]*P[M.bool()])

    # 全局种子
    alpha, beta = solve_scale_shift_huber(P, D, M)
    s_seed = torch.ones_like(P) * alpha
    b_seed = torch.ones_like(P) * beta

    # 可选：鲁棒裁剪 s_raw / b_raw（只对 M=1 的样本）
    if clip is not None and M.any():
        def qclip(x, m, lo, hi):
            vals = x[m.bool()].flatten()
            lo_v = torch.quantile(vals, lo/100.0)
            hi_v = torch.quantile(vals, hi/100.0)
            x = torch.clamp(x, lo_v, hi_v)
            return x
        s_raw = qclip(s_raw, M, *clip)
        b_raw = qclip(b_raw, M, *clip)

    # 稠密化（每个场独立解）
    s_dense = densify_field(I_rgb, s_raw, M, s_seed, lam=lam, mu=mu, gamma=gamma, sigma=sigma, iters=iters)
    b_dense = densify_field(I_rgb, b_raw, M, b_seed, lam=lam, mu=mu, gamma=gamma, sigma=sigma, iters=iters)

    # 重建并在观测处硬覆盖
    z = s_dense * P + b_dense
    z = M*D + (1-M)*z
    return z, s_dense, b_dense, alpha, beta