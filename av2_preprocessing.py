"""Lidar egoview visualization."""

import logging
import os
import sys
from pathlib import Path
from typing import Final
import argparse

import cv2
import numpy as np

import av2.rendering.color as color_utils
import av2.rendering.rasterize as raster_rendering_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt
import pandas as pd
from scipy.spatial.transform import Rotation as R

import torch
from moge.model.v2 import MoGeModel
import torch.nn.functional as F

from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

from mismatch_filter import mismatch_filter_lidar_vs_pred, align_with_spatial_scale_bias_dense

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moge_model_path", type=str, default="/mnt/nfs/SpatialAI/weights/moge-2-vitl-normal/model.pt")
    parser.add_argument("--av2_data_path", type=str, default="/mnt/nfs/SpatialAI/Datasets/av2/test")
    args = parser.parse_args()

    NUM_RANGE_BINS: Final[int] = 50
    RING_CAMERA_FPS: Final[int] = 20
    
    # repeat red to green colormap every 50 m.
    colors_arr_rgb = color_utils.create_colormap(
        color_list=[RED_HEX, GREEN_HEX], n_colors=NUM_RANGE_BINS
    )
    colors_arr_rgb = (colors_arr_rgb * 255).astype(np.uint8)
    colors_arr_bgr: NDArrayByte = np.fliplr(colors_arr_rgb)

    # Initialize the MoGe model
    device = torch.device("cuda")
    MoGe_model = MoGeModel.from_pretrained(args.moge_model_path).to(device)  

    data_path = args.av2_data_path
    print('MoGe-2 model path: ', args.moge_model_path )
    print('AV2 data path: ', args.av2_data_path ) 
    data_path_Path = Path(data_path)

    loader = AV2SensorDataLoader(data_dir=data_path_Path, labels_dir=data_path_Path)

    scene_ids = sorted( os.listdir(data_path) )
    for num, scene_id in enumerate(scene_ids):
        if os.path.exists(os.path.join(data_path, scene_id, 'trajectory')):
            print('Skip ', scene_id)
            continue

        print(num, scene_id)
        # read sensor calibration file
        extr_df = pd.read_feather( os.path.join(data_path, scene_id, "calibration", "egovehicle_SE3_sensor.feather") )
        intr_df = pd.read_feather( os.path.join(data_path, scene_id, "calibration", "intrinsics.feather") )
        
        cameras = os.listdir(os.path.join(data_path, scene_id, 'sensors', 'cameras'))
        os.makedirs( os.path.join(data_path, scene_id, 'trajectory'), exist_ok = True)

        for _, camera in enumerate(list(RingCameras)):
            print(camera)
            if 'depth' in camera or 'center' not in camera:
                continue
            camera_depth = camera+'_depth'
            os.makedirs( os.path.join(data_path, scene_id, 'sensors', 'cameras', camera_depth), exist_ok=True )

            cam_im_fpaths = loader.get_ordered_log_cam_fpaths(scene_id, camera)
            num_cam_imgs = len(cam_im_fpaths)

            # compute camera to ego SE3 transform
            cam_extr = extr_df[extr_df["sensor_name"] == camera]
            quat = [cam_extr['qx'].iloc[0].item(), cam_extr['qy'].iloc[0].item(), cam_extr['qz'].iloc[0].item(), cam_extr['qw'].iloc[0].item()]
            trans = [cam_extr['tx_m'].iloc[0].item(), cam_extr['ty_m'].iloc[0].item(), cam_extr['tz_m'].iloc[0].item()]
            R_mat = R.from_quat(quat).as_matrix()
            T_sensor2ego = np.eye(4)
            T_sensor2ego[:3,:3] = R_mat
            T_sensor2ego[:3,3] = trans

            if os.path.exists(os.path.join(data_path, scene_id, 'trajectory', camera+'.txt')):
                os.remove(os.path.join(data_path, scene_id, 'trajectory', camera+'.txt'))
            
            poses_cam2city = []

            for i, im_fpath in enumerate(cam_im_fpaths):
                cam_timestamp_ns = int(im_fpath.stem)
                city_SE3_ego = loader.get_city_SE3_ego(scene_id, cam_timestamp_ns)
                if city_SE3_ego is None:
                    logger.exception("missing LiDAR pose")
                    continue

                # load feather file path, e.g. '315978406032859416.feather"
                lidar_fpath = loader.get_closest_lidar_fpath(scene_id, cam_timestamp_ns)
                if lidar_fpath is None:
                    logger.info(
                        "No LiDAR sweep found within the synchronization interval for %s, so skipping...",
                        camera,
                    )
                    continue

                lidar_points_ego = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
                lidar_timestamp_ns = int(lidar_fpath.stem)

                # motion compensate always
                (
                    uv,
                    points_cam,
                    is_valid_points,
                ) = loader.project_ego_to_img_motion_compensated(
                    points_lidar_time=lidar_points_ego,
                    cam_name=camera,
                    cam_timestamp_ns=cam_timestamp_ns,
                    lidar_timestamp_ns=lidar_timestamp_ns,
                    log_id=scene_id,
                )

                if is_valid_points is None or uv is None or points_cam is None:
                    continue

                if is_valid_points.sum() == 0:
                    continue

                uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(np.int32)
                points_cam = points_cam[is_valid_points]
                pt_ranges: NDArrayFloat = np.linalg.norm(points_cam[:, :3], axis=1)
                color_bins: NDArrayInt = np.round(pt_ranges).astype(np.int32)
                # account for moving past 100 meters, loop around again
                color_bins = color_bins % NUM_RANGE_BINS
                uv_colors_bgr = colors_arr_bgr[color_bins]

                img_bgr = io_utils.read_img(im_fpath, channel_order="BGR")

                img_empty = np.full_like(img_bgr, fill_value=255)
                img_empty = raster_rendering_utils.draw_points_xy_in_img(
                    img_empty, uv_int, uv_colors_bgr, diameter=5
                )
                blended_bgr = raster_utils.blend_images(img_bgr, img_empty)
                frame_rgb = blended_bgr[:, :, ::-1]

                cv2.imwrite("aa.jpg", blended_bgr)

                # compute camera to city SE3 transform
                T_ego2city = np.eye(4)
                T_ego2city[0:3, 0:3] = city_SE3_ego.rotation
                T_ego2city[0:3, 3] = city_SE3_ego.translation
                T_cam2city = T_ego2city @ T_sensor2ego
                T_cam2city = T_cam2city.reshape(16)                  # trajectory cam to city pose

                poses_cam2city.append(T_cam2city)

                # read image            
                ori_image = cv2.cvtColor(cv2.imread( im_fpath ), cv2.COLOR_BGR2RGB)                       
                input_image = torch.tensor(ori_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1) 
                MoGe_output = MoGe_model.infer(input_image)
                masked_depth = MoGe_output['depth'].masked_fill(~MoGe_output['mask'], 500)
                depth_path = im_fpath.with_suffix('.png')
                depth_path_list = list(depth_path.parts)
                depth_path_list[-2] = depth_path_list[-2]+'_depth'
                depth_path = os.path.join(*depth_path_list)
                # np.save( depth_path, masked_depth.cpu().detach().numpy() )
                scaled_depth = (masked_depth.cpu().detach().numpy() / 300 * 65535).astype(np.uint16)
                # cv2.imwrite(depth_path, scaled_depth)

                # masked_depth[None,None,:,:] # MoGe-2 predicted depth
                z_lidar = torch.zeros_like(masked_depth[None,None,:,:])
                z_lidar[0, 0, uv_int[:,1], uv_int[:,0]] = torch.tensor( pt_ranges, dtype=torch.float32, device=device )
                m_lidar = torch.zeros_like(masked_depth[None,None,:,:], dtype=torch.bool)
                m_lidar[0, 0, uv_int[:,1], uv_int[:,0]] = True

                inlier, residual, w_robust = mismatch_filter_lidar_vs_pred( masked_depth[None,None,:,:],  # [B,1,H,W]  预测深度 (相对或近似绝对均可)
                                                                            z_lidar,                # [B,1,H,W]  稀疏 LiDAR 深度（无处为0或任意值）
                                                                            m_lidar,                # [B,1,H,W]  LiDAR 有效掩码 (bool 或 0/1)
                                                                            ksize = 11,             # 窗口大小（奇数）
                                                                            sigma = None,           # 高斯权重的标准差；None 则用 ksize/3
                                                                            tau_rel = 0.15,          # 残差相对阈值系数：tau = tau_rel * z_center + tau_abs
                                                                            tau_abs = 0.4,          # 残差绝对阈值（米等尺度单位）
                                                                            min_pts = 4,            # 每个窗口内用于回归的最小 LiDAR 点数
                                                                            robust = False,         # 是否输出鲁棒权重（Huber）
                                                                            huber_delta = 0.008     # Huber 损失阈值（和深度单位一致） 
                                                                        )

                dense_depth, s_dense, b_dense, alpha, beta = align_with_spatial_scale_bias_dense(
                    input_image[None,:,:,:], masked_depth[None,None,:,:], z_lidar, inlier.float(),
                    lam=250.,   # 观测项力度：越大在LiDAR处越贴合
                    mu=2.,      # 种子约束：小，避免过度拉向全局
                    gamma=10.,  # 平滑力度：越大越“铺开”
                    sigma=0.1,  # 边缘权重敏感度：越小越贴边
                    iters=400,  # 迭代步数：200~800
                    clip=(2,98) # 去掉极端离群比例
                )

                depth_path = im_fpath.with_suffix('.png')
                depth_path_list = list(depth_path.parts)
                depth_path_list[-2] = depth_path_list[-2]+'_depth'
                depth_path = os.path.join(*depth_path_list)

                dense_depth = torch.squeeze(torch.clamp(dense_depth, max=300.0))
                scaled_depth = (dense_depth.cpu().detach().numpy() / 300 * 65535).astype(np.uint16)
                cv2.imwrite(depth_path, scaled_depth)
                
            poses_cam2city = np.asarray(poses_cam2city)
            np.savetxt(os.path.join(data_path, scene_id, 'trajectory', camera+'.txt'), poses_cam2city)
        print(num, scene_id, 'done.')