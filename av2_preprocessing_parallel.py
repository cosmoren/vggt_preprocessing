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
import multiprocessing as mp

from mismatch_filter import mismatch_filter_lidar_vs_pred, align_with_spatial_scale_bias_dense

logger = logging.getLogger(__name__)

def process_scene(scene_ids: list, gpu_id: int, args):
    # Bind process to GPU
    print('GPU: ', gpu_id, ' Number of scene_id: ', len(scene_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    device = torch.device("cuda:0")

    data_path = args.av2_data_path
    print('MoGe-2 model path: ', args.moge_model_path )
    print('AV2 data path: ', args.av2_data_path ) 
    data_path_Path = Path(data_path)

    # Load MoGe model
    MoGe_model = MoGeModel.from_pretrained(args.moge_model_path).to(device).eval()
    loader = AV2SensorDataLoader(data_dir=Path(args.av2_data_path), labels_dir=Path(args.av2_data_path))

    for num, scene_id in enumerate(scene_ids):
        if os.path.exists(os.path.join(data_path, scene_id, 'trajectory')):
            print('Skip ', scene_id)
            continue

        print('GPU ', gpu_id, 'is processing its ', num, '/', len(scene_ids), ' scene ', scene_id)
        # read sensor calibration file
        extr_df = pd.read_feather( os.path.join(data_path, scene_id, "calibration", "egovehicle_SE3_sensor.feather") )
        intr_df = pd.read_feather( os.path.join(data_path, scene_id, "calibration", "intrinsics.feather") )
        
        cameras = os.listdir(os.path.join(data_path, scene_id, 'sensors', 'cameras'))
        os.makedirs( os.path.join(data_path, scene_id, 'trajectory'), exist_ok = True)

        try:
            data_path = args.av2_data_path
            extr_df = pd.read_feather(os.path.join(data_path, scene_id, "calibration", "egovehicle_SE3_sensor.feather"))
            intr_df = pd.read_feather(os.path.join(data_path, scene_id, "calibration", "intrinsics.feather"))

            os.makedirs(os.path.join(data_path, scene_id, 'trajectory'), exist_ok=True)

            for _, camera in enumerate(list(RingCameras)):
                if 'depth' in camera: # or 'center' not in camera:
                    continue
                camera_depth = camera + '_depth'
                camera_sparse_depth = camera + '_sparse_depth'
                os.makedirs(os.path.join(data_path, scene_id, 'sensors', 'cameras', camera_sparse_depth), exist_ok=True)
                os.makedirs(os.path.join(data_path, scene_id, 'sensors', 'cameras', camera_depth), exist_ok=True)

                cam_im_fpaths = loader.get_ordered_log_cam_fpaths(scene_id, camera)
                # Compute T_sensor2ego
                cam_extr = extr_df[extr_df["sensor_name"] == camera]
                quat = [cam_extr['qx'].iloc[0].item(), cam_extr['qy'].iloc[0].item(), cam_extr['qz'].iloc[0].item(), cam_extr['qw'].iloc[0].item()]
                trans = [cam_extr['tx_m'].iloc[0].item(), cam_extr['ty_m'].iloc[0].item(), cam_extr['tz_m'].iloc[0].item()]
                R_mat = R.from_quat(quat).as_matrix()
                T_sensor2ego = np.eye(4)
                T_sensor2ego[:3, :3] = R_mat
                T_sensor2ego[:3, 3] = trans

                traj_path = os.path.join(data_path, scene_id, 'trajectory', camera + '.txt')
                if os.path.exists(traj_path): os.remove(traj_path)
                poses_cam2city = []

                for i, im_fpath in enumerate(cam_im_fpaths):
                    cam_timestamp_ns = int(im_fpath.stem)
                    city_SE3_ego = loader.get_city_SE3_ego(scene_id, cam_timestamp_ns)
                    if city_SE3_ego is None: continue
                    lidar_fpath = loader.get_closest_lidar_fpath(scene_id, cam_timestamp_ns)
                    if lidar_fpath is None: continue

                    lidar_points_ego = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
                    lidar_timestamp_ns = int(lidar_fpath.stem)
                    uv, points_cam, is_valid_points = loader.project_ego_to_img_motion_compensated(
                        points_lidar_time=lidar_points_ego, cam_name=camera,
                        cam_timestamp_ns=cam_timestamp_ns, lidar_timestamp_ns=lidar_timestamp_ns, log_id=scene_id
                    )
                    if is_valid_points is None or uv is None or points_cam is None: continue
                    if is_valid_points.sum() == 0: continue

                    uv_int = np.round(uv[is_valid_points]).astype(np.int32)
                    points_cam = points_cam[is_valid_points]
                    pt_ranges = np.linalg.norm(points_cam[:, :3], axis=1)

                    # MoGe depth estimation 
                    ori_image = cv2.cvtColor(cv2.imread(im_fpath), cv2.COLOR_BGR2RGB)
                    input_image = torch.tensor(ori_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
                    with torch.inference_mode():
                        MoGe_output = MoGe_model.infer(input_image)
                    masked_depth = MoGe_output['depth'].masked_fill(~MoGe_output['mask'], 500)

                    # mismatch filter + densification
                    z_lidar = torch.zeros_like(masked_depth[None, None, :, :], device=device)
                    z_lidar[0, 0, uv_int[:, 1], uv_int[:, 0]] = torch.tensor(pt_ranges, dtype=torch.float32, device=device)
                    m_lidar = torch.zeros_like(masked_depth[None, None, :, :], dtype=torch.bool, device=device)
                    m_lidar[0, 0, uv_int[:, 1], uv_int[:, 0]] = True

                    inlier, residual, w_robust = mismatch_filter_lidar_vs_pred(
                        masked_depth[None, None, :, :], z_lidar, m_lidar,
                        ksize=11, sigma=None, tau_rel=0.15, tau_abs=0.4, min_pts=4, robust=False, huber_delta=0.008
                    )
                    sparse_depth = z_lidar * inlier.float()

                    dense_depth, s_dense, b_dense, alpha, beta = align_with_spatial_scale_bias_dense(
                        input_image[None, :, :, :], masked_depth[None, None, :, :], z_lidar, inlier.float(),
                        lam=250., mu=2., gamma=10., sigma=0.1, iters=400, clip=(2, 98)
                    )

                    # Save sparse & dense depth
                    sparse_depth_path = im_fpath.with_suffix('.png')
                    path_parts = list(sparse_depth_path.parts); path_parts[-2] = path_parts[-2] + '_sparse_depth'
                    sparse_depth_path = os.path.join(*path_parts)
                    sd = torch.squeeze(torch.clamp(sparse_depth, max=300.0))
                    cv2.imwrite(sparse_depth_path, (sd.detach().cpu().numpy() / 300 * 65535).astype(np.uint16))

                    depth_path = im_fpath.with_suffix('.png')
                    path_parts = list(depth_path.parts); path_parts[-2] = path_parts[-2] + '_depth'
                    depth_path = os.path.join(*path_parts)
                    dd = torch.squeeze(torch.clamp(dense_depth, max=300.0))
                    cv2.imwrite(depth_path, (dd.detach().cpu().numpy() / 300 * 65535).astype(np.uint16))

                    # Trajectory
                    T_ego2city = np.eye(4)
                    T_ego2city[0:3, 0:3] = city_SE3_ego.rotation
                    T_ego2city[0:3, 3]   = city_SE3_ego.translation
                    T_cam2city = T_ego2city @ T_sensor2ego
                    poses_cam2city.append(T_cam2city.reshape(16))

                if len(poses_cam2city):
                    poses_cam2city = np.asarray(poses_cam2city)
                    np.savetxt(traj_path, poses_cam2city)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error in scene {scene_id}: {e}", flush=True)

# PYTHONUNBUFFERED=1 python av2_preprocessing_parallel.py --av2_data_path /mnt/nfs/SpatialAI/Datasets/av2_temp/train --gpus 0 1 2 3 4 5 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moge_model_path", type=str, default="/mnt/nfs/SpatialAI/weights/moge-2-vitl-normal/model.pt")
    parser.add_argument("--av2_data_path", type=str, default="/mnt/nfs/SpatialAI/Datasets/av2/train")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU id list, for example --gpus 0 1 2 3")
    args = parser.parse_args()

    # List all the scenes
    data_path = args.av2_data_path
    scene_ids = sorted([s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))])

    # GPU list
    if args.gpus is None:
        num = torch.cuda.device_count()
        gpu_list = list(range(num))
    else:
        gpu_list = args.gpus

    # Asign scenes to each GPU
    def chunks(lst, n):
        k = len(lst) // n
        for i in range(n):
            yield lst[i*k:(i+1)*k] if i < n-1 else lst[i*k:]

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id, subsets in zip(gpu_list, chunks(scene_ids, len(gpu_list))):
        p = mp.Process(target=process_scene, args=(subsets, gpu_id, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()