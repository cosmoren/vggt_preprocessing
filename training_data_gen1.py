"""Lidar egoview visualization."""

import logging
import os
import sys
from pathlib import Path
from typing import Final

import click
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
import torchvision

import torch
from moge.model.v2 import MoGeModel
import torch.nn.functional as F


if __name__ == "__main__":

    # Initialize the MoGe model
    device = torch.device("cuda")
    MoGe_model = MoGeModel.from_pretrained("/work/weights/moge-2-vitl-normal/model.pt").to(device)  

    data_path = "/work/datasets/av2/test"
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

                # compute camera to city SE3 transform
                T_ego2city = np.eye(4)
                T_ego2city[0:3, 0:3] = city_SE3_ego.rotation
                T_ego2city[0:3, 3] = city_SE3_ego.translation
                T_cam2city = T_ego2city @ T_sensor2ego
                T_cam2city = T_cam2city.reshape(16)                  # trajectory cam to city pose

                poses_cam2city.append(T_cam2city)
            
                ori_image = cv2.cvtColor(cv2.imread( im_fpath ), cv2.COLOR_BGR2RGB)                       
                input_image = torch.tensor(ori_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1) 
                MoGe_output = MoGe_model.infer(input_image)
                masked_depth = MoGe_output['depth'].masked_fill(~MoGe_output['mask'], 300)
                depth_path = im_fpath.with_suffix('.png')
                depth_path_list = list(depth_path.parts)
                depth_path_list[-2] = depth_path_list[-2]+'_depth'
                depth_path = os.path.join(*depth_path_list)
                # np.save( depth_path, masked_depth.cpu().detach().numpy() )
                scaled_depth = (masked_depth.cpu().detach().numpy() / 300 * 65535).astype(np.uint16)
                cv2.imwrite(depth_path, scaled_depth)

            poses_cam2city = np.asarray(poses_cam2city)
            np.savetxt(os.path.join(data_path, scene_id, 'trajectory', camera+'.txt'), poses_cam2city)
        print(num, scene_id, 'done.')