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
import shutil

if __name__ == "__main__":

    # Initialize the MoGe model
    device = torch.device("cuda")
    # MoGe_model = MoGeModel.from_pretrained("/work/weights/moge-2-vitl-normal/model.pt").to(device)  

    data_path = "/work/datasets/av2/val"
    data_path_Path = Path(data_path)

    # loader = AV2SensorDataLoader(data_dir=data_path_Path, labels_dir=data_path_Path)

    scene_ids = sorted( os.listdir(data_path) )
    for num, scene_id in enumerate(scene_ids):
        if os.path.exists(os.path.join(data_path, scene_id, 'trajectory')):
            shutil.rmtree(os.path.join(data_path, scene_id, 'trajectory'))
        
        cam_folders = os.listdir(os.path.join(data_path, scene_id, 'sensors', 'cameras'))
        for cam_folder in cam_folders:
            if 'depth' in cam_folder:
                shutil.rmtree(os.path.join(data_path, scene_id, 'sensors', 'cameras', cam_folder))
        

