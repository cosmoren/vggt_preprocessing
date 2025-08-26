# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

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

logger = logging.getLogger(__name__)


NUM_RANGE_BINS: Final[int] = 50
RING_CAMERA_FPS: Final[int] = 20

def sobel_filters(device='cuda', dtype=torch.float32):
    """生成 Sobel 卷积核（形状 [out_channels,in_channels,H,W]）"""
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=dtype, device=device)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=dtype, device=device)
    # 将核扩展到 [1,1,H,W] 以便于 conv2d 调用
    return sobel_x.view(1, 1, 3, 3), sobel_y.view(1, 1, 3, 3)

def image_gradients(img):
    """
    计算灰度图像的梯度。
    img: Tensor, shape [B,1,H,W], dtype float32
    返回 grad_x, grad_y, magnitude
    """
    device, dtype = img.device, img.dtype
    sobel_x, sobel_y = sobel_filters(device, dtype)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    magnitude = (grad_x**2 + grad_y**2).sqrt()
    return grad_x, grad_y, magnitude


def generate_egoview_overlaid_lidar(
    data_root: Path,
    output_dir: Path,
    log_id: str,
    MoGe_model: MoGeModel,
    render_ground_pts_only: bool,
    dump_single_frames: bool,
) -> None:
    """Render LiDAR points from a particular camera's viewpoint (color by ground surface, and apply ROI filtering).

    Args:
        data_root: path to directory where the logs live on disk.
        output_dir: path to directory where renderings will be saved.
        log_id: unique ID for AV2 scenario/log.
        render_ground_pts_only: whether to only render LiDAR points located close to the ground surface.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering, in addition to generating
            the mp4 file.

    Raises:
        RuntimeError: If vehicle log data is not present at `data_root` for `log_id`.
    """
    # read sensor calibration file
    extr_df = pd.read_feather( os.path.join(data_root, log_id, "calibration", "egovehicle_SE3_sensor.feather") )
    intr_df = pd.read_feather( os.path.join(data_root, log_id, "calibration", "intrinsics.feather") )

    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    # repeat red to green colormap every 50 m.
    colors_arr_rgb = color_utils.create_colormap(
        color_list=[RED_HEX, GREEN_HEX], n_colors=NUM_RANGE_BINS
    )
    colors_arr_rgb = (colors_arr_rgb * 255).astype(np.uint8)
    colors_arr_bgr: NDArrayByte = np.fliplr(colors_arr_rgb)

    for _, cam_name in enumerate(list(RingCameras)):
        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        # compute camera to ego SE3 transform
        cam_extr = extr_df[extr_df["sensor_name"] == cam_name]
        quat = [cam_extr['qx'].iloc[0].item(), cam_extr['qy'].iloc[0].item(), cam_extr['qz'].iloc[0].item(), cam_extr['qw'].iloc[0].item()]
        trans = [cam_extr['tx_m'].iloc[0].item(), cam_extr['ty_m'].iloc[0].item(), cam_extr['tz_m'].iloc[0].item()]
        R_mat = R.from_quat(quat).as_matrix()
        T_sensor2ego = np.eye(4)
        T_sensor2ego[:3,:3] = R_mat
        T_sensor2ego[:3,3] = trans

        video_list = []
        for i, im_fpath in enumerate(cam_im_fpaths):
            if i % 50 == 0:
                logging.info(
                    f"\tOn file {i}/{num_cam_imgs} of camera {cam_name} of {log_id}"
                )

            cam_timestamp_ns = int(im_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                logger.exception("missing LiDAR pose")
                continue

            # compute camera to city SE3 transform
            T_ego2city = np.eye(4)
            T_ego2city[0:3, 0:3] = city_SE3_ego.rotation
            T_ego2city[0:3, 3] = city_SE3_ego.translation
            T_cam2city = T_ego2city @ T_sensor2ego

            # load feather file path, e.g. '315978406032859416.feather"
            lidar_fpath = loader.get_closest_lidar_fpath(log_id, cam_timestamp_ns)
            if lidar_fpath is None:
                logger.info(
                    "No LiDAR sweep found within the synchronization interval for %s, so skipping...",
                    cam_name,
                )
                continue

            img_bgr = io_utils.read_img(im_fpath, channel_order="BGR")
            
            # using MoGe-2 model to estimate the depth 
            ori_image = cv2.cvtColor(cv2.imread(im_fpath), cv2.COLOR_BGR2RGB)                       
            input_image = torch.tensor(ori_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1) 
            MoGe_output = MoGe_model.infer(input_image)
            masked_depth = MoGe_output['depth'].masked_fill(~(MoGe_output['depth']<200), 300)
            grad_x_gt, grad_y_gt, magnitude_gt = image_gradients(torch.log(masked_depth)[None, None, :, :])

            edge_mask = torch.ones(magnitude_gt.shape, dtype=torch.float32, device="cuda")
            edge_mask = edge_mask.masked_fill(magnitude_gt>0.1, 0)
            # torchvision.utils.save_image(kkk, 'ggggg.png')
            
            # torchvision.utils.save_image(magnitude_gt, 'depth.png', normalize=True, scale_each=True)

            lidar_points_ego = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
            lidar_timestamp_ns = int(lidar_fpath.stem)

            # motion compensate always
            (
                uv,
                points_cam,
                is_valid_points,
            ) = loader.project_ego_to_img_motion_compensated(
                points_lidar_time=lidar_points_ego,
                cam_name=cam_name,
                cam_timestamp_ns=cam_timestamp_ns,
                lidar_timestamp_ns=lidar_timestamp_ns,
                log_id=log_id,
            )

            if is_valid_points is None or uv is None or points_cam is None:
                continue

            if is_valid_points.sum() == 0:
                continue

            uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(np.int32)
            points_cam = points_cam[is_valid_points]
            pt_ranges: NDArrayFloat = np.linalg.norm(points_cam[:, :3], axis=1)
            color_bins: NDArrayInt = np.round(pt_ranges).astype(np.int32)

            # 
            uv_int_torch = torch.tensor(uv_int).to("cuda")
            pt_ranges_torch = torch.tensor(pt_ranges.astype(np.float32)).to("cuda")
            pt_ranges_torch.requires_grad_(True)

            # depth_opt = torch.clone(masked_depth)
            weights = torch.exp(-pt_ranges_torch)/torch.sum(torch.exp(-pt_ranges_torch))
            raw_scales = torch.clamp( (pt_ranges_torch/masked_depth[uv_int_torch[:,1], uv_int_torch[:,0]]), min=0.8, max=1.2)

            scale = torch.sum( raw_scales*weights )
            scale = scale.detach()
            
            depth_opt = torch.zeros([masked_depth.shape[0], masked_depth.shape[1]], dtype=torch.float32, device="cuda")
            depth_opt = masked_depth * 1.08
            depth_opt.requires_grad_(True)

            pc = (MoGe_output['points']*1.08)[MoGe_output['depth']<200]
            
            np.savetxt( 'fixed.txt', pc.cpu().numpy()[0::20, :] )

            for j in range(500000):
                grad_x, grad_y, magnitude = image_gradients(torch.log(depth_opt)[None, None, :, :])
                magnitude_norm = torch.linalg.norm( (magnitude-magnitude_gt) * edge_mask[None, None, :, :] )
                lidar_norm = torch.linalg.norm(torch.log(depth_opt[uv_int_torch[:,1], uv_int_torch[:,0]])-torch.log(pt_ranges_torch))
                loss = magnitude_norm + lidar_norm
                loss.backward()

                with torch.no_grad():
                    depth_opt_grad = torch.nan_to_num(depth_opt.grad , nan=0.0)
                    depth_opt -= depth_opt_grad * 1
                    depth_opt.grad = None

                if j%10000==0:
                    depth_opt1 = torch.clone(depth_opt)
                    depth_opt1[uv_int_torch[:,1], uv_int_torch[:,0]] = pt_ranges_torch
                    torchvision.utils.save_image(depth_opt1, str(j)+'.png', normalize=True, scale_each=True)

                print(j, loss, magnitude_norm, lidar_norm)
            

            color = np.zeros_like(points_cam[:, :3])
            color[:, 0] = 255
            np.savetxt('lidar.txt', np.hstack((points_cam[:, :3], color)))
            sys.exit()

            # account for moving past 100 meters, loop around again
            color_bins = color_bins % NUM_RANGE_BINS
            uv_colors_bgr = colors_arr_bgr[color_bins]

            img_empty = np.full_like(img_bgr, fill_value=255)
            img_empty = raster_rendering_utils.draw_points_xy_in_img(
                img_empty, uv_int, uv_colors_bgr, diameter=3 # 10
            )
            blended_bgr = raster_utils.blend_images(img_bgr, img_empty)
            frame_rgb = blended_bgr[:, :, ::-1]

            if dump_single_frames:
                save_dir = output_dir / log_id / cam_name
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(
                    str(save_dir / f"{cam_name}_{lidar_timestamp_ns}.jpg"), blended_bgr
                )

            video_list.append(frame_rgb)

        if len(video_list) == 0:
            raise RuntimeError(
                "No video frames were found; log data was not found on disk."
            )

        video: NDArrayByte = np.stack(video_list).astype(np.uint8)
        video_output_dir = output_dir / "videos"
        video_utils.write_video(
            video=video,
            dst=video_output_dir / f"{log_id}_{cam_name}.mp4",
            fps=RING_CAMERA_FPS,
        )


@click.command(
    help="Generate LiDAR + map visualizations from the Argoverse 2 Sensor Dataset."
)
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to local directory where renderings will be saved.",
    type=str,
)
@click.option(
    "-l",
    "--log-id",
    default="00a6ffc1-6ce9-3bc3-a060-6006e9893a1a",
    help="unique log identifier.",
    type=str,
)
@click.option(
    "-g",
    "--render-ground-pts-only",
    default=True,
    help="Boolean argument whether to only render LiDAR points located close to the ground surface.",
    type=bool,
)
@click.option(
    "-s",
    "--dump-single-frames",
    default=False,
    help="Whether to save to disk individual RGB frames of the rendering, in addition to generating the mp4 file"
    "(defaults to False). Note: can quickly generate 100s of MBs, for 200 KB frames.",
    type=bool,
)
def run_generate_egoview_overlaid_lidar(
    data_root: str,
    output_dir: str,
    log_id: str,
    render_ground_pts_only: bool,
    dump_single_frames: bool,
) -> None:
    """Click entry point for visualizing LiDAR returns rendered on top of sensor imagery."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_root_path = Path(data_root)
    output_dir_path = Path(output_dir)

    logger.info(
        "data_root: %s, output_dir: %s, log_id: %s, render_ground_pts_only: %s, dump_single_frames: %s",
        data_root_path,
        output_dir_path,
        log_id,
        render_ground_pts_only,
        dump_single_frames,
    )

    # Initialize the MoGe model
    device = torch.device("cuda")
    MoGe_model = MoGeModel.from_pretrained("/work/weights/moge-2-vitl-normal/model.pt").to(device)  

    generate_egoview_overlaid_lidar(
        data_root=data_root_path,
        output_dir=output_dir_path,
        log_id=log_id,
        MoGe_model=MoGe_model,
        render_ground_pts_only=render_ground_pts_only,
        dump_single_frames=dump_single_frames,
    )


if __name__ == "__main__":

    run_generate_egoview_overlaid_lidar()
    # python generate_egoview_overlaid_lidar.py --data-root '/work/datasets/av2/train/' --output-dir './output' --render-ground-pts-only False --dump-single-frames True