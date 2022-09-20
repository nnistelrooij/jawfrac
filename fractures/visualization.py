from typing import Any, List, Optional

import numpy as np
import open3d
from open3d.visualization.tensorboard_plugin import summary  # do not remove
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torchtyping import TensorType


palette = np.array([
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148], 
    [23, 190, 207], 
    [247, 182, 210], 
    [219, 219, 141], 
    [255, 127, 14], 
    [158, 218, 229], 
    [44, 160, 44], 
    [112, 128, 144], 
    [227, 119, 194], 
    [82, 84, 163],
    [50, 50, 50],
], dtype=np.uint8)


def draw_positive_voxels(
    volumes: List[TensorType['D', 'H', 'W', Any]],
    output_type: Optional[str]=None,
) -> Optional[List[open3d.geometry.PointCloud]]:
    points, colors = np.empty((2, 0, 3))
    for i, volume in enumerate(volumes, -1):
        pos_voxel_idxs = volume.nonzero()
        coordinates = pos_voxel_idxs + i / len(volumes)
        color = palette[i]

        points = np.concatenate((points, coordinates.cpu().numpy()))
        colors = np.concatenate(
            (colors, np.tile(color, (coordinates.shape[0], 1))),
        )

    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(points),
    )
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)

    if output_type == 'tensorboard':
        return to_dict_batch([pcd])

    open3d.visualization.draw_geometries([pcd], width=1600, height=900)
