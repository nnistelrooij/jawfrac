from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import open3d
from scipy import ndimage
import torch
from torchtyping import TensorType

from miccai import PointTensor
from miccai.models import PointClassifier


class VoxelClassifier(PointClassifier):

    def __init__(
        self,
        prob_threshold: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.prob_thresh = prob_threshold

    @staticmethod
    def fill_volume(
        coords: TensorType['N', 3, torch.float32],
        affine: TensorType[4, 4, torch.float32],
        shape: TensorType[3, torch.int64],
    ) -> TensorType['D', 'H', 'W', torch.bool]:
        # homogeneous coordinates
        coords = torch.column_stack(
            (coords, torch.ones_like(coords[:, 0])),
        )

        # apply inverse affine transformation to compute voxel indices
        voxels = torch.einsum('kj,ij->ki', coords, torch.linalg.inv(affine))
        voxels = voxels[:, :3].round().long()
        voxels = tuple(voxels.T)

        # fill volume with determined voxels
        volume = torch.zeros(
            shape.tolist(), dtype=torch.bool, device=coords.device,
        )
        volume[voxels] = True

        return volume

    def predict_step(
        self,
        batch: Tuple[
            PointTensor,
            TensorType['N', 3, torch.float32],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> NDArray[np.bool_]:
        x, fg_coords, affine, shape = batch

        # forward pass
        logits = self(x)

        # fill volume at locations predicted by model
        probs = torch.nn.functional.softmax(logits.F, dim=-1)
        x_coords = logits.C[probs[:, 1] >= self.prob_thresh]
        pred_volume = self.fill_volume(x_coords, affine, shape)

        # fill volume at foreground locations
        fg_volume = self.fill_volume(fg_coords, affine, shape)

        # dilate sparse predictions
        pred_volume = ndimage.binary_dilation(
            input=pred_volume.cpu().numpy(),
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=5,
            mask=fg_volume.cpu().numpy(),
        )

        return pred_volume
