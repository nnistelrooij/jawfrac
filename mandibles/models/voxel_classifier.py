from typing import Tuple

import torch
from torchtyping import TensorType

from miccai import PointTensor
from miccai.models import PointClassifier


class VoxelClassifier(PointClassifier):

    def predict_step(
        self,
        batch: Tuple[
            PointTensor,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.float64],
        ],
        batch_idx: int,
    ) -> TensorType['D', 'H', 'W', torch.bool]:
        x, affine, shape = batch

        # forward pass
        logits = self(x)

        # foreground homogeneous coordinates
        coords = logits.C[logits.F[:, 1] >= 2]
        coords = torch.column_stack((coords, torch.ones_like(coords[:, 0])))

        # indices of original voxels
        voxel_idxs = torch.einsum('kj,ij->ki', coords, torch.linalg.inv(affine))
        voxel_idxs = voxel_idxs[:, :3].round().long()
        voxel_idxs = tuple(voxel_idxs.T)

        # fill volume at locations predicted by model
        volume = torch.zeros(
            shape.tolist(), dtype=torch.bool, device=coords.device,
        )
        volume[voxel_idxs] = True

        return volume
