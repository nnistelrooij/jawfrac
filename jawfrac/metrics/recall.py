from typing import Any, List

from scipy import ndimage
import torch
from torchmetrics import Metric
from torchtyping import TensorType


class FracRecall(Metric):

    full_state_update = False

    def __init__(
        self,
        iou_thresh: float=0.1,
    ):
        super().__init__()

        self.iou_thresh = iou_thresh
        
        self.add_state('pos', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def cluster_voxels(
        self,
        volume: TensorType['D', 'H', 'W', Any],
    ) -> List[TensorType['N', 3, torch.int64]]:
        voxels = torch.cartesian_prod(*[torch.arange(d) for d in volume.shape])
        voxels = voxels.reshape(volume.shape + (3,))

        cluster_idxs, _ = ndimage.label(
            input=volume.cpu(),
            structure=ndimage.generate_binary_structure(3, 1),
        )
        cluster_idxs = torch.from_numpy(cluster_idxs).to(volume.device)

        _, inverse, counts = torch.unique(
            cluster_idxs.flatten(), return_inverse=True, return_counts=True,
        )
        cluster_idxs[(counts < 1000)[inverse].reshape(cluster_idxs.shape)] = 0

        out = []
        for i in torch.unique(cluster_idxs.flatten())[1:]:
            out.append(voxels[cluster_idxs == i])
        
        return out

    def compute_iou(
        self,
        pred: TensorType['N', 3, torch.int64],
        target: TensorType['N', 3, torch.int64],
    ) -> float:
        start_idxs = torch.maximum(pred.amin(dim=0), target.amin(dim=0))
        stop_idxs = torch.minimum(pred.amax(dim=0), target.amax(dim=0))

        intersection = (stop_idxs - start_idxs + 1).clip(0, None)
        intersection = torch.prod(intersection)
        
        pred_volume = torch.prod(pred.amax(dim=0) - pred.amin(dim=0) + 1)
        target_volume = torch.prod(target.amax(dim=0) - target.amin(dim=0) + 1)

        return intersection / (pred_volume + target_volume - intersection)

    def compute_ious(
        self,
        pred: List[TensorType['N', 3, torch.int64]],
        target: List[TensorType['N', 3, torch.int64]],
    ) -> TensorType['P', 'T', torch.float32]:
        ious = torch.empty(len(pred), len(target))
        for i in range(len(pred)):
            for j in range(len(target)):
                ious[i, j] = self.compute_iou(pred[i], target[j])
        
        return ious

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.int64],
    ) -> None:
        if not torch.any(target):
            return

        target_voxels = self.cluster_voxels(target)
        self.total += len(target_voxels)

        if not torch.any(pred):
            return

        pred_voxels = self.cluster_voxels(pred)
        if not pred_voxels:
            return

        ious = self.compute_ious(pred_voxels, target_voxels)
        
        self.pos += torch.sum(ious.amax(dim=0) >= self.iou_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
