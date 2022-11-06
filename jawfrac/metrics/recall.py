from typing import List

from scipy import ndimage
import torch
from torchmetrics import Metric
from torchtyping import TensorType


class FracRecall(Metric):

    full_state_update = False

    def __init__(
        self,
        voxel_thresh: int=1000,
        dist_thresh: float=12.5,  # voxels
        iou_thresh: float=0.1,
    ):
        super().__init__()

        self.dist_thresh = dist_thresh
        self.voxel_thresh = voxel_thresh
        self.iou_thresh = iou_thresh
        
        self.add_state('pos', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def cluster_voxels(
        self,
        volume: TensorType['D', 'H', 'W', torch.bool],
    ) -> List[TensorType['N', 3, torch.int64]]:
        voxels = torch.cartesian_prod(*[torch.arange(d) for d in volume.shape])
        voxels = voxels.reshape(volume.shape + (3,)).to(volume.device)

        cluster_idxs, _ = ndimage.label(
            input=volume.cpu(),
            structure=ndimage.generate_binary_structure(3, 1),
        )
        cluster_idxs = torch.from_numpy(cluster_idxs).to(volume.device)

        _, inverse, counts = torch.unique(
            cluster_idxs, return_inverse=True, return_counts=True,
        )
        cluster_idxs[(counts < self.voxel_thresh)[inverse]] = 0

        out = []
        for i in torch.unique(cluster_idxs)[1:]:
            out.append(voxels[cluster_idxs == i])
        
        return out

    def compute_target_counts(
        self,
        preds: List[TensorType['N', 3, torch.int64]],
        targets: List[TensorType['M', 3, torch.int64]],
    ) -> TensorType['N', 'M', torch.float32]:
        target_counts = torch.zeros(len(preds), len(targets))
        for i, pred_voxels in enumerate(preds):
            centroid = pred_voxels.float().mean(dim=0)

            for j, target_voxels in enumerate(targets):
                dists = torch.sum((target_voxels - centroid) ** 2, dim=1)
                target_counts[i, j] = (dists <= self.dist_thresh ** 2).sum()

        return target_counts      

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.bool],
    ) -> None:
        if not torch.any(target):
            return

        target_voxels = self.cluster_voxels(target)
        self.total += len(target_voxels)

        if not torch.any(pred):
            return

        pred_voxels = self.cluster_voxels(pred)
        target_counts = self.compute_target_counts(pred_voxels, target_voxels)
        
        self.pos += torch.sum(target_counts.amax(dim=0) >= self.voxel_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
