from typing import Any, List

from sklearn.cluster import DBSCAN
import torch
from torchmetrics import Metric
from torchtyping import TensorType


class FracRecall(Metric):

    full_state_update = False

    def __init__(
        self,
        iou_thresh: float=0.2,
        max_neighbor_dist: float=10.0,
    ):
        super().__init__()

        self.iou_thresh = iou_thresh
        self.max_neighbor_dist = max_neighbor_dist
        
        self.add_state('pos', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def cluster_voxels(
        self,
        volume: TensorType['D', 'H', 'W', Any],
    ) -> List[TensorType['N', 3, torch.int64]]:
        voxel_idxs = volume.nonzero()
        dbscan = DBSCAN(eps=self.max_neighbor_dist, min_samples=100, n_jobs=-1)
        dbscan.fit(voxel_idxs.cpu())

        cluster_idxs = torch.from_numpy(dbscan.labels_).to(voxel_idxs)
        out = []
        for i in range(cluster_idxs.max() + 1):
            out.append(voxel_idxs[cluster_idxs == i])
        
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
