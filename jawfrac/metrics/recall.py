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

    def compute_ious(
        self,
        pred: List[TensorType['N', 3, torch.int64]],
        target: List[TensorType['N', 3, torch.int64]],
    ) -> TensorType['P', 'T', torch.float32]:
        ious = torch.empty(len(pred), len(target))
        for i in range(len(pred)):
            for j in range(len(target)):
                voxel_idxs = torch.cat((pred[i], target[j]))
                unique, counts = torch.unique(
                    voxel_idxs, return_counts=True, dim=0,
                )
                iou = (counts == 2).sum() / unique.shape[0]

                ious[i, j] = iou
        
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
