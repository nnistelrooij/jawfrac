from typing import Any

from sklearn.cluster import DBSCAN
import torch
from torchmetrics import Metric
from torchtyping import TensorType
from torch_scatter import scatter_mean

from miccai import PointTensor


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

    def cluster_centroids(
        self,
        volume: TensorType['D', 'H', 'W', Any],
    ) -> PointTensor:
        voxels = volume.nonzero()
        dbscan = DBSCAN(eps=self.max_neighbor_dist, min_samples=100, n_jobs=-1)
        dbscan.fit(voxels.cpu())

        cluster_idxs = torch.from_numpy(dbscan.labels_).to(voxels)     
        centroids = scatter_mean(
            src=voxels[cluster_idxs >= 0].float(),
            index=cluster_idxs[cluster_idxs >= 0],
            dim=0,
        )

        pt = PointTensor(centroids)
        for i in range(cluster_idxs.max() + 1):
            pt.cache[f'voxels_{i}'] = voxels[cluster_idxs == i]

        return pt

    def compute_ious(
        self,
        pred: PointTensor,
        target: PointTensor,
    ) -> TensorType['P', 'T', torch.float32]:
        ious = torch.empty(pred.num_points, target.num_points)
        for i in range(pred.num_points):
            for j in range(target.num_points):
                pred_voxels = pred.cache[f'voxels_{i}']
                target_voxels = target.cache[f'voxels_{j}']
                voxels = torch.cat((pred_voxels, target_voxels))

                unique, counts = torch.unique(voxels, return_counts=True, dim=0)
                iou = (counts == 2).sum() / unique.shape[0]

                ious[i, j] = iou
        
        return ious

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.int64],
    ) -> None:
        target_centroids = self.cluster_centroids(target)
        self.total += target_centroids.num_points

        if not torch.any(pred):
            return

        pred_centroids = self.cluster_centroids(pred)
        if not pred_centroids:
            return

        ious = self.compute_ious(pred_centroids, target_centroids)        
        
        self.pos += torch.sum(ious.amax(dim=0) >= self.iou_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / self.total
