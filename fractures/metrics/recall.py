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
        dist_thresh: float=25.0,
        max_neighbor_dist: float=10.0,
    ):
        super().__init__()

        self.dist_thresh = dist_thresh
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

        return PointTensor(centroids)

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

        _, sq_dists = pred_centroids.neighbors(target_centroids, k=1)

        self.pos += torch.sum(torch.sqrt(sq_dists) < self.dist_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / self.total
