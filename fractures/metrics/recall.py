from typing import Any

import torch
from torchmetrics import Metric
from torchtyping import TensorType

from miccai import PointTensor


class FracRecall(Metric):

    full_state_update = False

    def __init__(self, dist_thresh: float=10.0):
        super().__init__()

        self.dist_thresh = dist_thresh
        
        self.add_state('pos', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def cluster_centroids(
        self,
        volume: TensorType['D', 'H', 'W', Any],
    ) -> PointTensor:
        voxels = volume.nonzero()
        pt = PointTensor(coordinates=voxels.float())
        centroids = pt.cluster(
            max_neighbor_dist=self.dist_thresh,
            min_points=100,
        )

        return centroids

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.int64],
    ) -> None:
        pred_centroids = self.cluster_centroids(pred)
        target_centroids = self.cluster_centroids(target)
        
        self.total += target_centroids.num_points

        if not pred_centroids:
            return

        _, sq_dists = pred_centroids.neighbors(target_centroids, k=1)

        self.pos += torch.sum(torch.sqrt(sq_dists) < self.dist_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / self.total
