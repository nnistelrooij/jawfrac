import torch
from torchtyping import TensorType

from fractures.metrics.recall import FracRecall


class FracPrecision(FracRecall):

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.int64],
    ) -> None:
        if not torch.any(pred):
            return

        pred_centroids = self.cluster_centroids(pred)
        target_centroids = self.cluster_centroids(target)
        
        ious = self.compute_ious(pred_centroids, target_centroids)
    
        self.total += pred_centroids.num_points
        self.pos += torch.sum(ious.amax(dim=1) >= self.iou_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / self.total
