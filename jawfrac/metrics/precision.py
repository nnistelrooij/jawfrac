import torch
from torchtyping import TensorType

from jawfrac.metrics.recall import FracRecall


class FracPrecision(FracRecall):

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        targets: TensorType['F', 3, torch.float32],
    ) -> None:
        if not torch.any(pred):
            return

        dists = self.compute_distances(pred, targets)

        self.total += dists.shape[0]

        if dists.shape[1] == 0:
            return
    
        self.pos += torch.sum(dists.amin(dim=1) < self.dist_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
