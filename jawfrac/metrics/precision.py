import torch
from torchtyping import TensorType

from jawfrac.metrics.recall import FracRecall


class FracPrecision(FracRecall):

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.bool],
    ) -> None:    
        if not torch.any(pred):
            return

        pred_voxels = self.cluster_voxels(pred)
        self.total += len(pred_voxels)

        if not torch.any(target):
            return

        target_voxels = self.cluster_voxels(target)
        target_counts = self.compute_target_counts(pred_voxels, target_voxels)
        
        self.pos += torch.sum(target_counts.amax(dim=1) >= self.voxel_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
