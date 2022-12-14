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

        pred_voxels = self.cluster_voxels(pred, self.pred_thresh)
        self.total += len(pred_voxels)

        if not torch.any(target):
            print('FP')
            return

        target_voxels = self.cluster_voxels(target, self.target_thresh)

        if not target_voxels:
            print('FP')
            return

        target_counts = self.compute_target_counts(pred_voxels, target_voxels)
        
        self.pos += torch.sum(target_counts.amax(dim=1) >= self.target_thresh)
        
        if not torch.all(target_counts.amax(dim=1) >= self.target_thresh):
            print('FP')

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
