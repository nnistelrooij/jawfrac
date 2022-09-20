import torch
from torchtyping import TensorType

from jawfrac.metrics.recall import FracRecall


class FracPrecision(FracRecall):

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        target: TensorType['D', 'H', 'W', torch.int64],
    ) -> None:
        if not torch.any(pred):
            return

        pred_voxels = self.cluster_voxels(pred)
        target_voxels = self.cluster_voxels(target)
        
        ious = self.compute_ious(pred_voxels, target_voxels)
    
        self.total += len(pred_voxels)
        self.pos += torch.sum(ious.amax(dim=1) >= self.iou_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / self.total
