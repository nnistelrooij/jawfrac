from scipy import ndimage
import torch
from torchmetrics import Metric
from torchtyping import TensorType


class FracRecall(Metric):

    full_state_update = False

    def __init__(
        self,
        dist_thresh: float=25.0,  # voxels
    ):
        super().__init__()

        self.dist_thresh = dist_thresh
        
        self.add_state('pos', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def compute_distances(
        self,
        mask: TensorType['D', 'H', 'W', torch.bool],
        targets: TensorType['F_target', 3, torch.float32],        
    ) -> TensorType['F_pred', 'F_target', torch.float32]:
        labels, _ = ndimage.label(
            input=mask.cpu().numpy(),
            structure=ndimage.generate_binary_structure(3, 1),
        )
        labels = torch.from_numpy(labels).to(mask.device)

        preds = torch.empty(0, 3).to(targets)
        for label in range(1, labels.max() + 1):
            voxels = torch.nonzero(labels == label)
            centroid = voxels.float().mean(dim=0, keepdim=True)
            preds = torch.cat((preds, centroid))

        return torch.cdist(preds, targets)        

    def update(
        self,
        pred: TensorType['D', 'H', 'W', torch.bool],
        targets: TensorType['F', 3, torch.float32],
    ) -> None:
        if targets.numel() == 0:
            return

        self.total += targets.shape[0]

        if not torch.any(pred):
            return

        dists = self.compute_distances(pred, targets)
        
        self.pos += torch.sum(dists.amin(dim=0) < self.dist_thresh)

    def compute(self) -> TensorType[torch.float32]:
        return self.pos / (self.total + 1e-6)
