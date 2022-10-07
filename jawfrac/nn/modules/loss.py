from typing import Union, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType


class SegmentationLoss(nn.Module):
    "Implements binary segmentation loss function."

    def __init__(
        self,
        focal_loss: bool,
        dice_loss: bool,
        ignore_index: int=-1,
    ) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: Union[
            TensorType['B', '...', torch.float32],
            Tuple[TensorType['B', '...', torch.float32], ...],
        ],
        target: Union[
            TensorType['B', '...', torch.bool],
            TensorType['B', '...', torch.float32],
        ],
    ) -> TensorType[torch.float32]:
        if isinstance(pred, torch.Tensor):
            pred = (pred,)

        target = target.float()
        ignore_mask = target == self.ignore_index
        target = target[~ignore_mask]

        loss = torch.tensor(0).to(pred[0])
        for pred in pred:
            # remove voxels to be ignored
            pred = pred[~ignore_mask]

            # return zero loss if all voxels have been removed
            if pred.numel() == 0:
                continue

            loss_i = self.bce(pred, target)

            if self.focal_loss:
                probs = torch.sigmoid(pred)
                alphat = 0.25 * target + 0.75 * (1 - target)
                pt = probs * target + (1 - probs) * (1 - target)
                loss_i *= alphat * (1 - pt) ** 2

            loss_i = torch.mean(loss_i)

            if self.dice_loss:
                dim = tuple(range(1, len(pred.shape)))
                numerator = 2 * torch.sum(pred * target, dim=dim)
                denominator = torch.sum(pred ** 2 + target ** 2, dim=dim)
                dice_loss = 1 - torch.mean((numerator + 1e-6) / (denominator + 1e-6))
                loss_i = dice_loss + 0.5 * loss_i

            loss += loss_i
        
        return loss
