import torch
import torch.nn as nn
from torchtyping import TensorType


class SegmentationLoss(nn.Module):
    "Implements binary segmentation loss function."

    def __init__(
        self,
        focal_loss: bool,
        dice_loss: bool,
    ) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss

    def forward(
        self,
        pred: TensorType['B', '...', torch.float32],
        target: TensorType['B', '...', torch.float32],
    ) -> TensorType[torch.float32]:
        if pred.numel() == 0:
            return 0

        target = target.float()
        loss = self.bce(pred, target)

        if self.focal_loss:
            probs = torch.sigmoid(pred)
            alphat = 0.25 * target + 0.75 * (1 - target)
            pt = probs * target + (1 - probs) * (1 - target)
            loss *= alphat * (1 - pt) ** 2

        loss = torch.mean(loss)

        if self.dice_loss:
            dim = tuple(range(1, len(pred.shape)))
            numerator = 2 * torch.sum(pred * target, dim=dim)
            denominator = torch.sum(pred ** 2 + target ** 2, dim=dim)
            dice_loss = 1 - torch.mean((numerator + 1e-6) / (denominator + 1e-6))
            loss = dice_loss + 0.5 * loss
        
        return loss
