from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.unet import UNet


class MandibleNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        self.unet = UNet(
            in_channels=1,
            num_classes=num_classes,
            channels_list=channels_list,
            num_awms=num_awms,
        )

        self.loc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1), 
            nn.ReLU(), 
            nn.Linear(64, 3)
        )
        
        # self.loc_head = nn.Linear(64, 3)
        self.seg_head = nn.Conv3d(32, num_classes, 3, padding=1)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', 3, torch.float32],
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ]:
        x = self.gapm(x)
        encoding, x = self.unet(x)

        embedding = encoding.mean(axis=(2, 3, 4))
        coords = self.loc_head(embedding)
        seg = self.seg_head(x)

        return coords, seg.squeeze(dim=1)


class MandibleLoss(nn.Module):

    def __init__(
        self,
        alpha: float,
        gamma: float,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        coords: TensorType['B', 3, torch.float32],
        seg: TensorType['B', 'D', 'H', 'W', torch.float32],
        y: Tuple[
            TensorType['B', 3, torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.int64],
        ],
    ) -> Tuple[
        TensorType[torch.float32],
        Dict[str, TensorType[torch.float32]],
    ]:
        y_coords, y_seg = y

        coords_loss = self.smooth_l1(coords, y_coords)

        seg_loss = self.bce(seg, y_seg.float())

        probs = torch.sigmoid(seg)
        pt = y_seg * probs + (1 - y_seg) * (1 - probs)
        alphat = y_seg * self.alpha + (1 - y_seg) * (1 - self.alpha)
        seg_loss *= alphat * (1 - pt) ** self.gamma
        seg_loss = seg_loss.mean()

        loss = coords_loss + seg_loss
        log_dict = {
            'loss/': loss,
            'loss/coords': coords_loss,
            'loss/seg': seg_loss,
        }

        return loss, log_dict
