from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.loss import SegmentationLoss
from jawfrac.nn.modules.unet import Decoder, Encoder


class JawFracCascadeNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
        backbone: str,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=3,
            channels_list=channels_list,
        )
        self.decoder = Decoder(
            num_classes=1,
            channels_list=[128, 64, 32, 16],
        )

        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.seg_head = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
        )

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        mandible: TensorType['B', 'D', 'H', 'W', torch.float32],
        fractures: TensorType['B', 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
    ]:
        # concatenate intensities and mandible and fracture logits
        x = torch.cat(
            (x, mandible.unsqueeze(dim=1), fractures.unsqueeze(dim=1)),
            dim=1,
        )

        # determine whether patch has displaced fracture
        xs = self.encoder(x)
        embedding = xs[0].mean(dim=(2, 3, 4))
        logits = self.class_head(embedding).squeeze(dim=1)

        # determine fracture segmentation of remaining patches
        x = self.decoder(xs)
        x = self.seg_head(x)

        return logits, x.squeeze(dim=1)


class JawFracLoss(nn.Module):

    def __init__(
        self,
        focal_loss: bool,
        dice_loss: bool,
    ) -> None:
        super().__init__()
        
        self.seg_criterion = SegmentationLoss(focal_loss, dice_loss)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        masks1: TensorType['p', 'size', 'size', 'size', torch.float32],
        logits: TensorType['P', torch.float32],
        masks2: TensorType['p', 'size', 'size', 'size', torch.float32],
        target: Tuple[
            TensorType['P', torch.bool],
            TensorType['P', 'size', 'size', 'size', torch.float32],
        ],
    ) -> Tuple[
        TensorType[torch.float32],
        Dict[str, TensorType[torch.float32]],
    ]:
        y_classes, y_masks = target

        bce_loss = self.bce(logits, y_classes.float())

        # do not provide segmentation feedback for displaced fractures
        masks1 = masks1[logits < 0]
        masks2 = masks2[logits < 0]
        y_masks = y_masks[logits < 0]
        seg_loss1 = self.seg_criterion(masks1, y_masks)
        seg_loss2 = self.seg_criterion(masks2, y_masks)

        loss = bce_loss + seg_loss1 + seg_loss2
        log_dict = {
            'loss/': loss,
            'loss/bce': bce_loss,
            'loss/seg1': seg_loss1,
            'loss/seg2': seg_loss2,
        }

        return loss, log_dict
