from typing import Dict, List, Tuple, Union

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
        mandible_channels: int,
        channels_list: List[int],
        backbone: str,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=2 + mandible_channels,
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
            nn.Linear(64, 1 if num_classes <= 2 else num_classes),
        )
        self.seg_head = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
        )

        self.num_classes = max(2, num_classes)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        mandible: TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
        fractures: TensorType['B', 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
    ]:
        # concatenate intensities, mandible, and fracture features
        mandible = mandible.reshape(x.shape[:1] + (-1,) + x.shape[2:])
        x = torch.cat(
            (x, mandible, fractures.unsqueeze(dim=1)),
            dim=1,
        )

        # determine whether patch has displaced fracture
        xs = self.encoder(x)
        embedding = xs[0].mean(dim=(2, 3, 4))
        logits = self.class_head(embedding)
        logits = logits.squeeze(dim=1) if self.num_classes == 2 else logits

        # determine fracture segmentation of remaining patches
        x = self.decoder(xs)
        x = self.seg_head(x)
        x = x.squeeze(dim=1)

        return logits, x


class JawFracLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        focal_loss: bool,
        dice_loss: bool,
        ignore_index: int=-1,
        beta: float=0.1,
    ) -> None:
        super().__init__()
        
        if num_classes <= 2:
            self.class_criterion = nn.BCEWithLogitsLoss()
        else:
            self.class_criterion = nn.CrossEntropyLoss()

        self.seg_criterion = SegmentationLoss(
            focal_loss, dice_loss, ignore_index,
        )

        self.num_classes = max(2, num_classes)
        self.beta = beta

    def forward(
        self,
        masks1: TensorType['p', 'size', 'size', 'size', torch.float32],
        logits: TensorType['P', torch.float32],
        masks2: TensorType['p', 'size', 'size', 'size', torch.float32],
        target: Tuple[
            TensorType['P', torch.int64],
            TensorType['P', 'size', 'size', 'size', torch.float32],
        ],
    ) -> Tuple[
        TensorType[torch.float32],
        Dict[str, TensorType[torch.float32]],
    ]:
        y_classes, y_masks = target

        y_classes = y_classes.float() if self.num_classes == 2 else y_classes
        class_loss = self.class_criterion(logits, y_classes)

        if self.num_classes == 2:
            # only provide segmentation feedback on patches without displacements
            seg_loss1 = self.seg_criterion(masks1, y_masks)
            seg_loss2 = self.seg_criterion(masks2, y_masks)

            loss = self.beta * class_loss + seg_loss1 + seg_loss2
            log_dict = {
                'loss/': loss,
                'loss/class': class_loss,
                'loss/seg1': seg_loss1,
                'loss/seg2': seg_loss2,
            }
        else:
            loss = class_loss
            log_dict = {
                'loss/': loss,
                'loss/class': class_loss,
            }

        return loss, log_dict
