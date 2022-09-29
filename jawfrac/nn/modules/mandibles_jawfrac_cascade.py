from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.loss import SegmentationLoss
from jawfrac.nn.modules.mandibles import MandibleNet
from jawfrac.nn.modules.swin_unetr import SwinUNETRBackbone
from jawfrac.nn.modules.unet import Decoder, Encoder


class MandibleFracCascadeNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
        checkpoint_path: str,
        backbone: str,
        coords: Optional[str],
    ) -> None:
        super().__init__()

        self.mandible_net = MandibleNet(
            num_awms=num_awms,
            num_classes=num_classes,
            channels_list=channels_list,
            checkpoint_path=checkpoint_path,
            backbone='conv',
        )
        self.mandible_net.requires_grad_(False)

        if backbone == 'conv':
            self.encoder1 = Encoder(
                in_channels=2,
                channels_list=channels_list,
            )
            self.encoder2 = Encoder(
                in_channels=3,
                channels_list=channels_list,
            )
            self.decoder1 = Decoder(
                num_classes=1,
                channels_list=[128, 64, 32, 16],
            )
            self.decoder2 = Decoder(
                num_classes=1,
                channels_list=[128, 64, 32, 16],
            )
        elif backbone == 'swin':
            self.unet1 = SwinUNETRBackbone(
                img_size=64,
                in_channels=2,
                out_channels=1,
            )
            self.unet2 = SwinUNETRBackbone(
                img_size=64,
                in_channels=3,
                out_channels=1,
            )
        else:
            raise ValueError(f'Backbone not recognized: {backbone}.')

        self.seg_head1 = nn.Conv3d(
            in_channels=32 - 8 * (backbone == 'swin'),
            out_channels=1,
            kernel_size=1,
        )
        self.seg_head2 = nn.Conv3d(
            in_channels=32 - 8 * (backbone == 'swin'),
            out_channels=1,
            kernel_size=1,
        )
        self.class_head = nn.Sequential(
            nn.Linear(128 + 448 * (backbone == 'swin'), 64),
            nn.BatchNorm1d(64, momentum=0.1), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

        self.backbone = backbone

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
        TensorType['B', torch.float32],
    ]:
        _, mandible = self.mandible_net(x)

        x = torch.cat((x, mandible.unsqueeze(dim=1)), dim=1)

        if self.backbone == 'conv':
            xs = self.encoder1(x)
            z = self.decoder1(xs)
        elif self.backbone == 'swin':
            _, z = self.unet1(x)

        masks1 = self.seg_head1(z).squeeze(dim=1)

        x = torch.cat((x, masks1.unsqueeze(dim=1)), dim=1)
        if self.backbone == 'conv':
            xs = self.encoder2(x)
            x = self.decoder2(xs)
        elif self.backbone == 'swin':
            xs, x = self.unet2(x)

        masks2 = self.seg_head2(x).squeeze(dim=1)

        x = xs[0].mean(dim=(2, 3, 4))
        logits = self.class_head(x).squeeze(dim=1)

        return masks1, masks2, logits


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
        masks1: TensorType['B', 'D', 'H', 'W', torch.float32],
        masks2: TensorType['B', 'D', 'H', 'W', torch.float32],
        logits: TensorType['B', torch.float32],
        target: Tuple[
            TensorType['B', 'D', 'H', 'W', torch.bool],
            TensorType['B', torch.bool],
        ],
    ) -> Tuple[
        TensorType[torch.float32],
        Dict[str, TensorType[torch.float32]],
    ]:
        y_masks, y_classes = target

        seg_loss1 = self.seg_criterion(masks1, y_masks)
        seg_loss2 = self.seg_criterion(masks2, y_masks)
        bce_loss = self.bce(logits, y_classes.float())

        loss = seg_loss1 + bce_loss
        log_dict = {
            'loss/': loss,
            'loss/seg1': seg_loss1,
            'loss/seg2': seg_loss2,
            'loss/bce': bce_loss,
        }

        return loss, log_dict
