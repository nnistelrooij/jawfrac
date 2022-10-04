from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.loss import SegmentationLoss
from jawfrac.nn.modules.swin_unetr import SwinUNETRBackbone
from jawfrac.nn.modules.unet import UNet


class MandibleNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
        backbone: str,
        head_kernel_size: int,
        return_features: bool=False,
        checkpoint_path: str='',
    ) -> None:
        super().__init__()

        if backbone == 'conv':
            self.unet = UNet(
                in_channels=1,
                num_classes=num_classes,
                channels_list=channels_list,
                num_awms=num_awms,
            )
            num_features = channels_list[1]
        elif backbone == 'swin':
            self.unet = SwinUNETRBackbone(
                img_size=64,
                in_channels=1,
                out_channels=1,
                feature_size=36,
            )
            num_features = 36
        else:
            raise ValueError(f'Backbone not recognized: {backbone}.')

        num_latents = channels_list[-1] if backbone == 'conv' else 16 * 36
        self.loc_head = nn.Sequential(
            nn.Linear(num_latents, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # self.loc_head = nn.Linear(64, 3)
        self.seg_head = nn.Conv3d(
            num_features,
            num_classes,
            head_kernel_size,
            padding=head_kernel_size // 2,
        )

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            self.load_state_dict(state_dict)
            self.requires_grad_(False)

        self.return_features = return_features
        self.out_channels = 1 + return_features * num_features

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', 3, torch.float32],
        TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
    ]:
        encoding, features = self.unet(x)

        embedding = encoding.mean(axis=(2, 3, 4))
        coords = self.loc_head(embedding)

        seg = self.seg_head(features)

        if self.return_features:
            out = torch.cat((seg, features), dim=1)
        else:
            out = seg.squeeze(dim=1)

        return coords, out


class MandibleLoss(nn.Module):

    def __init__(
        self,
        focal_loss: bool,
        dice_loss: bool,
    ) -> None:
        super().__init__()

        self.seg_criterion = SegmentationLoss(focal_loss, dice_loss)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        coords: TensorType['B', 3, torch.float32],
        seg: TensorType['B', 'D', 'H', 'W', torch.float32],
        y: Tuple[
            TensorType['B', 3, torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.bool],
        ],
    ) -> Tuple[
        TensorType[torch.float32],
        Dict[str, TensorType[torch.float32]],
    ]:
        y_coords, y_seg = y

        coords_loss = self.smooth_l1(coords, y_coords)
        seg_loss = self.seg_criterion(seg, y_seg)

        loss = coords_loss + seg_loss
        log_dict = {
            'loss/': loss,
            'loss/coords': coords_loss,
            'loss/seg': seg_loss,
        }

        return loss, log_dict
