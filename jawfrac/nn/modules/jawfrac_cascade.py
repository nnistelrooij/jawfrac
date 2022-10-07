from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.gapm import GrayscaleAdaptivePerceptionModule
from jawfrac.nn.modules.loss import SegmentationLoss
from jawfrac.nn.modules.swin_unetr import SwinUNETRBackbone
from jawfrac.nn.modules.unet import Decoder, Encoder


class JawFracCascadeNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        mandible_channels: int,
        fracture_channels: int,
        channels_list: List[int],
        coords: str,
        backbone: str,
    ) -> None:

        super().__init__()

        assert backbone == 'conv' or 'sparse' not in coords, (
            'Cannot combine swin backbone with sparse coordinates.'
        )

        self.gapm = GrayscaleAdaptivePerceptionModule(num_awms)
        
        in_channels = (
            1 +
            num_awms +
            mandible_channels +
            fracture_channels +
            3 * ('dense' in coords)
        )

        if backbone == 'conv':
            self.encoder = Encoder(
                in_channels=in_channels,
                channels_list=channels_list,
            )
        elif backbone == 'swin':
            self.unet = SwinUNETRBackbone(
                img_size=64,
                in_channels=in_channels,
                out_channels=1,
            )
        else:
            raise ValueError(f'Backbone not recognized: {backbone}.')

        if 'dynamic' in coords and 'sparse' in coords:
            self.init_sparse_coords()

        self.head = nn.Sequential(
            nn.Linear(channels_list[-1] if backbone == 'conv' else 24, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.LeakyReLU(),
            nn.Linear(64, 1 if num_classes <= 2 else num_classes),
        )

        self.backbone = backbone
        self.coords = coords
        self.num_classes = max(num_classes, 2)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        coords: TensorType['B', 3, torch.float32],
        mandible: TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
        fractures: TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', torch.float32],
        TensorType['B', 'D', 'H', 'W', torch.float32],
    ]:
        x = self.gapm(x)

        # concatenate intensities, mandible, and fracture features
        mandible = mandible.reshape(x.shape[:1] + (-1,) + x.shape[2:])
        fractures = fractures.reshape(x.shape[:1] + (-1,) + x.shape[2:])
        x = torch.cat((x, mandible, fractures), dim=1)

        # determine whether patch has displaced fracture
        xs = self.encoder(x)
        embedding = xs[0].mean(dim=(2, 3, 4))
        logits = self.head(embedding)
        logits = logits.squeeze(dim=1) if self.num_classes == 2 else logits
        
        return logits


class JawFracLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        focal_loss: bool,
        dice_loss: bool,
        ignore_index: int=-1,
        beta: float=0.1,
        only_classification: bool=False,
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
        self.only_classification = only_classification

    def forward(
        self,
        masks: TensorType['p', 'size', 'size', 'size', torch.float32],
        logits: TensorType['P', torch.float32],
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

        if self.only_classification:
            log_dict = {
                'loss/': class_loss,
                'loss/class': class_loss,
            }
        else:
            # only provide segmentation feedback on patches without displacements
            seg_loss = self.seg_criterion(masks, y_masks)

            loss = self.beta * class_loss + seg_loss
            log_dict = {
                'loss/': loss,
                'loss/class': self.beta * class_loss,
                'loss/seg1': seg_loss,
            }

        return loss, log_dict
