from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
from scipy import interpolate, ndimage
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import F1Score
from torchtyping import TensorType
from torch_scatter import scatter_mean

import jawfrac.nn as nn
from jawfrac.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from jawfrac.models.common import (
    aggregate_dense_predictions,
    aggregate_sparse_predictions,
    batch_forward,
    fill_source_volume,
)
from jawfrac.visualization import draw_positive_voxels


def filter_connected_components(
    coords: TensorType['D', 'H', 'W', 3, torch.float32],
    seg: TensorType['D', 'H', 'W', torch.float32],
    conf_thresh: float=0.5,
) -> TensorType['D', 'H', 'W', torch.bool]:
    # determine connected components in volume
    probs = torch.sigmoid(seg)
    labels = (probs >= conf_thresh).long()
    component_idxs, _ = ndimage.label(
        input=labels.cpu(),
        structure=ndimage.generate_binary_structure(3, 1),
    )
    component_idxs = torch.from_numpy(component_idxs).to(labels)

    # determine components comprising at least 20k voxels
    _, component_idxs, component_counts = torch.unique(
        component_idxs, return_inverse=True, return_counts=True,
    )
    count_mask = component_counts >= 20_000

    # determine components with mean confidence at least 0.70
    component_probs = scatter_mean(
        src=probs.flatten(),
        index=component_idxs.flatten(),
    )
    prob_mask = component_probs >= 0.6

    # determine components within two variance of centroid
    coords[..., 0] -= 1  # left-right symmetry
    component_coords = scatter_mean(
        src=coords.reshape(-1, 3),
        index=component_idxs.flatten(),
        dim=0,
    )
    component_dists = torch.sum(component_coords ** 2, dim=1)
    dist_mask = component_dists < 2

    # project masks back to volume
    component_mask = count_mask & prob_mask & dist_mask
    volume_mask = component_mask[component_idxs]

    return volume_mask


class MandibleSegModule(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        focal_loss: bool,
        dice_loss: bool,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model = nn.MandibleNet(**model_cfg)
        self.criterion = nn.MandibleLoss(focal_loss, dice_loss)
        self.f1 = F1Score(num_classes=2, average='macro')

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

    def forward(
        self,
        x: TensorType['P', 'C', 'size', 'size', 'size', torch.float32],       
    ) -> Tuple[
        TensorType['P', 3, torch.float32],
        TensorType['P', 'size', 'size', 'size', torch.float32],
    ]:
        coords, seg = self.model(x)

        return coords, seg

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 3, torch.float32],
                TensorType['P', 'size', 'size', 'size', torch.float32],
            ],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, y = batch

        coords, seg = self(x)

        loss, log_dict = self.criterion(coords, seg, y)

        self.log_dict({
            k.replace('/', '/train_' if k[-1] != '/' else '/train'): v
            for k, v in log_dict.items()
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 3, torch.float32],
                TensorType['P', 'size', 'size', 'size', torch.float32],
            ],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        coords, seg = self(x)

        _, log_dict = self.criterion(coords, seg, y)
        self.f1((seg[y[1] != -1] >= 0).long(), y[1][y[1] != -1].long())

        self.log_dict({
            **{
                k.replace('/', '/val_' if k[-1] != '/' else '/val'): v
                for k, v in log_dict.items()
            },
            'f1/val': self.f1,
        })    

    def predict_volumes(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
    ) -> Tuple[
        TensorType['D', 'H', 'W', 3, torch.float32],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        coords, seg = batch_forward(self.model, features, patch_idxs)

        coords = aggregate_sparse_predictions(
            coords, patch_idxs, features.shape[1:],
        )
        seg = aggregate_dense_predictions(
            seg, patch_idxs, features.shape[1:],
        )
        
        return coords, seg

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.float32],
        ],
        batch_idx: int,
    ) -> None:
        features, patch_idxs, labels = batch

        # predict binary segmentation
        coords, seg = self.predict_volumes(features, patch_idxs)

        # remove small or low-confidence connected components
        mask = filter_connected_components(coords, seg)

        # compute metrics
        self.f1(mask.long().flatten(), labels.long().flatten())
        
        # log metrics
        self.log('f1/test', self.f1)
        
        draw_positive_voxels(mask)

    def predict_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType['D', 'H', 'W', torch.bool]:
        features, patch_idxs, affine, shape = batch

        # predict binary segmentation
        coords, seg = self.predict_volumes(features, patch_idxs)

        # remove small or low-confidence connected components
        volume_mask = filter_connected_components(coords, seg)

        # fill volume with original shape given foreground mask
        out = fill_source_volume(volume_mask, affine, shape)

        return out

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.01)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.0001)

        return [opt], [sch]
