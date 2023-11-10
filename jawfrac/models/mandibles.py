from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pytorch_lightning as pl
from scipy import ndimage
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import F1Score, Dice
from torchmetrics.classification import BinaryJaccardIndex
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
    half_stride_patch_idxs,
)
from jawfrac.visualization import draw_positive_voxels


def filter_connected_components(
    coords: TensorType['D', 'H', 'W', 3, torch.float32],
    seg: TensorType['D', 'H', 'W', torch.float32],
    conf_thresh: float,
    min_component_size: int,
    max_dist: float,
) -> TensorType['D', 'H', 'W', torch.bool]:
    # determine connected components in volume
    labels = (seg >= conf_thresh).long()
    component_idxs, _ = ndimage.label(
        input=labels.cpu(),
        structure=ndimage.generate_binary_structure(3, 1),
    )
    component_idxs = torch.from_numpy(component_idxs).to(labels)

    # determine components comprising at least 20k voxels
    _, component_idxs, component_counts = torch.unique(
        component_idxs, return_inverse=True, return_counts=True,
    )
    count_mask = component_counts >= min_component_size

    # determine components with mean confidence at least 0.70
    component_probs = scatter_mean(
        src=seg.flatten(),
        index=component_idxs.flatten(),
    )
    prob_mask = component_probs >= 0.5

    # determine components within two variance of centroid
    coords[..., 0] -= 1  # left-right symmetry
    component_centroids = scatter_mean(
        src=coords.reshape(-1, 3),
        index=component_idxs.flatten(),
        dim=0,
    )
    component_dists = torch.sum(component_centroids ** 2, dim=1)
    dist_mask = component_dists < max_dist

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
        x_axis_flip: bool,
        conf_threshold: float,
        min_component_size: int,
        max_dist: float,
        batch_size: int=0,
        interpolation: Literal['slow', 'fast', 'none']='fast',
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model = nn.MandibleNet(**model_cfg)
        self.criterion = nn.MandibleLoss(focal_loss, dice_loss)
        self.f1 = F1Score(task='binary', num_classes=2, average='macro')
        self.iou = BinaryJaccardIndex()
        self.dice = Dice(multiclass=False)

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.x_axis_flip = x_axis_flip
        self.conf_thresh = conf_threshold
        self.min_component_size = min_component_size
        self.max_dist = max_dist
        self.batch_size = batch_size
        self.interpolation = interpolation

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
        patch_idxs: TensorType['d', 'h', 'w', 3, 2, torch.int64],
    ) -> Tuple[
        TensorType['D', 'H', 'W', 3, torch.float32],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        patch_idxs = patch_idxs[half_stride_patch_idxs(patch_idxs)]
        subgrid_shape = patch_idxs.shape[:3]

        # initialize generators that aggregate predictions
        coords_generator = aggregate_sparse_predictions(
            pred_shape=(3,),
            patch_idxs=patch_idxs.reshape(-1, 3, 2),
            out_shape=features.shape[1:],
        )

        # get initial empty predictions
        coords_pred = next(coords_generator)
        seg = torch.empty(subgrid_shape).to(features)
        counter = 0

        # run the model and aggregate its predictions
        batches = batch_forward(
            model=self,
            features=features,
            patch_idxs=patch_idxs.reshape(-1, 3, 2),
            x_axis_flip=False,
            batch_size=self.batch_size,
        )
        for coords, masks in batches:
            for coords in coords:
                coords_pred -= coords_pred
                coords_pred += coords
                next(coords_generator)

            for mask in masks:
                seg[np.unravel_index(counter, subgrid_shape)] = mask.amax()
                counter += 1


        # do any post-processing steps and compute probabilities
        coords = next(coords_generator)
        seg = torch.sigmoid(seg)

        return coords, seg

    def finetune_volume(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['d', 'h', 'w', 3, 2, torch.int64],
        seg: TensorType['d', 'h', 'w', torch.float32],
        conf_thresh: float=0.1,
    ) -> TensorType['D', 'H', 'W', torch.float32]:
        # determine which patches to use for fine-tuning
        pos_mask = torch.zeros(patch_idxs.shape[:3]).to(seg.device, torch.bool)
        pos_mask[half_stride_patch_idxs(patch_idxs)] = seg >= conf_thresh
        pos_mask = ndimage.binary_dilation(
            input=pos_mask.cpu().numpy(),
            structure=ndimage.generate_binary_structure(3, 3),
            iterations=1,
        )
        pos_mask = torch.from_numpy(pos_mask).to(patch_idxs.device)

        mask_generator = aggregate_dense_predictions(
            patch_idxs=patch_idxs[pos_mask],
            out_shape=features.shape[1:],
        )

        # get initial empty predictions
        mask_pred = next(mask_generator)

        # run the model and aggregate its predictions
        batches = batch_forward(
            model=self,
            features=features,
            patch_idxs=patch_idxs[pos_mask],
            x_axis_flip=self.x_axis_flip,
            batch_size=self.batch_size,
        )
        for _, masks in batches:
            for mask in masks:
                mask_pred -= mask_pred
                mask_pred += mask
                next(mask_generator)

        # do any post-processing steps and compute probabilities
        seg = torch.sigmoid(next(mask_generator))

        return seg

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
        seg = self.finetune_volume(features, patch_idxs, seg)

        # remove small or low-confidence connected components
        mask = filter_connected_components(
            coords, seg, self.conf_thresh, self.min_component_size, self.max_dist,
        )

        # compute metrics
        self.iou(mask.long().flatten(), labels.long().flatten())
        self.dice(mask.long().flatten(), labels.long().flatten())
        
        # log metrics
        self.log('iou/test', self.iou)
        self.log('dice/test', self.dice)
        
        # draw_positive_voxels(mask)

    def predict_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> Tuple[
        TensorType['D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.bool],
        TensorType['D', 'H', 'W', torch.bool],
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.int64],
    ]:
        files = self.trainer.datamodule.predict_dataset.files[batch_idx]
        print(files[0].parent.stem)
        
        features, patch_idxs, affine, shape = batch

        # predict binary segmentation
        coords, seg = self.predict_volumes(features, patch_idxs)
        seg = self.finetune_volume(features, patch_idxs, seg)

        if self.interpolation == 'fast':
            # remove small or low-confidence connected components
            # volume_mask = filter_connected_components(
            #     coords, seg, self.conf_thresh, self.min_component_size, self.max_dist,
            # )
            volume_mask = (seg >= self.conf_thresh)

            # fill volume with original shape given foreground mask
            volume_mask = fill_source_volume(volume_mask, affine, shape, method='fast')
        else:
            if self.interpolation == 'slow':
                # fill volume with original shape given foreground mask
                volume_probs = fill_source_volume(seg, affine, shape, method='slow')
                coords = torch.zeros(volume_probs.shape + (3,)).to(coords)
            else:
                volume_probs = seg

            # remove small or low-confidence connected components
            volume_mask = filter_connected_components(
                coords,
                volume_probs,
                self.conf_thresh, 
                self.min_component_size,
                self.max_dist,
            )

        return features[0].cpu().numpy(), volume_mask

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
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.001)

        return [opt], [sch]
