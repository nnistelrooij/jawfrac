from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
from scipy import ndimage
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import ConfusionMatrix, F1Score
from torchtyping import TensorType
from torch_scatter import scatter_mean

from jawfrac.metrics import FracPrecision, FracRecall
import jawfrac.nn as nn
from jawfrac.models.common import (
    batch_forward,
    fill_source_volume,
)
from jawfrac.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from jawfrac.visualization import (
    draw_confusion_matrix,
    draw_fracture_result,
    draw_positive_voxels,
)


def filter_connected_components(
    seg: TensorType['D', 'H', 'W', torch.float32],
    conf_thresh: float,
    min_component_size: int,
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
    count_mask = component_counts >= min_component_size

    # determine components with mean confidence at least 0.70
    component_probs = scatter_mean(
        src=probs.flatten(),
        index=component_idxs.flatten(),
    )
    print(probs.amax())
    prob_mask = component_probs >= conf_thresh

    # project masks back to volume
    component_mask = count_mask & prob_mask
    volume_mask = component_mask[component_idxs]

    return volume_mask


class JawFracModule(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        two_stage: Dict[str, Any],
        focal_loss: bool,
        dice_loss: bool,
        conf_threshold: float,
        min_component_size: int,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        if two_stage['use']:
            two_stage = {k: v for k, v in two_stage.items() if k != 'use'}
            self.model = nn.MandibleFracNet(**two_stage, **model_cfg)
        else:
            self.model = nn.FracNet(**model_cfg)

        self.criterion = nn.SegmentationLoss(focal_loss, dice_loss)

        self.confmat = ConfusionMatrix(num_classes=2)
        self.f1 = F1Score(num_classes=2, average='macro')
        self.precision_metric = FracPrecision()
        self.recall = FracRecall()

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.conf_thresh = conf_threshold
        self.min_component_size = min_component_size

    def forward(
        self,
        x: TensorType['P', 'C', 'size', 'size', 'size', torch.float32],       
    ) -> TensorType['P', 'size', 'size', 'size', torch.float32]:
        x = self.model(x)

        return x

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, y = batch

        x = self(x)

        loss = self.criterion(x, y.float())

        self.log('loss/train', loss)

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.int64],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        x = self(x)

        loss = self.criterion(x, y.float())
        x = torch.sigmoid(x) >= self.conf_thresh
        self.f1(
            x.long().flatten(),
            (y > 0).long().flatten(),
        )

        self.log('loss/val', loss)
        self.log('f1/val', self.f1)

    def predict_volume(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
        batch_size: int=12,
    ) -> TensorType['D', 'H', 'W', torch.float32]:
        # convert patch indices to patch slices
        patch_slices = []
        for patch_idxs in patch_idxs:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch_slices.append(slices)

        # do model processing in batches
        out = batch_forward(self.model, features, patch_slices)
        seg = torch.cat(out)

        # compute maximum of overlapping predictions
        out = torch.full_like(features[0], -float('inf'))
        for slices, seg in zip(patch_slices, seg):
            out[slices] = torch.maximum(out[slices], seg)

        # # compute mean of overlapping predictions
        # out = torch.zeros_like(features[0])
        # for slices, seg in zip(patch_slices, seg):
        #     out[slices] += seg

        return out

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.bool],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.bool],
        ],
        batch_idx: int,
    ) -> None:
        features, mandible, patch_idxs, target = batch

        # predict binary segmentation
        x = self.predict_volume(features, patch_idxs)
        mask = filter_connected_components(
            x, self.conf_thresh, self.min_component_size,
        )

        # compute metrics
        self.confmat(
            torch.any(mask)[None].long(),
            torch.any(target)[None].long(),
        )
        self.f1(mask.long().flatten(), target.long().flatten())
        self.precision_metric(mask, target)
        self.recall(mask, target)
        
        # log metrics
        self.log('f1/test', self.f1)
        self.log('precision/test', self.precision_metric)
        self.log('recall/test', self.recall)
        
        draw_fracture_result(mandible, mask, target)

    def test_epoch_end(self, _) -> None:
        draw_confusion_matrix(self.confmat)

    def predict_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType['P', torch.float32]:
        features, patch_idxs, affine, shape = batch

        # predict binary segmentation
        x = self.predict_volume(features, patch_idxs)

        mask = filter_connected_components(
            x ,self.conf_thresh, self.min_component_size,
        )

        draw_positive_voxels(mask)

        out = fill_source_volume(mask, affine, shape)
        
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
