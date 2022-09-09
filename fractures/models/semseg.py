from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import ConfusionMatrix, F1Score
from torchtyping import TensorType

from fractures.metrics import FracRecall
import fractures.nn as nn
from miccai import PointTensor
from miccai.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)


class SemSegModule(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model = nn.FracNet(**model_cfg)
        self.criterion = nn.FocalLoss(alpha=0.25, gamma=2.0)
        self.f1 = F1Score(num_classes=2, average='macro')
        self.recall = FracRecall()

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

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

        loss = self.criterion(x, y)

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

        loss = self.criterion(x, y)
        self.f1((x >= 0).long().flatten(), y.flatten())

        self.log('loss/val', loss)
        self.log('f1/val', self.f1)

    def predict_volume(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
        batch_size: int=24,
    ) -> TensorType['D', 'H', 'W', torch.float32]:
        # out = torch.zeros_like(features[0])
        out = torch.full_like(features[0], float('inf'))

        patch_slices = []
        for patch_idxs in patch_idxs:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch_slices.append(slices)

        for i in range(0, patch_idxs.shape[0], batch_size):
            # extract batch of patches from features volume
            x = []
            for slices in patch_slices[i:i + batch_size]:
                slices = (slice(None),) + slices
                x.append(features[slices])

            # predict segmentation of patches
            x = torch.stack(x)
            x = self(x)

            # compute maximum of overlapping predictions
            for slices, x in zip(patch_slices[i:i + batch_size], x):
                # out[crop[1:]] += x
                out[slices] = torch.maximum(out[slices], x)

        return torch.sigmoid(out)

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.int64],
        ],
        batch_idx: int,
    ) -> None:
        features, patch_idxs, target = batch

        # predict binary segmentation
        x = self.predict_volume(features, patch_idxs)
        mask = x < 0.4

        # compute metrics
        self.f1(mask.long().flatten(), (target > 0).long().flatten())
        self.recall(mask, target)
        
        # log metrics
        self.log('f1/test', self.f1)
        self.log('recall/test', self.recall)

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
        y = x < 0.9

        # compute voxel indices into source volume
        voxels = y.nonzero().float()
        hom_voxels = torch.column_stack((voxels, torch.ones_like(voxels[:, 0])))

        orig_voxels = torch.einsum('ij,kj->ki', torch.linalg.inv(affine), hom_voxels)
        orig_voxels = orig_voxels[:, :3].round().long()

        # initialize empty volume and fill with binary segmentation
        out = torch.zeros(shape.tolist(), dtype=int)
        out[tuple(orig_voxels.T)] = 1

        # cluster foreground voxels with DBSCAN
        pt = PointTensor(coordinates=voxels)
        cluster_idxs = pt.cluster(
            max_neighbor_dist=10.0,
            min_points=100,
            return_index=True,
        )

        # determine clusters with at least 1200 voxels
        _, inverse, counts = torch.unique(
            cluster_idxs, return_inverse=True, return_counts=True,
        )        
        frac_mask = (cluster_idxs != -1) & (counts >= 1200)[inverse]

        # fill volume with large clusters
        out[tuple(orig_voxels[frac_mask].T)] = 2

        return x

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
