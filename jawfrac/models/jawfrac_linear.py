from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
from scipy import ndimage
from sklearn import neighbors
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import ConfusionMatrix, F1Score
from torchtyping import TensorType
from torch_scatter import scatter_mean

from jawfrac.metrics import FracPrecision, FracRecall
import jawfrac.nn as nn
from jawfrac.models.common import (
    aggregate_dense_predictions,
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
)


def filter_connected_components(
    mandible: TensorType['D', 'H', 'W', torch.bool],
    seg: TensorType['D', 'H', 'W', torch.float32],
    conf_thresh: float,
    min_component_size: int,
    max_dist: float=20.0,
) -> TensorType['D', 'H', 'W', torch.int64]:
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

    # determine components with mean confidence at least conf_thresh
    component_probs = scatter_mean(
        src=seg.flatten(),
        index=component_idxs.flatten(),
    )
    print(seg.amax())
    prob_mask = component_probs >= 0.5

    # determine components within max_dist voxels of mandible
    voxels = torch.cartesian_prod(*[torch.arange(d) for d in seg.shape]).to(seg)
    component_centroids = scatter_mean(
        src=voxels.reshape(-1, 3),
        index=component_idxs.flatten(),
        dim=0,
    )

    nbrs = neighbors.NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nbrs.fit(mandible.nonzero().cpu())
    component_dists, _ = nbrs.kneighbors(component_centroids.cpu())
    component_dists = torch.from_numpy(component_dists[:, 0]).to(seg)
    dist_mask = component_dists < max_dist

    # remove connected components and make their indices consecutive
    component_mask = count_mask & prob_mask & dist_mask
    component_idxs[~component_mask[component_idxs]] = 0
    component_idxs = (component_mask.cumsum(dim=0))[component_idxs]

    return component_idxs


class LinearJawFracModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        first_stage: Dict[str, Any],
        second_stage: Dict[str, Any],
        focal_loss: bool,
        dice_loss: bool,
        conf_threshold: float,
        min_component_size: int,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.mandible_net = nn.MandibleNet(
            num_classes=1, **first_stage, **model_cfg,
        )
        self.frac_net = nn.JawFracNet(
            num_classes=1,
            mandible_channels=self.mandible_net.out_channels,
            **second_stage,
            **model_cfg,
        )

        self.criterion = nn.SegmentationLoss(focal_loss, dice_loss)

        self.confmat = ConfusionMatrix(num_classes=2)
        self.f1 = F1Score(num_classes=2, average='macro')
        self.precision_metric = FracPrecision()
        self.recall_metric = FracRecall()

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.conf_thresh = conf_threshold
        self.min_component_size = min_component_size

    def forward(
        self,
        x: TensorType['P', 1, 'size', 'size', 'size', torch.float32],       
    ) -> Union[
        TensorType['B', 'D', 'H', 'W', torch.float32],
        Tuple[
            TensorType['B', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.float32],
        ],
    ]:
        coords, mandible = self.mandible_net(x)
        seg = self.frac_net(x, coords, mandible)

        return seg

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.float32],
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
            TensorType['P', 'size', 'size', 'size', torch.float32],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        x = self(x)

        loss = self.criterion(x, y)

        if isinstance(x, tuple):
            x = x[1]

        x = torch.sigmoid(x) >= self.conf_thresh
        self.f1(
            x[y != -1].long(),
            (y[y != -1] >= self.conf_thresh).long(),
        )

        self.log('loss/val', loss)
        self.log('f1/val', self.f1)

    def predict_volume(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
    ) -> TensorType['D', 'H', 'W', torch.float32]:
        # initialize generator that aggregates predictions
        mask_generator = aggregate_dense_predictions(
            patch_idxs=patch_idxs,
            out_shape=features.shape[1:],
        )

        # get initial empty prediction
        mask_pred = next(mask_generator)

        # run the model and aggregate its predictions
        batches = batch_forward(
            model=self,
            features=features,
            patch_idxs=patch_idxs,
            x_axis_flip=True,
            batch_size=32,
        )
        for masks in batches:
            if isinstance(masks, tuple):
                masks = masks[1]

            for mask in masks:
                mask_pred -= mask_pred
                mask_pred += mask
                next(mask_generator)

        # do any post-processing steps and compute probabilities
        return torch.sigmoid(next(mask_generator))

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.bool],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.float32],
        ],
        batch_idx: int,
    ) -> None:
        features, mandible, patch_idxs, labels = batch

        # predict binary segmentation
        x = self.predict_volume(features, patch_idxs)
        mask = filter_connected_components(
            mandible, x, self.conf_thresh, self.min_component_size,
        ) > 0

        # compute metrics
        target = labels >= self.conf_thresh
        self.confmat(
            torch.any(mask)[None].long(),
            torch.any(target)[None].long(),
        )
        self.f1(mask.long().flatten(), target.long().flatten())
        self.precision_metric(mask, target)
        self.recall_metric(mask, target)
        
        # log metrics
        self.log('f1/test', self.f1)
        self.log('precision/test', self.precision_metric)
        self.log('recall/test', self.recall_metric)
        
        # visualize results with Open3D
        draw_fracture_result(mandible, mask, labels >= self.conf_thresh)

    def test_epoch_end(self, _) -> None:
        draw_confusion_matrix(self.confmat)

    def predict_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.bool],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType['P', torch.float32]:
        features, mandible, patch_idxs, affine, shape = batch

        # predict binary segmentation
        x = self.predict_volume(features, patch_idxs)

        mask = filter_connected_components(
            mandible, x ,self.conf_thresh, self.min_component_size,
        ) > 0

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
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.00)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.001)

        return [opt], [sch]
