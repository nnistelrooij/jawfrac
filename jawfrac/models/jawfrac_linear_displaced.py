from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
from scipy import ndimage
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import ConfusionMatrix, F1Score, JaccardIndex
from torchtyping import TensorType

from jawfrac.metrics import FracPrecision, FracRecall
import jawfrac.nn as nn
from jawfrac.models.common import (
    aggregate_sparse_predictions,
    aggregate_dense_predictions,
    batch_forward,
    fill_source_volume,
)
from jawfrac.models.jawfrac_linear import filter_connected_components
from jawfrac.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from jawfrac.visualization import (
    draw_confusion_matrix,
    draw_fracture_result,
    draw_positive_voxels,
)


def combine_linear_displaced_predictions(
    mandible: TensorType['D', 'H', 'W', torch.bool],
    linear: TensorType['D', 'H', 'W', torch.float32],
    displaced: TensorType['D', 'H', 'W', torch.float32],
    max_dist: float,
    linear_conf_threshold: float,
    linear_min_component_size: int,
    displaced_conf_threshold: float,
    displaced_min_component_size: int,
    mean_conf_threshold: float,
    mean_min_component_size: int,
    verbose: bool,
) -> TensorType['D', 'H', 'W', torch.int64]:
    # make a volume from the combination of both volumes
    out = filter_connected_components(
        mandible, (linear + displaced) / 2,
        mean_conf_threshold, mean_min_component_size, max_dist,
    )

    # filter connected components in each volume separately
    linear = filter_connected_components(
        mandible, linear,
        linear_conf_threshold, linear_min_component_size, max_dist,
    )
    displaced = filter_connected_components(
        mandible, displaced,
        displaced_conf_threshold, displaced_min_component_size, max_dist,
    )

    if verbose:
        draw_positive_voxels(mandible, linear, displaced, out)

    # add components only present in segmentation volume
    for i in range(1, linear.max() + 1):
        if not torch.any(out[linear == i]):
            continue

        labels = torch.unique(out[linear == i])
        label = labels[labels > 0][0]
        out[out == label] = 0
        out[linear == i] = label
    
    # add components only present in interpolation volume
    for i in range(1, displaced.max() + 1):
        if torch.any(out[displaced == i]):
            continue

        out[displaced == i] = out.max() + 1

    return out


class LinearDisplacedJawFracModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        first_stage: Dict[str, Any],
        second_stage: Dict[str, Any],
        third_stage: Dict[str, Any],
        x_axis_flip: bool,
        post_processing: Dict[str, Union[float, int]],
        batch_size: int=0,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        # initialize model stages
        self.mandible_net = nn.MandibleNet(
            num_classes=1, **first_stage, **model_cfg,
        )
        self.frac_net = nn.JawFracNet(
            num_classes=1,
            mandible_channels=self.mandible_net.out_channels,
            **second_stage,
            **model_cfg,
        )
        self.frac_cascade_net = nn.JawFracCascadeNet(
            num_classes=num_classes,
            mandible_channels=self.mandible_net.out_channels,
            fracture_channels=self.frac_net.out_channels,
            **third_stage,
            **model_cfg,
        )

        # initialize loss function
        self.criterion = nn.JawFracLoss(num_classes)

        self.confmat = ConfusionMatrix(num_classes=2)
        self.f1_1 = F1Score(num_classes=2, average='macro')
        self.f1_2 = F1Score(num_classes=2, average='macro')
        self.iou = JaccardIndex(num_classes=2)
        self.precision_metric = FracPrecision()
        self.recall_metric = FracRecall()

        self.num_classes = max(2, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.x_axis_flip = x_axis_flip
        self.post_processing = post_processing
        self.conf_thresh = post_processing['linear_conf_threshold']
        self.verbose = post_processing['verbose']

    def forward(
        self,
        x: TensorType['P', 1, 'size', 'size', 'size', torch.float32],       
    ) -> Tuple[
        TensorType['B', 'D', 'H', 'W', torch.float32],
        TensorType['B', torch.float32],
    ]:
        coords, mandible = self.mandible_net(x)
        masks = self.frac_net(x, coords, mandible)
        if isinstance(masks, tuple):
            masks = masks[1]
        logits = self.frac_cascade_net(x, coords, mandible, masks)

        if masks.dim() == 5:
            masks = masks[:, 0]

        return masks, logits

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 'size', 'size', 'size', torch.bool],
                TensorType['P', torch.bool],
            ],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, y = batch

        _, logits = self(x)

        loss = self.criterion(logits, y)
        self.log('loss/train', loss)

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 'size', 'size', 'size', torch.float32],
                TensorType['P', torch.int64],
            ],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        masks, logits = self(x)

        # losses
        loss = self.criterion(logits, y)

        # classification metric
        if self.num_classes == 2:
            classes = (logits >= 0).long()
        else:
            classes = logits.argmax(dim=1).clip(0, 1)
        self.f1_1(classes, (y[1] >= 1).long())

        # segmentation metrics
        masks = torch.sigmoid(masks) >= self.conf_thresh
        target = y[0] >= self.conf_thresh
        self.f1_2(masks[y[0] != -1].long(), target[y[0] != -1].long())
        
        # log metrics
        self.log_dict({
            'loss/val': loss,
            'f1/val_classes': self.f1_1,
            'f1/val_masks': self.f1_2,
        })

    def predict_volumes(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
    ) -> Tuple[
        TensorType['D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        # initialize generators that aggregate predictions
        mask_generator = aggregate_dense_predictions(
            patch_idxs=patch_idxs,
            out_shape=features.shape[1:],
        )
        class_generator = aggregate_sparse_predictions(
            pred_shape=(),
            patch_idxs=patch_idxs,
            out_shape=features.shape[1:],
        )

        # get initial empty predictions
        mask_pred = next(mask_generator)
        class_pred = next(class_generator)

        # run the model and aggregate its predictions
        batches = batch_forward(
            model=self,
            features=features,
            patch_idxs=patch_idxs,
            x_axis_flip=self.x_axis_flip,
            batch_size=self.batch_size,
        )
        for masks, logits in batches:
            for mask in masks:
                mask_pred -= mask_pred
                mask_pred += mask
                next(mask_generator)

            for logit in logits:
                class_pred -= class_pred
                class_pred += logit
                next(class_generator)

        # do any post-processing steps and compute probabilities
        linear = torch.sigmoid(next(mask_generator))
        displaced = torch.sigmoid(next(class_generator))

        return linear, displaced

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.bool],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.float32],
        ],
        batch_idx: int,
    ) -> Tuple[
        TensorType['D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        files = self.trainer.datamodule.test_dataset.files[batch_idx]
        print(files[0].parent.stem)

        features, mandible, patch_idxs, target = batch

        # predict binary segmentations
        linear, displaced = self.predict_volumes(features, patch_idxs)
        
        components = combine_linear_displaced_predictions(
            mandible, linear, displaced, **self.post_processing,
        )
        mask = components > 0

        if self.verbose:
            draw_fracture_result(mandible, mask, target >= self.conf_thresh)

        # compute metrics
        self.confmat(
            torch.any(mask)[None].long(),
            torch.any(target > 0)[None].long(),
        )

        self.precision_metric(mask, target >= self.conf_thresh)
        self.recall_metric(mask, target >= self.conf_thresh)


        for label in torch.unique(components)[1:]:
            if not torch.any(target[components == label] == 2):
                continue

            components[components == label] = 0
        mask = components > 0        
        target = (self.conf_thresh <= target) & (target != 2)
        self.iou(
            mask.long().flatten(),
            target.long().flatten(),
        )
        
        # log metrics
        self.log('iou/test', self.iou)
        self.log('precision/test', self.precision_metric)
        self.log('recall/test', self.recall_metric)

        return linear.cpu(), displaced.cpu()

    def test_epoch_end(
        self,
        outputs: List[Tuple[
            TensorType['D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.float32],
        ]]
    ) -> None:
        # torch.save(outputs, 'outputs.pth')

        draw_confusion_matrix(self.confmat.compute(), title='Binary scan')

    def predict_step(
        self,
        batch: Tuple[
            TensorType[1, 'D', 'H', 'W', torch.float32],
            TensorType['D', 'H', 'W', torch.bool],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType['D', 'H', 'W', torch.bool]:
        files = self.trainer.datamodule.predict_dataset.files[batch_idx]
        print(files[0].parent.stem)

        features, mandible, patch_idxs, affine, shape = batch

        # predict dense binary segmentations
        linear, displaced = self.predict_volumes(features, patch_idxs)

        # filter small connected components
        mask = combine_linear_displaced_predictions(
            mandible, linear, displaced, **self.post_processing,
        ) > 0

        # fill corresponding voxels in source volume
        mask = fill_source_volume(mask, affine, shape)
        
        return mask

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
