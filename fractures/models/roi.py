from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchtyping import TensorType

import fractures.nn as nn
from miccai.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)


class PatchROI(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model = nn.ConvNet(**model_cfg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

    def forward(
        self,
        x: TensorType['P', 'C', 'size', 'size', 'size', torch.float32],       
    ) -> TensorType['P', torch.float32]:
        x = self.model(x)

        return x

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            TensorType['P', torch.float32],
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
            TensorType['P', torch.float32],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        x = self(x)

        loss = self.criterion(x, y)

        self.log('loss/val', loss)

    def predict_step(
        self,
        batch: TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        batch_idx: int,
    ) -> TensorType['P', torch.float32]:
        x = self(batch)

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
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
