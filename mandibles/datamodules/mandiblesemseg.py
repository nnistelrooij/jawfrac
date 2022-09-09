from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

from miccai import PointTensor
from mandibles.data.datasets import MandibleSegDataset
import mandibles.data.transforms as T
from mandibles.datamodules.mandibleseg import MandibleSegDataModule


class MandibleSemSegDataModule(MandibleSegDataModule):


    def __init__(
        self,
        **dm_cfg,
    ):
        super().__init__(**dm_cfg)

        self.default_transforms = T.ToTensor()

    def setup(self, stage: Optional[str]=None):
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files = self._split(files)

            rng = np.random.default_rng(self.seed)
            train_transforms = T.Compose(
                T.RandomAxisFlip(axis=0, rng=rng),
                T.RandomScale(rng=rng),
                # T.RandomTranslate(max_dist=6, rng=rng),
                # T.RandomJitter(rng=rng),
                # RandomZAxisRotate(rng=rng),
                self.default_transforms,
            )

            self.train_dataset = MandibleSegDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = MandibleSegDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            self.pred_dataset = MandibleSegDataset(
                stage='predict',
                root=self.root,
                files=self._files('predict'),
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

    @property
    def num_channels(self) -> int:
        return 4

    @property
    def num_classes(self) -> int:
        return 2
    
    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Union[
        Tuple[PointTensor, PointTensor],
        Tuple[
            PointTensor,
            TensorType['N', 3, torch.float32],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}        

        coords = torch.cat(batch_dict['points'])
        feats = torch.cat(batch_dict['features'])
        batch_counts = torch.stack(batch_dict['point_count'])

        x = PointTensor(
            coordinates=coords,
            features=feats,
            batch_counts=batch_counts,
        )

        if 'labels' not in batch_dict:
            fg_coords = x.C.clone()

            downsample_idxs = batch_dict['downsample_idxs'][0]
            x = x[downsample_idxs]

            affine = batch_dict['affine'][0]
            shape = batch_dict['shape'][0]

            return x, fg_coords, affine, shape

        labels = torch.cat(batch_dict['labels'])
        labels = (labels == 2).long()

        y = PointTensor(
            coordinates=coords,
            features=labels,
            batch_counts=batch_counts,
        )

        return x, y

    def transfer_batch_to_device(
        self,
        batch: Union[
            Tuple[PointTensor, PointTensor],
            Tuple[
                PointTensor,
                TensorType['N', 3, torch.float32],
                TensorType[4, 4, torch.float32],
                TensorType[3, torch.int64],
            ],
        ],
        device: torch.device,
        dataloader_idx: int,
    ) -> Union[
        Tuple[PointTensor, PointTensor],
        Tuple[
            PointTensor,
            TensorType['N', 3, torch.float32],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
    ]:
        return tuple(map(lambda t: t.to(device), batch))
