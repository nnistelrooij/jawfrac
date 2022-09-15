import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning.trainer.states import RunningStage
import torch
from torchtyping import TensorType

from mandibles.data.datasets import MandiblePatchSegDataset
import mandibles.data.transforms as T
from mandibles.datamodules.mandibleseg import MandibleSegDataModule


class MandiblePatchSegDataModule(MandibleSegDataModule):

    def __init__(
        self,
        max_patches_per_scan: int,
        seed: int,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(seed=seed, **dm_cfg)

        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(),
            T.ToTensor(),
        )

        self.max_patches_per_scan = max_patches_per_scan

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.RelativePatchCoordinates(),
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan, rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(
                    max_voxels=self.dataset_cfg['patch_size'] // 4, rng=rng,
                ),
                val_transforms,
            )

            self.train_dataset = MandiblePatchSegDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = MandiblePatchSegDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                transform=val_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage in ['test', 'predict']:
            stage = stage or 'test'
            files = self._files(stage)

            setattr(self, f'{stage}_dataset', MandiblePatchSegDataset(
                stage=stage,
                root=self.root,
                files=files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            ))

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 1

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        Tuple[
            TensorType['P', 3, torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.int64],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        coords = torch.cat(batch_dict['coords'])
        labels = torch.cat(batch_dict['labels'])

        return features, (coords, labels)

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['P', 3, 2, torch.int64],
        TensorType['P', 3, torch.float32],
        TensorType['D', 'H', 'W', torch.int64],
    ]:
        features = batch[0]['features']
        patch_idxs = batch[0]['patch_idxs']
        patch_coords = batch[0]['patch_coords']
        labels = batch[0]['labels']
        
        return features, patch_idxs, patch_coords, labels

    def predict_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['P', 3, 2, torch.int64],
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.int64],
    ]:
        features = batch[0]['features']
        patch_idxs = batch[0]['patch_idxs']
        affine = batch[0]['affine']
        shape = batch[0]['shape']

        return features, patch_idxs, affine, shape
