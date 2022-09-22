from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType

from jawfrac.data.datasets import JawFracDataset
import jawfrac.data.transforms as T
from jawfrac.datamodules.base import VolumeDataModule


class JawFracDataModule(VolumeDataModule):

    def __init__(
        self,
        patch_size: int,
        expand_label: Dict[str, int],
        max_patches_per_scan: int,
        seed: int,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(
            exclude=[
                '12', '103', '123', '125', '128',
                '130', '148', '155', '171', '186',
            ],
            patch_size=patch_size,
            expand_label=expand_label,
            seed=seed,
            **dm_cfg,
        )

        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(),
            T.ToTensor(),
        )

        self.count = sum(v > 0 for k, v in expand_label.items() if 'iters' in k)
        self.count = expand_label['smooth'] * self.count + 1
        self.patch_size = patch_size
        self.max_patches_per_scan = max_patches_per_scan

    def _filter_files(self, pattern: str) -> List[Path]:
        files = super(type(self), self)._filter_files(pattern)

        if self.filter == 'Controls':
            return files

        df = pd.read_csv(self.root / 'Sophie overview.csv')
        idxs = df.index[pd.isna(df['HU']) & pd.isna(df['Dislocated'])]

        patients = list(map(lambda p: int(p.parent.stem), files))
        files = [f for f, p in zip(files, patients) if p - 1 in idxs]

        return files

    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files('**/*main*.nii.gz')

        if not isinstance(self, JawFracDataModule):
            return list(zip(scan_files))

        mandible_files = self._filter_files('**/mandible.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files, mandible_files))
        
        frac_files = self._filter_files('**/Patient*segm*.nii.gz')
        frac_files += self._filter_files('**/*seg_normal*.nii.gz')
        frac_files += self._filter_files('**/Seg*.nii.gz')
        frac_files = sorted(frac_files)

        return list(zip(scan_files, mandible_files, frac_files))

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan, rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(
                    max_voxels=self.patch_size // 4, rng=rng,
                ),
                val_transforms,
            )

            self.train_dataset = JawFracDataset(
                stage='fit',
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = JawFracDataset(
                stage='fit',
                files=val_files,
                transform=val_transforms,
                **self.dataset_cfg,
            )

            self.trainer.logger.log_hyperparams({
                'pre_transform': str(self.train_dataset.pre_transform),
                'train_transform': str(self.train_dataset.transform),
                'val_transform': str(self.val_dataset.transform),
            })


        if stage is None or stage =='test':
            files = self._files('test')
            _, files, _ = self._split(files)

            self.test_dataset = JawFracDataset(
                stage='test',
                files=files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            files = self._files('predict')

            self.predict_dataset = JawFracDataset(
                stage='predict',
                files=files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

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
        TensorType['P', 'size', 'size', 'size', torch.float32],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        labels = torch.cat(batch_dict['labels']) / self.count

        return features, labels

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.bool],
        TensorType['P', 3, 2, torch.int64],
        TensorType['D', 'H', 'W', torch.bool],
    ]:
        features = batch[0]['features']
        mandible = batch[0]['mandible']
        patch_idxs = batch[0]['patch_idxs']
        labels = batch[0]['labels'] > 0
        
        return features, mandible, patch_idxs, labels

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
