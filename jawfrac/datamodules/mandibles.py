from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType

from jawfrac.data.datasets.mandibles import MandibleSegDataset
import jawfrac.data.transforms as T
from jawfrac.datamodules.base import VolumeDataModule
from jawfrac.datamodules.jawfrac import JawFracDataModule


class MandibleSegDataModule(VolumeDataModule):

    def __init__(
        self,
        root: Union[str, Path],
        patch_size: int,
        gamma_adjust: bool,
        max_patches_per_scan: int,
        ignore_outside: bool,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        # use files functions from JawFrac when inferring for fracture data
        if 'fractures' in str(root) or str(root) == '/input':
            self._files = partial(JawFracDataModule._files, self)
            self._filter_files = partial(JawFracDataModule._filter_files, self)

        super().__init__(
            exclude=[],
            root=root,
            patch_size=patch_size,
            **dm_cfg,
        )

        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(),
            T.ToTensor(),
        )

        self.patch_size = patch_size
        self.gamma_adjust = gamma_adjust
        self.max_patches_per_scan = max_patches_per_scan
        self.ignore_outside = ignore_outside

    def _filter_files(self, pattern: str) -> List[Path]:
        files = super()._filter_files(pattern)
        
        overview_file = self.root / 'Fabian overview.csv'
        if not overview_file.exists():
            return files

        df = pd.read_csv(overview_file)
        df = df[pd.isna(df['Note']) & ~pd.isna(df['Complete'])]
        df = df[~df['Complete'].str.match(r'.*[,+]')]
        pseudonyms = df['Pseudonym'].tolist()

        dirs = list(map(lambda p: p.parent.stem, files))
        files = [f for f, d in zip(files, dirs) if d in pseudonyms]

        return files

    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files('**/image.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files))

        seg_files = self._filter_files('**/seg.nii.gz')

        return list(zip(scan_files, seg_files))

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.RelativePatchCoordinates(),
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan,
                    ignore_outside=self.ignore_outside,
                    rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(max_voxels=16, rng=rng),
                val_transforms,
                T.RandomGammaAdjust(rng=rng) if self.gamma_adjust else dict,
            )

            self.train_dataset = MandibleSegDataset(
                stage='fit',
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = MandibleSegDataset(
                stage='fit',
                files=val_files,
                transform=val_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'test':
            files = self._files('test')
            _, test_files, _ = self._split(files)

            self.test_dataset = MandibleSegDataset(
                stage='test',
                files=test_files[::-1],
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            all_files = self._files('predict')
            
            non_mandible_files = []
            for files in all_files:
                mandible_file = self.root / files[0].parent / 'mandible.nii.gz'
                if mandible_file.exists():
                    continue
                
                non_mandible_files.append(files)

            self.predict_dataset = MandibleSegDataset(
                stage='predict',
                files=all_files[:1],
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
        Tuple[
            TensorType['P', 3, torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.float32],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        coords = torch.cat(batch_dict['coords'])
        labels = torch.cat(batch_dict['masks'])

        return features, (coords, labels)

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['P', 3, 2, torch.int64],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        features = batch[0]['features']
        patch_idxs = batch[0]['patch_idxs']
        labels = batch[0]['labels']
        
        return features, patch_idxs, labels

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
