from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchtyping import TensorType

from jawfrac.data.datasets import FracNetDataset
import jawfrac.data.transforms as T
from jawfrac.datamodules.base import VolumeDataModule
from jawfrac.datamodules.jawfrac import JawFracDataModule


class FracNetDataModule(VolumeDataModule):

    REGIONS = JawFracDataModule.REGIONS

    def __init__(
        self,
        linear: bool,
        displacements: bool,
        patch_size: int,
        expand_label: Dict[str, int],
        max_patches_per_scan: int,
        seed: int,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(
            exclude=[],
            patch_size=patch_size,
            expand_label=expand_label,
            seed=seed,
            **dm_cfg,
        )

        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(),
            T.ToTensor(),
        )

        self._filter_files = partial(JawFracDataModule._filter_files, self)
        self._split = partial(JawFracDataModule._split, self)
        self._split_patients = partial(JawFracDataModule._split_patients, self)
        self._split_controls = partial(JawFracDataModule._split_controls, self)

        self.linear = linear
        self.displacements = displacements
        self.patch_size = patch_size
        self.max_patches_per_scan = max_patches_per_scan

    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files('**/*main*.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files))

        frac_files = self._filter_files('**/label.nii.gz')

        return list(zip(scan_files, frac_files))

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan,
                    ignore_outside=False,
                    rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(
                    max_voxels=16, classes=[1], rng=rng
                ),
                val_transforms,
            )

            self.train_dataset = FracNetDataset(
                stage='fit',
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = FracNetDataset(
                stage='fit',
                files=val_files,
                transform=val_transforms,
                **self.dataset_cfg,
            )

            if self.trainer is not None:
                try:
                    self.trainer.logger.log_hyperparams({
                        'pre_transform': repr(self.train_dataset.pre_transform),
                        'train_transform': repr(train_transforms),
                        'val_transform': repr(val_transforms),
                    })
                except:
                    pass

        if stage is None or stage =='test':
            files = self._files('test')
            _, _, files = self._split(files)

            self.test_dataset = FracNetDataset(
                stage='test',
                files=files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            all_files = self._files('predict')

            non_frac_files = []
            for files in all_files:
                frac_file = self.root / files[0].parent / 'frac_pred_fracnet.nii.gz'
                if frac_file.exists():
                    continue
                
                non_frac_files.append(files)

            self.predict_dataset = FracNetDataset(
                stage='predict',
                files=non_frac_files[:1],
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
        TensorType['P', 'size', 'size', 'size', torch.int64],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        masks = torch.cat(batch_dict['masks'])

        return features, masks

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
