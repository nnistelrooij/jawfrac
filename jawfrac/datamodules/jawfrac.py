from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from skmultilearn.model_selection import IterativeStratification
import torch
from torchtyping import TensorType

from jawfrac.data.datasets import JawFracDataset
import jawfrac.data.transforms as T
from jawfrac.datamodules.base import VolumeDataModule


class JawFracDataModule(VolumeDataModule):

    REGIONS = [
        'Coronoid',
        'Condyle',
        'Ramus',
        'Angulus',
        'Paramedian',
        'Median',
    ]

    def __init__(
        self,
        linear: bool,
        displacements: bool,
        patch_size: int,
        expand_label: Dict[str, int],
        gamma_adjust: bool,
        max_patches_per_scan: int,
        ignore_outside: bool,
        class_label_to_idx: List[int],
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
        self.linear = linear
        self.displacements = displacements
        self.patch_size = patch_size
        self.gamma_adjust = gamma_adjust
        self.max_patches_per_scan = max_patches_per_scan
        self.ignore_outside = ignore_outside
        self.class_label_to_idx = torch.tensor(class_label_to_idx)

    def _filter_files(self, pattern: str) -> List[Path]:
        files = super(type(self), self)._filter_files(pattern)

        patient_files = [f for f in files if 'Controls' not in str(f)]
        control_files = [f for f in files if 'Controls' in str(f)]

        df = pd.read_csv(self.root / 'Sophie overview 2.0.csv')
        mask = df['Note'].isna() & (
            (getattr(self, 'displacements', True) & df['Displaced'])
            |
            (getattr(self, 'linear', True) & df['Linear'])
        )
        idxs = df.index[mask]

        patients = [int(f.parent.stem) - 1 for f in patient_files]
        patient_files = [f for f, p in zip(patient_files, patients) if p in idxs]

        return patient_files + control_files

    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files('**/*main*.nii.gz')

        if not isinstance(self, JawFracDataModule):
            return list(zip(scan_files))

        mandible_files = self._filter_files('**/mandible2.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files, mandible_files))

        frac_files = self._filter_files('**/label.nii.gz')

        return list(zip(scan_files, mandible_files, frac_files))

    def _split(
        self,
        files: List[Tuple[Path, ...]],
    ) -> Tuple[
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
    ]:
        if self.val_size == 0 and self.test_size == 0:
            return files, [], []
            
        if self.val_size == 1:
            return [], files, []

        patient_files = self._split_patients(files)
        control_files = self._split_controls(files)
        files = [pfs + cfs for pfs, cfs in zip(patient_files, control_files)]

        return tuple(files)

    def _split_patients(
        self,
        files: List[Tuple[Path, ...]],
    ) -> Tuple[
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
    ]:
        patient_files = [fs for fs in files if 'Controls' not in str(fs[0])]
        if not patient_files:
            return [], [], []

        # take subsample of DataFrame comprised by given patient files
        df = pd.read_csv(self.root / 'Sophie overview 2.0.csv')
        idxs = [int(fs[0].parent.stem) - 1 for fs in patient_files]
        df = df.iloc[idxs].reset_index()

        # make one-hot vectors about fracture region and displacement
        one_hots = np.zeros((df.shape[0], len(self.REGIONS) + 2), dtype=int)
        for i, row in df.iterrows():
            regions = row['Locations'].split(', ')
            for region in regions:
                one_hots[i][self.REGIONS.index(region)] = 1

            one_hots[i][-2] = row['Displaced']
            one_hots[i][-1] = row['Linear']

        # make multi-label stratified split for train and validation/test files
        splitter = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[
                self.val_size + self.test_size,
                1 - self.val_size - self.test_size,
            ],
        )
        train_idxs, val_test_idxs = next(splitter.split(one_hots, one_hots))

        train_files = [patient_files[i] for i in train_idxs]
        val_test_files = [patient_files[i] for i in val_test_idxs]
        one_hots = one_hots[val_test_idxs]

        # make multi-label stratified split for validation and test files
        splitter = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[
                self.test_size / (self.val_size + self.test_size),
                self.val_size / (self.val_size + self.test_size),
            ],
        )
        val_idxs, test_idxs = next(splitter.split(one_hots, one_hots))

        val_files = [val_test_files[i] for i in val_idxs]
        test_files = [val_test_files[i] for i in test_idxs]

        return train_files, val_files, test_files

    def _split_controls(
        self,
        files: List[Tuple[Path, ...]],
    ) -> Tuple[
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
    ]:
        patient_files = [fs for fs in files if 'Controls' not in str(fs[0])]
        control_files = [fs for fs in files if 'Controls' in str(fs[0])]
        if not control_files:
            return [], [], []

        # randomly split control files among validation and test splits
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.test_size * len(patient_files),
            train_size=self.val_size * len(patient_files),
            random_state=self.seed,
        )
        val_idxs, test_idxs = next(splitter.split(control_files))

        val_files = [control_files[i] for i in val_idxs]
        test_files = [control_files[i] for i in test_idxs]

        return [], val_files, test_files

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan,
                    pos_classes=[1]*self.linear + [2]*self.displacements,
                    ignore_outside=self.ignore_outside,
                    rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(
                    max_voxels=self.patch_size // 4, rng=rng,
                ),
                val_transforms,
                T.RandomGammaAdjust(rng=rng) if self.gamma_adjust else dict,
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

            if self.trainer is not None: self.trainer.logger.log_hyperparams({
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
        return torch.unique(self.class_label_to_idx).shape[0]

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        Union[
            TensorType['P', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 'size', 'size', 'size', torch.float32],
                TensorType['P', torch.int64],
            ],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        masks = torch.cat(batch_dict['masks'])

        if not self.displacements:
            return features, masks
        
        classes = torch.cat(batch_dict['classes'])

        # do not provide segmentation feedback for displaced patches
        masks[classes == 2] = -1

        classes = self.class_label_to_idx[classes]

        return features, (masks, classes)

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.bool],
        TensorType['P', 3, 2, torch.int64],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        features = batch[0]['features']
        mandible = batch[0]['mandible']
        patch_idxs = batch[0]['patch_idxs']
        labels = batch[0]['labels']
        
        return features, mandible, patch_idxs, labels

    def predict_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['D', 'H', 'W', torch.bool],
        TensorType['P', 3, 2, torch.int64],
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.int64],
    ]:
        features = batch[0]['features']
        mandible = batch[0]['mandible']
        patch_idxs = batch[0]['patch_idxs']
        affine = batch[0]['affine']
        shape = batch[0]['shape']

        return features, mandible, patch_idxs, affine, shape
