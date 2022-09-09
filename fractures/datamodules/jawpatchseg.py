from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning.trainer.states import RunningStage
import torch
from torchtyping import TensorType

from fractures.data.datasets import JawFracDataset
import fractures.data.transforms as T
from fractures.datamodules.jawfrac import JawFracDataModule


class JawFracPatchDataModule(JawFracDataModule):

    def __init__(
        self,
        patch_size: int,
        bone_hu_threshold: int,
        seed: int,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(
            seed=seed,
            patch_size=patch_size,
            bone_hu_threshold=bone_hu_threshold,
            **dm_cfg,
        )

        self.rng = np.random.default_rng(seed=seed)

        self.default_transforms = T.Compose(
            T.IntensityAsFeatures(
                level=450.0, width=1100.0,
                low=-1.0, high=1.0,
            ),
            # T.RelativeXYZAsFeatures(),
            T.ToTensor(),
        )

        self.patch_size = patch_size
        self.bone_hu_threshold = bone_hu_threshold

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files = self._split(files)

            fit_transforms = T.Compose(
                T.ForegroundPatchIndices(patch_size=self.patch_size * 3 // 2),
                T.NegativePatchIndices(),
                T.PositivePatchIndices(patch_size=self.patch_size, rng=self.rng),
                T.ExpandLabel(
                    fg_iters=1,
                    all_iters=1,
                    fg_intensity_thresh=self.bone_hu_threshold,
                ),
                self.default_transforms,
            )

            self.train_dataset = JawFracDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                transform=fit_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = JawFracDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                transform=fit_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage in ['test', 'predict']:
            stage = stage or 'test'
            files = self._files(stage)

            setattr(self, f'{stage}_dataset', JawFracDataset(
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

    def crop_patches(
        self,    
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        labels: TensorType['D', 'H', 'W', torch.float32],
        pos_patch_idxs: TensorType['P', 3, 2, torch.int64],
        neg_patch_idxs: TensorType['P', 3, 2, torch.int64],
        **kwargs,
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        TensorType['P', 'size', 'size', 'size', torch.int64],
    ]:
        # sample as many random negative patches as there are positive patches
        neg_patch_idxs = neg_patch_idxs[torch.randperm(neg_patch_idxs.shape[0])]
        neg_patch_idxs = neg_patch_idxs[:pos_patch_idxs.shape[0]]

        patch_idxs = torch.cat((pos_patch_idxs, neg_patch_idxs))

        # crop patches from volume and stack
        feature_patches, label_patches = [], []
        for patch_idxs in patch_idxs.numpy():
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            label_patch = labels[slices]
            label_patch = (label_patch > 0).long()
            label_patches.append(label_patch)

            slices = (slice(None),) + slices
            feature_patches.append(features[slices])
        
        return torch.stack(feature_patches), torch.stack(label_patches)

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        TensorType['P', 'size', 'size', 'size', torch.int64],
    ]:
        features, labels = [], []
        for data_dict in batch:
            feature_patches, label_patches = self.crop_patches(**data_dict)

            features.append(feature_patches)
            labels.append(label_patches)

        return torch.cat(features), torch.cat(labels)

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['P', 3, 2, torch.int64],
        TensorType['D', 'H', 'W', torch.int64],
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

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Union[
        Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.int64],
        ],
        Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType['D', 'H', 'W', torch.int64],
        ],
        Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
    ]:
        if self.trainer.state.stage in [
            RunningStage.SANITY_CHECKING,
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
        ]:
            return self.fit_collate_fn(batch)
        elif self.trainer.state.stage == RunningStage.TESTING:
            return self.test_collate_fn(batch)
        elif self.trainer.state.stage == RunningStage.PREDICTING:
            return self.predict_collate_fn(batch)
            
        raise NotImplementedError(f'No collation for {self.trainer.state.stage}.')
